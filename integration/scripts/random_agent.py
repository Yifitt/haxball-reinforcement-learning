from __future__ import annotations

import argparse
import json
import time

from integration.controller.client import BridgeClient
from integration.controller.metrics import AgentMetrics
from integration.controller.policies import CONTROLLED_POLICY_NAMES, POLICY_NAMES, create_policy
from integration.controller.actions import get_action

DIAGNOSTIC_ACTIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 13)


def action_displacement_verified(action: int, delta_x: float, delta_y: float, minimum: float = 0.05) -> bool:
    keys = get_action(action)["keys"]
    checks = []
    if "left" in keys:
        checks.append(delta_x < -minimum)
    if "right" in keys:
        checks.append(delta_x > minimum)
    if "up" in keys:
        checks.append(delta_y < -minimum)
    if "down" in keys:
        checks.append(delta_y > minimum)
    return all(checks) if checks else True


def run_action_diagnostic(
    client: BridgeClient,
    metrics: AgentMetrics,
    clients: tuple[str, ...],
    *,
    hold_steps: int = 8,
) -> None:
    for client_role in clients:
        for action in DIAGNOSTIC_ACTIONS:
            state = client.receive_state(timeout=5.0)
            while not state.game_active:
                state = client.receive_state(timeout=5.0)
            player = getattr(state, client_role)
            if player is None:
                raise RuntimeError(f"action diagnostic missing {client_role} player state")
            start_x, start_y = player.x, player.y
            for _ in range(hold_steps):
                client.send_action(
                    tick_id=state.tick_id, client=client_role, action=action,
                    lifecycle_id=state.lifecycle_id,
                )
                metrics.record_action(client_role, action)
                state = client.receive_state(timeout=5.0)
                if not state.game_active:
                    break
            player = getattr(state, client_role)
            if player is None:
                raise RuntimeError(f"action diagnostic lost {client_role} player state")
            delta_x, delta_y = player.x - start_x, player.y - start_y
            verified = action_displacement_verified(action, delta_x, delta_y)
            print(
                f"action_diagnostic: client={client_role} "
                f"action={get_action(action)['name']} delta_x={delta_x:.3f} "
                f"delta_y={delta_y:.3f} verified={str(verified).lower()}"
            )
            if state.game_active:
                client.send_action(
                    tick_id=state.tick_id, client=client_role, action=0,
                    lifecycle_id=state.lifecycle_id,
                )
                metrics.record_action(client_role, 0)


def run_agent(
    *,
    duration: float,
    seed: int,
    bridge_url: str,
    policy: str = "random",
    opponent_policy: str = "stationary",
    action_hold_steps: int = 6,
    kick_probability: float = 0.10,
    action_diagnostic: bool = False,
    request_shutdown: bool = False,
    checkpoint: str | None = None,
    human_opponent: bool = False,
    public_room: bool = False,
) -> dict[str, float | int | None]:
    metrics = AgentMetrics()
    policies = {
        "controlled": create_policy(
            policy,
            "controlled",
            seed=seed,
            action_hold_steps=action_hold_steps,
            kick_probability=kick_probability,
            checkpoint_path=checkpoint,
        ),
    }
    if not human_opponent:
        policies["opponent"] = create_policy(
            opponent_policy,
            "opponent",
            seed=seed + 1,
            action_hold_steps=action_hold_steps,
            kick_probability=kick_probability,
        )
    opponent_control_required = opponent_policy != "stationary"
    if human_opponent:
        opponent_control_required = False
    checkpoint_policy = policies["controlled"] if policy == "checkpoint" else None
    if checkpoint_policy is not None:
        metadata = checkpoint_policy.metadata
        print(
            "checkpoint_loaded: "
            f"observation_version={metadata['observation_version']} "
            f"action_version={metadata['action_version']} "
            f"observation_size={metadata['observation_size']} actions={metadata['number_of_actions']}"
        )
    client = BridgeClient(
        bridge_url,
        opponent_control_required=opponent_control_required,
        human_opponent=human_opponent,
        public_room=public_room,
    )
    client.connect()
    print("python_controller_status: ready=true")
    try:
        client.wait_until_ready(timeout=60.0 if public_room else 86_400.0 if human_opponent else 60.0)
        mode = "public_human_queue" if public_room else "human_opponent" if human_opponent else (
            "dual_control" if opponent_control_required else "stationary_opponent")
        print(f"startup_barrier: mode={mode} ready=true")
        print("controller_action_loop_started: count=1")
        if action_diagnostic:
            diagnostic_clients = ("controlled", "opponent") if opponent_control_required else ("controlled",)
            run_action_diagnostic(client, metrics, diagnostic_clients)
            for selected_policy in policies.values():
                selected_policy.reset()
            metrics.reset_episode()
        deadline = time.monotonic() + duration
        observed_reset_count = client.reset_count
        observed_first_state_count = client.first_state_received_count
        inference_generation = -1
        action_return_generation = -1
        applied_generation = -1
        first_inference_count = 0
        first_action_return_count = 0
        first_action_applied_count = 0
        while time.monotonic() < deadline:
            try:
                state = client.receive_state(timeout=1.0)
            except TimeoutError:
                continue
            if client.reset_count != observed_reset_count:
                observed_reset_count = client.reset_count
                for selected_policy in policies.values():
                    selected_policy.reset()
                metrics.reset_episode()
                inference_generation = -1
                action_return_generation = -1
                applied_generation = -1
            generation = client.first_state_received_count
            if generation != observed_first_state_count:
                observed_first_state_count = generation
                print(
                    "lifecycle_event event=first_state_received "
                    f"count={generation} lifecycle_id={state.lifecycle_id} "
                    f"timestamp={client.first_state_received_timestamp_ms:.3f}"
                )
            for event in client.drain_applied_events():
                metrics.record_applied(event.get("client", "controlled"))
                if (event.get("client") == "controlled" and
                        event.get("lifecycle_id", state.lifecycle_id) == state.lifecycle_id and
                        applied_generation != generation):
                    applied_generation = generation
                    first_action_applied_count += 1
                    print(
                        "lifecycle_event event=first_action_applied_received "
                        f"count={first_action_applied_count} "
                        f"lifecycle_id={event.get('lifecycle_id', state.lifecycle_id)} "
                        f"timestamp={time.time() * 1000.0:.3f}"
                    )
            metrics.record_state(state.tick_id, state.timestamp)
            metrics.record_dual_state(state)
            if not state.game_active:
                continue
            controlled_action = policies["controlled"].select_action(state)
            if inference_generation != generation:
                inference_generation = generation
                first_inference_count += 1
                print(
                    "lifecycle_event event=first_inference_completed "
                    f"count={first_inference_count} lifecycle_id={state.lifecycle_id} "
                    f"timestamp={time.time() * 1000.0:.3f}"
                )
            client.send_action(
                tick_id=state.tick_id,
                client="controlled",
                action=controlled_action,
                lifecycle_id=state.lifecycle_id,
            )
            if action_return_generation != generation:
                action_return_generation = generation
                first_action_return_count += 1
                print(
                    "lifecycle_event event=first_action_returned "
                    f"count={first_action_return_count} lifecycle_id={state.lifecycle_id} "
                    f"timestamp={time.time() * 1000.0:.3f}"
                )
            metrics.record_action("controlled", controlled_action)
            if opponent_control_required:
                opponent_action = policies["opponent"].select_action(state)
                client.send_action(
                    tick_id=state.tick_id,
                    client="opponent",
                    action=opponent_action,
                    lifecycle_id=state.lifecycle_id,
                )
                metrics.record_action("opponent", opponent_action)
        if request_shutdown:
            client.shutdown()
    finally:
        for selected_policy in policies.values():
            selected_policy.reset()
        client.close()
    report = metrics.report()
    report["controlled_rejected_actions"] = client.rejected_actions["controlled"]
    report["opponent_rejected_actions"] = client.rejected_actions["opponent"]
    inference = getattr(policies["controlled"], "median_inference_ms", None)
    report["median_policy_inference_ms"] = inference() if callable(inference) else None
    if checkpoint_policy is not None:
        report["checkpoint_observation_version"] = checkpoint_policy.metadata["observation_version"]
        report["checkpoint_action_version"] = checkpoint_policy.metadata["action_version"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--policy", choices=CONTROLLED_POLICY_NAMES, default="random")
    parser.add_argument("--opponent-policy", choices=POLICY_NAMES, default="stationary")
    parser.add_argument("--action-hold-steps", type=int, default=6)
    parser.add_argument("--kick-probability", type=float, default=0.10)
    parser.add_argument("--action-diagnostic", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--human-opponent", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.policy == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required with --policy checkpoint")
    report = run_agent(
        duration=args.duration,
        seed=args.seed,
        bridge_url=args.bridge_url,
        policy=args.policy,
        opponent_policy=args.opponent_policy,
        action_hold_steps=args.action_hold_steps,
        kick_probability=args.kick_probability,
        action_diagnostic=args.action_diagnostic,
        request_shutdown=args.shutdown,
        checkpoint=args.checkpoint,
        human_opponent=args.human_opponent,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

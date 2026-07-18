from __future__ import annotations

import argparse
from collections import deque
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from integration.controller.actions import ACTIONS
from integration.controller.protocol import serialize_message, validate_state_message
from integration.scripts.random_agent import run_agent
from integration.controller.policies import create_policy

ROOT = Path(__file__).parents[2]
INTEGRATION = ROOT / "integration"
RELEASE_CHECKPOINT = ROOT / "checkpoints/releases/selfplay_v1/model.pt"
BOT_RECONNECT_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0)
ROOM_URL_PATTERN = re.compile(r"https?://(?:www\.)?haxball\.com/play\?\S+", re.IGNORECASE)


def resolve_startup_mode(*, public_room: bool, human_opponent: bool, opponent_policy: str) -> str:
    if public_room:
        return "public_human_queue"
    if human_opponent:
        return "human_opponent"
    if opponent_policy != "stationary":
        return "dual_control"
    return "stationary_opponent"


def resolve_room_options(
    *, public_room: bool, room_name: str | None, max_players: int | None,
    score_limit: int | None, time_limit: int | None,
    enable_player_queue: bool | None = None, matches_per_turn: int = 1,
    queue_afk_timeout: float = 0.0,
) -> dict[str, object]:
    queue_enabled = public_room if enable_player_queue is None else enable_player_queue
    if queue_enabled and not public_room:
        raise ValueError("player queue requires --public-room")
    if not isinstance(matches_per_turn, int) or matches_per_turn < 1 or matches_per_turn > 100:
        raise ValueError("matches_per_turn must be from 1 to 100")
    if queue_afk_timeout != 0:
        raise ValueError("queue_afk_timeout must be 0; AFK detection is disabled")
    options = {
        "public_room": public_room,
        "room_name": room_name or ("RL Bot | 1v1" if public_room else "RL Bot | Private Test"),
        "max_players": max_players if max_players is not None else (12 if public_room else 2),
        "score_limit": score_limit if score_limit is not None else (5 if public_room else 3),
        "time_limit": time_limit if time_limit is not None else (0 if public_room else 3),
        "enable_player_queue": queue_enabled,
        "matches_per_turn": matches_per_turn,
        "queue_afk_timeout": queue_afk_timeout,
    }
    ranges = (("max_players", 2, 30), ("score_limit", 0, 14), ("time_limit", 0, 60))
    for name, minimum, maximum in ranges:
        value = options[name]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be from {minimum} to {maximum}")
    return options


def sample_state() -> dict[str, object]:
    body = {"x": 0.0, "y": 1.0, "vx": 0.2, "vy": -0.1}
    return {
        "protocol_version": 1,
        "type": "state",
        "tick_id": 12,
        "timestamp": 1_700_000_000_000.0,
        "game_active": True,
        "controlled": {"id": 1, "team": 1, **body},
        "opponent": {"id": 2, "team": 2, **body},
        "ball": body,
        "match": {
            "controlled_side": "red",
            "controlled_score": 1,
            "opponent_score": 0,
            "elapsed_time": 10.0,
            "last_goal_event": None,
        },
    }


def offline_report() -> dict[str, object]:
    state = validate_state_message(sample_state())
    actions = [json.loads(serialize_message(
        "action", tick_id=state.tick_id, client="controlled", action=i)) for i in range(18)]
    rng = random.Random(0)
    random_actions = [rng.randrange(18) for _ in range(8)]
    return {
        "action_count": len(ACTIONS),
        "action_ids": [action["action"] for action in actions],
        "state_tick_id": state.tick_id,
        "state_valid": True,
        "protocol_version": 1,
        "random_actions": random_actions,
    }


def stationary_kickoff_report() -> dict[str, object]:
    def kickoff_state(lifecycle_id: int) -> dict[str, object]:
        def body(x: float) -> dict[str, float]:
            return {"x": x, "y": 0.0, "vx": 0.0, "vy": 0.0}

        return {
            "protocol_version": 1, "type": "state", "tick_id": 7,
            "timestamp": 1_700_000_000_000.0, "game_active": True,
            "controlled": {"id": 1, "team": 1, **body(-277.0)},
            "opponent": {"id": 2, "team": 2, **body(277.0)},
            "ball": body(0.0),
            "match": {
                "controlled_side": "red", "controlled_score": 0,
                "opponent_score": 0, "elapsed_time": 0.0,
                "last_goal_event": None,
            },
            "lifecycle_id": lifecycle_id, "state_sequence": 1,
            "forced_snapshot": True,
        }

    policy = create_policy(
        "checkpoint", "controlled", seed=0, checkpoint_path=RELEASE_CHECKPOINT,
    )
    states = [validate_state_message(kickoff_state(lifecycle)) for lifecycle in (1, 2)]
    actions = [policy.select_action(state) for state in states]
    if any(action == 0 for action in actions):
        raise RuntimeError("release checkpoint returned no movement at stationary kickoff")
    return {
        "stationary_kickoff": True,
        "checkpoint": str(RELEASE_CHECKPOINT.relative_to(ROOT)),
        "lifecycles": [state.lifecycle_id for state in states],
        "ticks": [state.tick_id for state in states],
        "actions": actions,
        "identical_opening_bodies": True,
        "inference_count": len(actions),
    }


def delayed_bridge_startup_report(delay_seconds: float = 0.2) -> dict[str, object]:
    node = os.environ.get("NODE", "node")
    host = "127.0.0.1"
    with socket.socket() as reservation:
        reservation.bind((host, 0))
        port = int(reservation.getsockname()[1])
    environment = os.environ.copy()
    environment.update({
        "HAXBALL_BRIDGE_HOST": host,
        "HAXBALL_BRIDGE_PORT": str(port),
        "HAXBALL_STARTUP_MODE": "stationary_opponent",
        "HAXBALL_PUBLIC_ROOM": "0",
        "HAXBALL_HUMAN_OPPONENT": "0",
        "HAXBALL_OPPONENT_CONTROL_REQUIRED": "0",
    })
    source = (
        "setTimeout(async () => {"
        "const { createBridgeServer } = await import('./bridge/websocket_server.js');"
        "createBridgeServer({"
        "host: process.env.HAXBALL_BRIDGE_HOST,"
        "port: Number(process.env.HAXBALL_BRIDGE_PORT),"
        "startupMode: process.env.HAXBALL_STARTUP_MODE"
        "});"
        f"}}, {int(delay_seconds * 1000)});"
    )
    bridge = BridgeChild.launch(
        [node, "--input-type=module", "-e", source],
        cwd=INTEGRATION,
        env=environment,
    )
    try:
        wait_for_bridge(bridge, host, port, timeout=5.0)
        return {
            "offline_delayed_bridge_startup": True,
            "delay_milliseconds": int(delay_seconds * 1000),
            "child_alive": bridge.poll() is None,
            "tcp_ready": True,
        }
    finally:
        bridge.stop()


def _redact_bridge_output(value: str, secrets: tuple[str, ...]) -> str:
    safe = ROOM_URL_PATTERN.sub("[REDACTED_ROOM_URL]", value)
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[REDACTED_SECRET]")
    return safe


class BridgeChild:
    """Own the bridge subprocess while continuously draining bounded diagnostics."""

    def __init__(
        self, process: subprocess.Popen[str], *, secrets: tuple[str, ...] = (),
    ) -> None:
        self.process = process
        self.secrets = secrets
        self.listening = threading.Event()
        self._stdout_lines: deque[str] = deque(maxlen=200)
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        for name, stream, destination, storage in (
            ("stdout", process.stdout, sys.stdout, self._stdout_lines),
            ("stderr", process.stderr, sys.stderr, self._stderr_lines),
        ):
            if stream is None:
                continue
            thread = threading.Thread(
                target=self._drain,
                args=(stream, destination, storage, name == "stdout"),
                name=f"bridge-{name}-drain",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    @classmethod
    def launch(
        cls, command: list[str], *, cwd: Path, env: dict[str, str],
    ) -> BridgeChild:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return cls(
            process,
            secrets=tuple(
                env.get(name, "")
                for name in (
                    "HAXBALL_HEADLESS_TOKEN", "HAXBALL_ROOM_PASSWORD", "HAXBALL_ROOM_URL",
                )
            ),
        )

    def _drain(
        self, stream: object, destination: object, storage: deque[str],
        observe_listening: bool,
    ) -> None:
        for raw_line in stream:  # type: ignore[union-attr]
            line = _redact_bridge_output(raw_line, self.secrets)
            with self._lock:
                storage.append(line)
            if observe_listening and line.startswith("bridge_listening:"):
                self.listening.set()
            print(line, end="", file=destination, flush=True)

    def poll(self) -> int | None:
        return self.process.poll()

    def stderr_text(self) -> str:
        with self._lock:
            return "".join(self._stderr_lines).strip()

    def wait_for_output_drain(self, timeout: float = 0.2) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout)

    def early_exit_error(self, exit_code: int) -> RuntimeError:
        self.wait_for_output_drain()
        stderr = self.stderr_text() or "(no stderr)"
        return RuntimeError(
            f"bridge exited before readiness: exit_code={exit_code} stderr={stderr}"
        )

    def stop(self) -> None:
        stop_process(self.process)
        self.wait_for_output_drain(timeout=1.0)


def wait_for_bridge(
    bridge: BridgeChild, host: str, port: int, timeout: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout
    delay = 0.05
    attempt = 0
    consecutive_accepts = 0
    last_tcp_accepting = False
    while True:
        exit_code = bridge.poll()
        if exit_code is not None:
            raise bridge.early_exit_error(exit_code)
        attempt += 1
        print(
            "local_connection_attempt: component=bridge_readiness "
            f"transport=tcp host={host} port={port} attempt={attempt}",
            flush=True,
        )
        try:
            with socket.create_connection((host, port), timeout=0.2):
                last_tcp_accepting = True
        except OSError:
            last_tcp_accepting = False

        if last_tcp_accepting and bridge.listening.is_set():
            consecutive_accepts += 1
            if consecutive_accepts >= 2:
                exit_code = bridge.poll()
                if exit_code is not None:
                    raise bridge.early_exit_error(exit_code)
                return
        else:
            consecutive_accepts = 0

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            exit_code = bridge.poll()
            if exit_code is not None:
                raise bridge.early_exit_error(exit_code)
            occupied = last_tcp_accepting and not bridge.listening.is_set()
            detail = " port_may_be_occupied_by_unrelated_process=true" if occupied else ""
            raise RuntimeError(
                "bridge readiness timed out: "
                f"host={host} port={port} child_alive=true "
                f"listening_log={str(bridge.listening.is_set()).lower()} "
                f"tcp_accepting={str(last_tcp_accepting).lower()}{detail}"
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 2.0, 0.5)


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class BotClientSupervisor:
    """Keep the controlled browser alive without touching the host room process."""

    def __init__(self, command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        self.command = command
        self.cwd = cwd
        self.env = env
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    def _spawn(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(self.command, cwd=self.cwd, env=self.env)

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("bot client supervisor already started")
            self._process = self._spawn()
        self._thread = threading.Thread(target=self._watch, name="bot-client-watchdog", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                process = self._process
            if process is None:
                return
            process.wait()
            if self._stop.is_set():
                return
            print("bot_disconnected reason=browser_process_exit", flush=True)
            replacement = None
            for attempt, delay in enumerate(BOT_RECONNECT_BACKOFF_SECONDS, start=1):
                if self._stop.wait(delay):
                    return
                print(
                    f"bot_reconnect_attempt attempt={attempt} delay_ms={int(delay * 1000)}",
                    flush=True,
                )
                try:
                    replacement = self._spawn()
                except OSError as error:
                    print(f"bot_reconnect_failed attempt={attempt} reason={error}", flush=True)
                    continue
                with self._lock:
                    self._process = replacement
                try:
                    replacement.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    break
                print(
                    f"bot_reconnect_failed attempt={attempt} reason=browser_process_exit",
                    flush=True,
                )
                replacement = None
            if replacement is None:
                # The room remains alive; begin another bounded health-check cycle.
                continue

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None:
            stop_process(process)
        if self._thread is not None:
            self._thread.join(timeout=6)


def real_smoke(
    duration: float,
    seed: int,
    headed: bool,
    policy: str,
    opponent_policy: str,
    action_hold_steps: int,
    kick_probability: float,
    action_diagnostic: bool,
    checkpoint: str | None,
    human_opponent: bool = False,
    room_options: dict[str, object] | None = None,
) -> dict[str, object]:
    if not os.environ.get("HAXBALL_HEADLESS_TOKEN"):
        raise RuntimeError("HAXBALL_HEADLESS_TOKEN is required")
    node = os.environ.get("NODE", "node")
    bridge_host = os.environ.get("HAXBALL_BRIDGE_HOST", "127.0.0.1")
    bridge_connect_host = (
        "127.0.0.1" if bridge_host in {"0.0.0.0", "::", "[::]"} else bridge_host
    )
    port = int(os.environ.get("HAXBALL_BRIDGE_PORT", "8765"))
    processes: list[subprocess.Popen[bytes]] = []
    bot_supervisor: BotClientSupervisor | None = None
    bridge: BridgeChild | None = None
    try:
        deployment_environment = os.environ.copy()
        deployment_environment["HAXBALL_OPPONENT_POLICY"] = opponent_policy
        deployment_environment["HAXBALL_HUMAN_OPPONENT"] = "1" if human_opponent else "0"
        startup_mode = resolve_startup_mode(
            public_room=bool(room_options and room_options["public_room"]),
            human_opponent=human_opponent,
            opponent_policy=opponent_policy,
        )
        deployment_environment["HAXBALL_STARTUP_MODE"] = startup_mode
        deployment_environment["HAXBALL_BRIDGE_HOST"] = bridge_host
        deployment_environment["HAXBALL_BRIDGE_PORT"] = str(port)
        deployment_environment["HAXBALL_BRIDGE_URL"] = (
            f"ws://{bridge_connect_host}:{port}"
        )
        deployment_environment["HAXBALL_OPPONENT_CONTROL_REQUIRED"] = (
            "1" if opponent_policy != "stationary" and not human_opponent else "0"
        )
        if room_options:
            deployment_environment.update({
                "HAXBALL_PUBLIC_ROOM": "1" if room_options["public_room"] else "0",
                "HAXBALL_ROOM_NAME": str(room_options["room_name"]),
                "HAXBALL_MAX_PLAYERS": str(room_options["max_players"]),
                "HAXBALL_SCORE_LIMIT": str(room_options["score_limit"]),
                "HAXBALL_TIME_LIMIT": str(room_options["time_limit"]),
                "HAXBALL_ENABLE_PLAYER_QUEUE": "1" if room_options["enable_player_queue"] else "0",
                "HAXBALL_MATCHES_PER_TURN": str(room_options["matches_per_turn"]),
                "HAXBALL_QUEUE_AFK_TIMEOUT": str(room_options["queue_afk_timeout"]),
            })
        bridge = BridgeChild.launch(
            [node, "bridge/websocket_server.js"], cwd=INTEGRATION,
            env=deployment_environment,
        )
        wait_for_bridge(bridge, bridge_connect_host, port)
        host_command = [node, "headless_host/launch_host.js"]
        client_command = [node, "browser/launch_clients.js"]
        if headed:
            host_command.append("--headed")
            client_command.append("--headed")
        processes.append(subprocess.Popen(
            host_command, cwd=INTEGRATION, env=deployment_environment))
        if room_options and room_options["public_room"]:
            bot_supervisor = BotClientSupervisor(
                client_command, cwd=INTEGRATION, env=deployment_environment)
            bot_supervisor.start()
        else:
            processes.append(subprocess.Popen(
                client_command,
                cwd=INTEGRATION,
                env=deployment_environment,
            ))
        exit_code = bridge.poll()
        if exit_code is not None:
            raise bridge.early_exit_error(exit_code)
        try:
            report = run_agent(
                duration=duration,
                seed=seed,
                bridge_url=deployment_environment["HAXBALL_BRIDGE_URL"],
                policy=policy,
                opponent_policy=opponent_policy,
                action_hold_steps=action_hold_steps,
                kick_probability=kick_probability,
                action_diagnostic=action_diagnostic,
                request_shutdown=True,
                checkpoint=checkpoint,
                human_opponent=human_opponent,
                public_room=bool(room_options and room_options["public_room"]),
            )
        except Exception as error:
            exit_code = bridge.poll()
            if exit_code is not None:
                raise bridge.early_exit_error(exit_code) from error
            raise
        return {"real_room": True, **report}
    finally:
        if bot_supervisor is not None:
            bot_supervisor.stop()
        for process in reversed(processes):
            stop_process(process)
        if bridge is not None:
            bridge.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline-stationary-kickoff", action="store_true")
    parser.add_argument("--offline-delayed-bridge-startup", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--policy", choices=("random", "persistent-random", "chase", "checkpoint"),
        default="random",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=("stationary", "random", "persistent-random", "chase"),
        default="stationary",
    )
    parser.add_argument("--action-hold-steps", type=int, default=6)
    parser.add_argument("--kick-probability", type=float, default=0.10)
    parser.add_argument("--action-diagnostic", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--human-opponent", action="store_true")
    parser.add_argument("--public-room", action="store_true")
    parser.add_argument("--room-name")
    parser.add_argument("--max-players", type=int)
    queue_group = parser.add_mutually_exclusive_group()
    queue_group.add_argument("--enable-player-queue", dest="enable_player_queue", action="store_true")
    queue_group.add_argument("--disable-player-queue", dest="enable_player_queue", action="store_false")
    parser.set_defaults(enable_player_queue=None)
    parser.add_argument("--matches-per-turn", type=int, default=1)
    parser.add_argument("--queue-afk-timeout", type=float, default=0.0)
    parser.add_argument("--score-limit", type=int)
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.public_room:
        args.human_opponent = True
        args.policy = "checkpoint"
        args.opponent_policy = "stationary"
        if args.checkpoint and Path(args.checkpoint).resolve() != RELEASE_CHECKPOINT.resolve():
            parser.error("--public-room always uses checkpoints/releases/selfplay_v1/model.pt")
        args.checkpoint = str(RELEASE_CHECKPOINT)
    if args.policy == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required with --policy checkpoint")
    try:
        room_options = resolve_room_options(
            public_room=args.public_room, room_name=args.room_name,
            max_players=args.max_players, score_limit=args.score_limit,
            time_limit=args.time_limit, enable_player_queue=args.enable_player_queue,
            matches_per_turn=args.matches_per_turn,
            queue_afk_timeout=args.queue_afk_timeout)
    except ValueError as error:
        parser.error(str(error))
    startup_mode = resolve_startup_mode(
        public_room=args.public_room,
        human_opponent=args.human_opponent,
        opponent_policy=args.opponent_policy,
    )
    print(
        "startup_config: "
        f"mode={startup_mode} "
        f"cli_public_room={str(args.public_room).lower()} "
        f"cli_human_opponent={str(args.human_opponent).lower()} "
        f"config_player_queue={str(room_options['enable_player_queue']).lower()} "
        f"config_opponent_policy={args.opponent_policy}",
        file=sys.stderr,
    )
    offline_modes = sum((
        args.offline,
        args.offline_stationary_kickoff,
        args.offline_delayed_bridge_startup,
    ))
    if offline_modes > 1:
        parser.error("choose only one offline smoke mode")
    if args.offline_delayed_bridge_startup:
        report = delayed_bridge_startup_report()
    elif args.offline_stationary_kickoff:
        report = stationary_kickoff_report()
    elif args.offline:
        report = offline_report()
    elif args.dry_run:
        report = {
            "dry_run": True, "authenticated_room_launched": False,
            "checkpoint": args.checkpoint, "human_opponent": args.human_opponent,
            "startup_mode": startup_mode,
            **room_options,
        }
    else:
        report = real_smoke(
            args.duration,
            args.seed,
            args.headed,
            args.policy,
            args.opponent_policy,
            args.action_hold_steps,
            args.kick_probability,
            args.action_diagnostic,
            args.checkpoint,
            args.human_opponent,
            room_options,
        )
    if args.json:
        print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"real_haxball_smoke_error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

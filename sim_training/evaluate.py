from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

from policy_contract.checkpoint_contract import PortablePolicy, load_checkpoint
from policy_contract.chase_contract import kick_request_masks
from sim_training.curriculum import configuration_for_stage
from sim_training.env_factory import make_training_env
from sim_training.goal_attribution import GoalAttributionTracker
from sim_training.opponent_pool import OpponentPool, OpponentPoolConfiguration
from sim_training.policy import deterministic_bins_with_entropy
from sim_training.self_play_pool import load_active_pool

HELD_OUT_SEEDS = (104_729, 130_363, 155_921, 196_613)
MIN_EVALUATION_EPISODES = 100


def _score_record() -> dict[str, int]:
    return {
        "episodes": 0, "goals_for": 0, "goals_against": 0, "wins": 0,
        "agent_normal_goals": 0, "opponent_normal_goals": 0,
        "own_goals": 0, "agent_own_goals": 0, "opponent_own_goals": 0,
    }


def _finish_scores(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(groups.items()):
        episodes = values["episodes"]
        result[name] = {
            **values,
            "goal_difference": values["goals_for"] - values["goals_against"],
            "win_rate": values["wins"] / episodes if episodes else 0.0,
        }
    return result


def _evaluate_batch(
    model: PortablePolicy,
    *,
    count: int,
    seed: int,
    action_repeat: int,
    max_decisions: int,
    random_learner_side: bool,
    opponent_configuration: OpponentPoolConfiguration,
    opponent_models: tuple[object, ...],
    opponent_labels: tuple[str, ...],
    forced_learner_sides: np.ndarray | None = None,
    forced_opponent_labels: np.ndarray | None = None,
) -> dict[str, object]:
    env = make_training_env(
        count, action_repeat=action_repeat, seed=seed)
    observations = env.reset()
    pool = OpponentPool(
        count, seed=seed + 1, configuration=opponent_configuration,
        self_play_models=opponent_models, self_play_labels=opponent_labels)
    side_rng = np.random.default_rng(seed + 2)
    learner_sides = np.asarray(forced_learner_sides, dtype=np.int64).copy() if forced_learner_sides is not None else (
        side_rng.integers(0, 2, count, dtype=np.int64)
        if random_learner_side else np.zeros(count, dtype=np.int64)
    )
    if learner_sides.shape != (count,) or not np.isin(learner_sides, (0, 1)).all():
        raise ValueError("forced learner sides must be a Red/Blue value for every environment")
    if forced_opponent_labels is not None:
        pool.force_assignments(forced_opponent_labels)
    opponent_sides = 1 - learner_sides
    initial_opponents = pool.labels()
    initial_kickoffs = env.state_mutator.kickoff_labels().copy()
    active = np.ones(count, dtype=bool)
    lengths = np.zeros(count, dtype=np.int64)
    previous_invalid = np.zeros(count, dtype=bool)
    attribution = GoalAttributionTracker(count)
    invalid_streak = np.zeros(count, dtype=np.int64)
    maximum_invalid_streak = np.zeros(count, dtype=np.int64)
    action_ids: set[int] = set()
    requested_total = valid_total = invalid_total = held_invalid_total = successful_total = 0
    entropy_total = reward_total = 0.0
    decision_total = 0
    groups_opponent: dict[str, dict[str, int]] = defaultdict(_score_record)
    groups_kickoff: dict[str, dict[str, int]] = defaultdict(_score_record)
    groups_side: dict[str, dict[str, int]] = defaultdict(_score_record)
    agent_normal_goals = opponent_normal_goals = 0
    goals_for = goals_against = agent_own_goals = opponent_own_goals = wins = 0
    action_buffer = np.empty((count, 2, 3), dtype=np.int64)
    rows = np.arange(count)

    try:
        for _ in range(max_decisions):
            learner_obs = observations[rows, learner_sides]
            opponent_obs = observations[rows, opponent_sides]
            learner, entropy = deterministic_bins_with_entropy(model, learner_obs)
            opponent = pool.actions(env.prev_state, opponent_sides, opponent_obs)
            requested, valid, invalid, held_invalid = kick_request_masks(
                learner, env.prev_state, player_index=learner_sides,
                previous_invalid=previous_invalid)
            opponent_requested, opponent_valid, _, _ = kick_request_masks(
                opponent, env.prev_state, player_index=opponent_sides)
            learner_valid = valid & active
            opponent_valid = opponent_valid & opponent_requested & active
            attribution.record_masks(learner_valid, opponent_valid, learner_sides)
            before_ball_velocity = env.prev_state.ball_vel.copy()

            selected = learner[active]
            action_ids.update((selected[:, 0] * 6 + selected[:, 1] * 2 + selected[:, 2]).tolist())
            requested_total += int((requested & active).sum())
            valid_total += int((valid & active).sum())
            invalid_total += int((invalid & active).sum())
            held_invalid_total += int((held_invalid & active).sum())
            invalid_streak = np.where(invalid & active, invalid_streak + 1, 0)
            maximum_invalid_streak = np.maximum(maximum_invalid_streak, invalid_streak)
            entropy_total += float(entropy[active].sum())
            decision_total += int(active.sum())
            lengths[active] += 1

            action_buffer[rows, learner_sides] = learner
            action_buffer[rows, opponent_sides] = opponent
            observations, rewards, terminated, truncated = env.step(action_buffer)
            velocity_change = np.linalg.norm(env.prev_state.ball_vel - before_ball_velocity, axis=1)
            successful_total += int((valid & active & (velocity_change > 1e-5)).sum())
            learner_rewards = rewards[rows, learner_sides]
            reward_total += float(learner_rewards[active].sum())
            done = (terminated | truncated) & active
            previous_invalid = invalid
            previous_invalid[done] = False
            if not done.any():
                continue

            conceding = env.prev_state.scored
            attributed = attribution.attribute(conceding, learner_sides)
            for index in np.flatnonzero(done):
                learner_team = 2 if learner_sides[index] == 0 else 4
                opponent_team = 4 if learner_sides[index] == 0 else 2
                learner_scored = conceding[index] == opponent_team
                opponent_scored = conceding[index] == learner_team
                agent_normal = bool(attributed["agent_normal_goals"][index])
                opponent_normal = bool(attributed["opponent_normal_goals"][index])
                agent_own = bool(attributed["agent_own_goals"][index])
                opponent_own = bool(attributed["opponent_own_goals"][index])
                own_goal = agent_own or opponent_own
                won = bool(learner_scored)
                goals_for += int(learner_scored)
                goals_against += int(opponent_scored)
                wins += int(won)
                agent_own_goals += int(agent_own)
                opponent_own_goals += int(opponent_own)
                agent_normal_goals += int(agent_normal)
                opponent_normal_goals += int(opponent_normal)
                for grouping, label in (
                    (groups_opponent, str(initial_opponents[index])),
                    (groups_kickoff, str(initial_kickoffs[index])),
                    (groups_side, "red" if learner_sides[index] == 0 else "blue"),
                ):
                    row = grouping[label]
                    row["episodes"] += 1
                    row["goals_for"] += int(learner_scored)
                    row["goals_against"] += int(opponent_scored)
                    row["wins"] += int(won)
                    row["agent_normal_goals"] += int(agent_normal)
                    row["opponent_normal_goals"] += int(opponent_normal)
                    row["own_goals"] += int(own_goal)
                    row["agent_own_goals"] += int(agent_own)
                    row["opponent_own_goals"] += int(opponent_own)
            active[done] = False
            attribution.reset(done)
            if not active.any():
                break
        if active.any():
            raise RuntimeError(
                f"held-out evaluation did not finish {int(active.sum())} episodes within "
                f"{max_decisions} decisions")
        return {
            "episodes": count, "goals_for": goals_for, "goals_against": goals_against,
            "agent_normal_goals": agent_normal_goals,
            "opponent_normal_goals": opponent_normal_goals,
            "agent_own_goals": agent_own_goals,
            "opponent_own_goals": opponent_own_goals, "wins": wins,
            "reward_total": reward_total, "decision_total": decision_total,
            "episode_length_total": int(lengths.sum()), "requested_total": requested_total,
            "valid_total": valid_total, "invalid_total": invalid_total,
            "held_invalid_total": held_invalid_total, "entropy_total": entropy_total,
            "successful_total": successful_total,
            "maximum_invalid_streak": int(maximum_invalid_streak.max(initial=0)),
            "action_ids": action_ids, "by_opponent": groups_opponent,
            "by_kickoff": groups_kickoff, "by_side": groups_side,
        }
    finally:
        del env


def evaluate_model(
    model: PortablePolicy,
    *,
    episodes: int = 128,
    n_envs: int = 64,
    action_repeat: int,
    max_decisions: int = 300,
    allow_fewer_episodes: bool = False,
    random_learner_side: bool = False,
    opponent_configuration: OpponentPoolConfiguration | None = None,
    opponent_models: tuple[object, ...] = (),
    opponent_labels: tuple[str, ...] = (),
) -> dict[str, object]:
    if episodes < MIN_EVALUATION_EPISODES and not allow_fewer_episodes:
        raise ValueError(f"evaluation requires at least {MIN_EVALUATION_EPISODES} episodes")
    if n_envs < 1 or max_decisions < 1:
        raise ValueError("n_envs and max_decisions must be positive")
    if bool(opponent_models) != bool(opponent_labels):
        raise ValueError("evaluation checkpoint models and labels must be provided together")
    configuration = opponent_configuration or configuration_for_stage(1)

    totals: dict[str, float | int] = defaultdict(float)
    action_ids: set[int] = set()
    opponent_groups: dict[str, dict[str, int]] = defaultdict(_score_record)
    kickoff_groups: dict[str, dict[str, int]] = defaultdict(_score_record)
    side_groups: dict[str, dict[str, int]] = defaultdict(_score_record)
    remaining = episodes
    batch_index = 0
    batch_capacity = min(n_envs, max(1, math.ceil(episodes / len(HELD_OUT_SEEDS))))
    used_seeds: list[int] = []
    while remaining:
        count = min(batch_capacity, remaining)
        seed = HELD_OUT_SEEDS[batch_index % len(HELD_OUT_SEEDS)]
        if seed not in used_seeds:
            used_seeds.append(seed)
        batch = _evaluate_batch(
            model, count=count, seed=seed, action_repeat=action_repeat,
            max_decisions=max_decisions, random_learner_side=random_learner_side,
            opponent_configuration=configuration, opponent_models=opponent_models,
            opponent_labels=opponent_labels)
        for key in (
            "episodes", "goals_for", "goals_against", "agent_own_goals",
            "opponent_own_goals", "agent_normal_goals", "opponent_normal_goals",
            "wins", "reward_total", "decision_total",
            "episode_length_total", "requested_total", "valid_total", "invalid_total",
            "held_invalid_total", "successful_total", "entropy_total",
        ):
            totals[key] += batch[key]  # type: ignore[operator]
        totals["maximum_invalid_streak"] = max(
            totals["maximum_invalid_streak"], batch["maximum_invalid_streak"])
        action_ids.update(batch["action_ids"])  # type: ignore[arg-type]
        for source_name, target in (
            ("by_opponent", opponent_groups), ("by_kickoff", kickoff_groups),
            ("by_side", side_groups),
        ):
            for label, row in batch[source_name].items():  # type: ignore[union-attr]
                for key, value in row.items():
                    target[label][key] += value
        remaining -= count
        batch_index += 1

    decisions = int(totals["decision_total"])
    requests = int(totals["requested_total"])
    finished_opponents = _finish_scores(opponent_groups)
    checkpoint_scores = {
        label: finished_opponents[label] for label in opponent_labels
        if label in finished_opponents
    }
    hard_rows = [row for label, row in opponent_groups.items() if label not in opponent_labels]
    hard_total = _score_record()
    for row in hard_rows:
        for key, value in row.items():
            hard_total[key] += value
    wins = int(totals["wins"])
    losses = int(totals["goals_against"])
    draws = episodes - wins - losses
    return {
        "evaluation_protocol": (
            "held-out-random-side-self-play-v1" if random_learner_side
            else "held-out-randomized-pool-v1"),
        "held_out_seeds": used_seeds, "episodes": episodes,
        "n_envs_per_batch": n_envs, "goals_for": int(totals["goals_for"]),
        "goals_against": int(totals["goals_against"]),
        "goal_difference": int(totals["goals_for"] - totals["goals_against"]),
        "agent_normal_goals": int(totals["agent_normal_goals"]),
        "opponent_normal_goals": int(totals["opponent_normal_goals"]),
        "own_goals": int(totals["agent_own_goals"] + totals["opponent_own_goals"]),
        "agent_own_goals": int(totals["agent_own_goals"]),
        "opponent_own_goals": int(totals["opponent_own_goals"]),
        "win_rate": float(totals["wins"] / episodes),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "draw_rate": draws / episodes,
        "loss_rate": losses / episodes,
        "mean_reward": float(totals["reward_total"] / decisions),
        "mean_episode_length": float(totals["episode_length_total"] / episodes),
        "average_match_duration_seconds": float(
            totals["episode_length_total"] * action_repeat / 60.0 / episodes),
        "timeout_count": draws,
        "crash_count": 0,
        "kick_fraction": requests / decisions,
        "kick_request_fraction": requests / decisions,
        "valid_kick_fraction_of_requests": (
            int(totals["valid_total"]) / requests if requests else 0.0),
        "valid_kick_fraction_of_decisions": int(totals["valid_total"]) / decisions,
        "invalid_kick_fraction_of_requests": (
            int(totals["invalid_total"]) / requests if requests else 0.0),
        "invalid_kick_fraction_of_decisions": int(totals["invalid_total"]) / decisions,
        "held_invalid_kick_fraction_of_decisions": int(totals["held_invalid_total"]) / decisions,
        "held_invalid_kick_fraction": int(totals["held_invalid_total"]) / decisions,
        "successful_contact_fraction": (
            int(totals["successful_total"]) / requests if requests else 0.0),
        "consecutive_invalid_kick_streak": int(totals["maximum_invalid_streak"]),
        "unique_action_count": len(action_ids),
        "policy_entropy": float(totals["entropy_total"] / decisions),
        "score_by_opponent_type": finished_opponents,
        "score_by_checkpoint_generation": checkpoint_scores,
        "score_vs_hard_rule_based": _finish_scores({"all_hard": hard_total})["all_hard"],
        "score_by_kickoff_configuration": _finish_scores(kickoff_groups),
        "score_by_learner_side": _finish_scores(side_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--self-play-pool-metadata")
    args = parser.parse_args()
    model, metadata = load_checkpoint(args.checkpoint)
    models: tuple[object, ...] = ()
    labels: tuple[str, ...] = ()
    random_side = False
    configuration = configuration_for_stage(1)
    if args.self_play_pool_metadata:
        models, labels, _ = load_active_pool(args.self_play_pool_metadata)
        configuration = configuration_for_stage(4)
        random_side = True
    report = evaluate_model(
        model, episodes=args.episodes, n_envs=args.n_envs,
        max_decisions=args.max_decisions, action_repeat=int(metadata["action_repeat"]),
        random_learner_side=random_side, opponent_configuration=configuration,
        opponent_models=models, opponent_labels=labels)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

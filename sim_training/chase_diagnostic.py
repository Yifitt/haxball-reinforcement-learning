from __future__ import annotations

import argparse
import json

import numpy as np

from sim_training.env_factory import make_training_env
from sim_training.policy import chase_bins

RED = 2
BLUE = 4


def evaluate_chase_own_goals(
    *, n_envs_per_team: int = 16, decisions: int = 512, action_repeat: int = 8
) -> dict[str, int]:
    """Run each team chase against a stationary opponent and count concessions as own goals."""
    n_envs = 2 * n_envs_per_team
    env = make_training_env(n_envs, action_repeat=action_repeat)
    env.reset()
    red_mask = np.arange(n_envs) < n_envs_per_team
    chaser_team = np.where(red_mask, RED, BLUE)
    stationary = np.ones((n_envs, 3), dtype=np.int64)
    stationary[:, 2] = 0
    goals_for = goals_against = own_goals = 0
    try:
        for _ in range(decisions):
            red = chase_bins(env.prev_state, 0)
            blue = chase_bins(env.prev_state, 1)
            actions = np.stack(
                (np.where(red_mask[:, None], red, stationary),
                 np.where(red_mask[:, None], stationary, blue)),
                axis=1,
            )
            _, _, terminated, _ = env.step(actions)
            if terminated.any():
                scored = env.prev_state.scored[terminated]
                active_team = chaser_team[terminated]
                conceded = scored == active_team
                goals_against += int(conceded.sum())
                own_goals += int(conceded.sum())
                goals_for += int((~conceded).sum())
        return {
            "decisions_per_team": n_envs_per_team * decisions,
            "total_chase_decisions": n_envs * decisions,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "own_goals": own_goals,
        }
    finally:
        del env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs-per-team", type=int, default=16)
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--action-repeat", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(evaluate_chase_own_goals(
        n_envs_per_team=args.n_envs_per_team,
        decisions=args.decisions,
        action_repeat=args.action_repeat,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Run a small deterministic random-agent smoke test (no training)."""

from __future__ import annotations

import argparse

import numpy as np

from haxball_env import HaxBallEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    env = HaxBallEnv()
    rng = np.random.default_rng(args.seed)
    observation, _ = env.reset(seed=args.seed)
    episodes = 0
    goals = 0
    for _ in range(args.steps):
        action = int(rng.integers(env.action_space.n))
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(observation)) or not np.isfinite(reward):
            raise RuntimeError("non-finite value encountered")
        goals += int(info["goal"] is not None)
        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset(seed=args.seed + episodes)
    env.close()
    print(f"ok: steps={args.steps} episodes={episodes} goals={goals}")


if __name__ == "__main__":
    main()

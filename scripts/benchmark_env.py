"""Short single-environment CPU throughput benchmark with rendering disabled."""

from __future__ import annotations

import argparse
import time

import numpy as np

from haxball_env import HaxBallEnv


def state_is_finite(env: HaxBallEnv) -> bool:
    arrays = (
        env.state.player_positions,
        env.state.player_velocities,
        env.state.ball_position,
        env.state.ball_velocity,
        env.state.kick_cooldowns,
    )
    return all(np.all(np.isfinite(array)) for array in arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    if args.steps < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("--steps and --episodes must be positive")

    env = HaxBallEnv(render_mode=None, opponent_mode=args.opponent)
    rng = np.random.default_rng(args.seed)
    observation, _ = env.reset(seed=args.seed)
    completed_steps = completed_episodes = goals = 0
    non_finite = not np.all(np.isfinite(observation)) or not state_is_finite(env)

    start = time.perf_counter()
    while completed_steps < args.steps and not non_finite:
        observation, reward, terminated, truncated, info = env.step(
            int(rng.integers(env.action_space.n))
        )
        completed_steps += 1
        goals += int(info["goal"] is not None)
        non_finite = (
            not np.all(np.isfinite(observation))
            or not np.isfinite(reward)
            or not state_is_finite(env)
        )
        if terminated or truncated:
            completed_episodes += 1
            if args.episodes is not None and completed_episodes >= args.episodes:
                break
            observation, _ = env.reset(seed=args.seed + completed_episodes)
    elapsed = time.perf_counter() - start
    env.close()
    throughput = completed_steps / elapsed if elapsed > 0.0 else float("inf")
    print(f"requested_steps: {args.steps}")
    print(f"completed_steps: {completed_steps}")
    print(f"completed_episodes: {completed_episodes}")
    print(f"goals: {goals}")
    print(f"elapsed_seconds: {elapsed:.6f}")
    print(f"environment_steps_per_second: {throughput:.1f}")
    print(f"non_finite_detected: {non_finite}")


if __name__ == "__main__":
    main()

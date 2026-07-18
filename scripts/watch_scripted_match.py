"""Visible deterministic match for environment validation, not training."""

from __future__ import annotations

import argparse

from haxball_env import EnvConfig, HaxBallEnv
from haxball_env.scripted import ScriptedOpponent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--opponent", choices=("scripted", "random"), default="scripted")
    args = parser.parse_args()
    if args.fps < 1 or args.max_steps < 1:
        parser.error("--fps and --max-steps must be positive")

    config = EnvConfig(render_fps=args.fps)
    env = HaxBallEnv(config, render_mode="human", opponent_mode=args.opponent)
    controlled_policy = ScriptedOpponent(config)
    episodes = goals = completed_steps = 0
    closed_by_user = False
    try:
        env.reset(seed=args.seed)
        while completed_steps < args.max_steps and not env.window_closed:
            action = controlled_policy.act(env.state, env.controlled_side)
            _, _, terminated, truncated, info = env.step(action)
            completed_steps += 1
            goals += int(info["goal"] is not None)
            if env.window_closed:
                closed_by_user = True
                break
            if terminated or truncated:
                episodes += 1
                env.reset(seed=args.seed + episodes)
    finally:
        env.close()
    print(
        f"done: steps={completed_steps} episodes={episodes} goals={goals} "
        f"closed_by_user={closed_by_user}"
    )


if __name__ == "__main__":
    main()

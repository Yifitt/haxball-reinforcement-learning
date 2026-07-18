from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from policy_contract.checkpoint_contract import PortablePolicy
from policy_contract.observation_contract import build_sim_observation
from sim_training.env_factory import make_training_env
from sim_training.policy import chase_bins, deterministic_bins


def _time(callable_, repetitions: int) -> float:
    started = time.perf_counter()
    for _ in range(repetitions):
        callable_()
    return time.perf_counter() - started


def benchmark_count(n_envs: int, *, physics_ticks: int, decisions: int, action_repeat: int):
    import haxball_core

    core = haxball_core.VecEnv(n_envs, 1, 1)
    core.reset_all()
    core.rollout_bench(20)
    started = time.perf_counter()
    core.rollout_bench(physics_ticks)
    raw_elapsed = time.perf_counter() - started

    env = make_training_env(n_envs, action_repeat=action_repeat)
    observations = env.reset()
    actions = np.ones((n_envs, 2, 3), dtype=np.int64)
    vector_elapsed = _time(lambda: env.step(actions), decisions)
    state = env.prev_state
    observation_elapsed = _time(lambda: build_sim_observation(state), decisions)
    model = PortablePolicy()
    inference_elapsed = _time(lambda: deterministic_bins(model, observations[:, 0]), decisions)

    observations = env.reset()
    def combined_step():
        nonlocal observations
        learner = deterministic_bins(model, observations[:, 0])
        opponent = chase_bins(env.prev_state, 1)
        observations = env.step(np.stack((learner, opponent), axis=1))[0]
    combined_elapsed = _time(combined_step, decisions)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "n_envs": n_envs,
        "physics_ticks_per_second": n_envs * physics_ticks / raw_elapsed,
        "environment_steps_per_second": n_envs * decisions / vector_elapsed,
        "agent_decisions_per_second": n_envs * decisions / combined_elapsed,
        "observation_builds_per_second": n_envs * 2 * decisions / observation_elapsed,
        "policy_inferences_per_second": n_envs * decisions / inference_elapsed,
        "combined_rollout_steps_per_second": n_envs * decisions / combined_elapsed,
        "elapsed_seconds": raw_elapsed + vector_elapsed + observation_elapsed + inference_elapsed + combined_elapsed,
        "peak_memory_mb": peak_kb / 1024.0,
        "action_repeat": action_repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=(1, 16, 64, 128, 256, 512))
    parser.add_argument("--physics-ticks", type=int, default=1000)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--action-repeat", type=int, default=8)
    args = parser.parse_args()
    records = [
        benchmark_count(
            count, physics_ticks=args.physics_ticks,
            decisions=args.decisions, action_repeat=args.action_repeat)
        for count in args.counts
    ]
    default = max(records, key=lambda record: record["combined_rollout_steps_per_second"])["n_envs"]
    print(json.dumps({"records": records, "recommended_local_n_envs": default}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

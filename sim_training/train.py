from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy_contract.checkpoint_contract import PortablePolicy, load_checkpoint
from policy_contract.chase_contract import kick_request_masks
from policy_contract.observation_contract import OBSERVATION_SIZE
from sim_training.checkpoint import (
    load_trainer_state,
    save_training_checkpoint,
    save_training_checkpoint_atomic,
)
from sim_training.env_factory import make_training_env
from sim_training.evaluate import HELD_OUT_SEEDS, evaluate_model
from sim_training.curriculum import (
    CURRICULUM_VERSION,
    configuration_for_stage,
    validate_stage_requirements,
)
from sim_training.opponent_pool import OpponentPool
from sim_training.goal_attribution import GoalAttributionTracker
from sim_training.promotion import promotion_decision
from sim_training.policy import evaluate_actions, sample_actions
from sim_training.randomized_reset import RESET_DISTRIBUTION_VERSION
from sim_training.reward_config import (
    DEFAULT_INVALID_KICK_PENALTY,
    DEFAULT_REPEATED_INVALID_KICK_PENALTY,
    reward_configuration,
)
from sim_training.self_play_pool import (
    FrozenSelfPlayPool,
    parse_frozen_checkpoint,
)

GAMMA = 0.9817
GAE_LAMBDA = 0.95
CLIP = 0.2
LEARNING_RATE = 3e-4
ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5
MAX_GRADIENT_NORM = 0.5

MODE_DEFAULTS = {
    "smoke": {"iterations": 1, "n_envs": 16, "rollout_steps": 16, "epochs": 1},
    "short": {"iterations": 8, "n_envs": 64, "rollout_steps": 32, "epochs": 2},
    "full": {"iterations": 128, "n_envs": 512, "rollout_steps": 64, "epochs": 4},
}


def resolve_configuration(args: argparse.Namespace) -> dict[str, object]:
    supported = {action.dest for action in build_parser()._actions}
    unsupported = sorted(set(vars(args)) - supported)
    if unsupported:
        raise ValueError("unsupported training configuration fields: " + ",".join(unsupported))
    defaults = MODE_DEFAULTS[args.mode]
    iterations = args.iterations if args.iterations is not None else defaults["iterations"]
    n_envs = args.n_envs if args.n_envs is not None else defaults["n_envs"]
    initialize_from = getattr(args, "initialize_from", None)
    if initialize_from and args.resume:
        raise ValueError("--initialize-from and --resume are mutually exclusive")
    starting_checkpoint = initialize_from or args.resume
    starting_checkpoint_hash = None
    if starting_checkpoint:
        supplied = Path(starting_checkpoint)
        model_path = supplied if supplied.suffix == ".pt" else supplied / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"starting checkpoint does not exist: {model_path}")
        starting_checkpoint_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    config = {
        "mode": args.mode,
        "iterations": iterations,
        "n_envs": n_envs,
        "rollout_steps": defaults["rollout_steps"],
        "epochs": defaults["epochs"],
        "seed": args.seed,
        "device": args.device,
        "checkpoint_dir": str(Path(args.checkpoint_dir)),
        "resume": args.resume,
        "initialize_from": initialize_from,
        "starting_checkpoint_sha256": starting_checkpoint_hash,
        "opponent": args.opponent,
        "opponent_configuration": {
            "names": list(configuration_for_stage(args.curriculum_stage).names),
            "weights": list(configuration_for_stage(args.curriculum_stage).weights),
        },
        "curriculum_stage": args.curriculum_stage,
        "curriculum_version": CURRICULUM_VERSION,
        "reset_distribution_version": RESET_DISTRIBUTION_VERSION,
        "action_repeat": args.action_repeat,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "eval_episodes": args.eval_episodes,
        "eval_n_envs": args.eval_n_envs,
        "eval_max_decisions": args.eval_max_decisions,
        "evaluation_seeds": list(HELD_OUT_SEEDS),
        "evaluation_opponents": (
            ["hard_scripted"]
            + [value.split("=", 1)[0] for value in args.frozen_checkpoint]
            + [value.split("=", 1)[0] for value in args.seed_self_play_checkpoint]
        ),
        "observation_size": OBSERVATION_SIZE,
        "action_size": 18,
        "network_architecture": {
            "type": "three_head_mlp", "hidden": args.hidden, "depth": args.depth,
            "heads": {"x": 3, "y": 3, "kick": 2},
        },
        "ppo_hyperparameters": {
            "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "clip": CLIP,
            "learning_rate": LEARNING_RATE, "entropy_coefficient": ENTROPY_COEFFICIENT,
            "value_coefficient": VALUE_COEFFICIENT,
            "maximum_gradient_norm": MAX_GRADIENT_NORM,
            "minibatch_size": min(1024, n_envs * defaults["rollout_steps"]),
        },
        "reward_configuration": reward_configuration(
            args.invalid_kick_penalty, args.repeated_invalid_kick_penalty),
        "frozen_checkpoints": list(args.frozen_checkpoint),
        "seed_self_play_checkpoints": list(args.seed_self_play_checkpoint),
        "self_play_snapshot_every": args.self_play_snapshot_every,
        "self_play_pool_cap": args.self_play_pool_cap,
        "random_learner_side": args.curriculum_stage == 4,
        "action_execution": "movement-repeat-kick-pulse-v1",
        "total_target_transitions": iterations * n_envs * defaults["rollout_steps"],
    }
    positive = (
        iterations, n_envs, defaults["rollout_steps"], args.action_repeat,
        args.eval_every, args.save_every, args.eval_episodes, args.eval_n_envs,
        args.eval_max_decisions,
        args.self_play_snapshot_every, args.self_play_pool_cap,
    )
    if min(positive) < 1:
        raise ValueError("training counts and intervals must be positive")
    if args.device != "cpu":
        raise ValueError("the current Rust rollout adapter supports --device cpu")
    if args.seed in HELD_OUT_SEEDS:
        raise ValueError("training seed is reserved for held-out evaluation")
    validated_stage4_resume = False
    if args.curriculum_stage == 4 and starting_checkpoint and not args.promotion_report:
        _, resume_metadata = load_checkpoint(starting_checkpoint)
        resume_environment = resume_metadata.get("training_environment", {})
        validated_stage4_resume = int(
            resume_environment.get("curriculum_stage", 0)) == 4
    validate_stage_requirements(
        args.curriculum_stage,
        previous_checkpoint=args.previous_policy_checkpoint,
        self_play_checkpoints=args.self_play_checkpoint,
        promotion_report=args.promotion_report,
        frozen_checkpoints=args.frozen_checkpoint,
        validated_stage4_resume=validated_stage4_resume,
    )
    if args.curriculum_stage == 4:
        parsed = [parse_frozen_checkpoint(value) for value in args.frozen_checkpoint]
        if len({label for label, _ in parsed}) != len(parsed):
            raise ValueError("frozen checkpoint labels must be unique")
        if args.self_play_pool_cap < len(parsed) + 1:
            raise ValueError("self-play pool cap must leave room for a generated snapshot")
        all_pool_sources = parsed + [
            parse_frozen_checkpoint(value) for value in args.seed_self_play_checkpoint]
        config["opponent_checkpoint_sources"] = [
            {
                "label": label,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for label, path in all_pool_sources
        ]
    return config


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as temporary:
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _git_metadata(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=root, check=True,
            capture_output=True, text=True).stdout.splitlines()
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff"], cwd=root, check=True,
            capture_output=True, text=True).stdout
        return {"available": True, "commit": commit, "status": status, "diff": diff}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False, "reason": "workspace is not a Git repository"}


def _is_better(evaluation: dict[str, float | int], best_goal_difference: int, best_reward: float) -> bool:
    score = (int(evaluation["goal_difference"]), float(evaluation["mean_reward"]))
    return score > (best_goal_difference, best_reward)


def train(args: argparse.Namespace) -> dict[str, object]:
    config = resolve_configuration(args)
    if args.dry_run:
        starting = getattr(args, "initialize_from", None) or args.resume
        if starting:
            _, metadata = load_checkpoint(starting)
            if int(metadata["action_repeat"]) != args.action_repeat:
                raise ValueError("starting checkpoint action_repeat is incompatible")
            architecture = metadata["network_architecture"]
            if (int(architecture["hidden"]), int(architecture["depth"])) != (args.hidden, args.depth):
                raise ValueError("starting checkpoint network architecture is incompatible")
        return {
            **config,
            "dry_run": True,
            "will_train": False,
            "simulator_only": True,
            "browser_dependencies": False,
            "best_checkpoint": str(Path(args.checkpoint_dir) / "best" / "model.pt"),
            "latest_checkpoint": str(Path(args.checkpoint_dir) / "latest" / "model.pt"),
        }

    iterations = int(config["iterations"])
    n_envs = int(config["n_envs"])
    rollout_steps = int(config["rollout_steps"])
    epochs = int(config["epochs"])
    checkpoint_root = Path(args.checkpoint_dir)
    initialize_from = getattr(args, "initialize_from", None)
    if initialize_from and checkpoint_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing experiment directory: {checkpoint_root}")
    if not args.resume and not initialize_from and checkpoint_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing experiment directory: {checkpoint_root}")
    if args.resume:
        existing_config_path = checkpoint_root / "run_configuration.json"
        if not existing_config_path.is_file():
            raise FileNotFoundError(
                f"resume requires the original run configuration: {existing_config_path}")
        existing_config = json.loads(existing_config_path.read_text())
        resume_contract_fields = (
            "ppo_hyperparameters", "network_architecture", "reward_configuration",
            "opponent_configuration", "opponent_checkpoint_sources", "n_envs",
            "rollout_steps", "epochs", "action_repeat", "iterations",
            "total_target_transitions", "eval_every", "save_every", "eval_episodes",
            "eval_n_envs", "eval_max_decisions", "evaluation_seeds",
            "evaluation_opponents", "seed",
        )
        mismatches = [
            name for name in resume_contract_fields
            if config.get(name) != existing_config.get(name)
        ]
        if mismatches:
            raise ValueError("resume configuration mismatch: " + ",".join(mismatches))
        resume_hash = config["starting_checkpoint_sha256"]
        config["starting_checkpoint_sha256"] = existing_config["starting_checkpoint_sha256"]
        config["initialize_from"] = existing_config.get("initialize_from")
        config["resume_checkpoint_sha256"] = resume_hash
        config["resume_history"] = [
            *existing_config.get("resume_history", []),
            {"path": str(args.resume), "sha256": resume_hash},
        ]
    upstream_revision = (
        Path(__file__).parents[1] / "external" / "HAXBALLGYM_REVISION"
    ).read_text().strip()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PortablePolicy(hidden=args.hidden, depth=args.depth)
    resume_metadata: dict[str, object] | None = None
    trainer_state: dict[str, object] | None = None
    if args.resume:
        model, resume_metadata = load_checkpoint(args.resume)
        if int(resume_metadata["action_repeat"]) != args.action_repeat:
            raise ValueError("resume checkpoint action_repeat is incompatible")
        trainer_state = load_trainer_state(args.resume)
    elif initialize_from:
        model, initialization_metadata = load_checkpoint(initialize_from)
        if int(initialization_metadata["action_repeat"]) != args.action_repeat:
            raise ValueError("initialization checkpoint action_repeat is incompatible")
        if (model.hidden, model.depth) != (args.hidden, args.depth):
            raise ValueError("initialization checkpoint network architecture is incompatible")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    completed_iterations = 0
    total_transitions = 0
    best_goal_difference = -(10**9)
    best_evaluation_reward = -math.inf
    metrics: list[dict[str, object]] = []
    resume_stage = None
    if resume_metadata is not None:
        resume_stage = resume_metadata.get("training_environment", {}).get("curriculum_stage")
    curriculum_changed = resume_metadata is not None and int(resume_stage or 0) != args.curriculum_stage
    if trainer_state is not None:
        completed_iterations = int(trainer_state["completed_iterations"])
        total_transitions = int(trainer_state["total_transitions"])
        if not curriculum_changed:
            best_goal_difference = int(trainer_state["best_goal_difference"])
            best_evaluation_reward = float(trainer_state["best_evaluation_reward"])
        optimizer.load_state_dict(trainer_state["optimizer_state"])
        np.random.set_state(trainer_state["numpy_random_state"])
        torch.set_rng_state(trainer_state["torch_random_state"])
        metrics_path = Path(args.resume).parent / "metrics.json" if Path(args.resume).suffix == ".pt" else Path(args.resume) / "metrics.json"
        if metrics_path.exists() and not curriculum_changed:
            metrics = json.loads(metrics_path.read_text())
    elif resume_metadata is not None:
        completed_iterations = int(resume_metadata["iterations"])
        total_transitions = int(resume_metadata["total_transitions"])
    if iterations <= completed_iterations:
        raise ValueError(
            f"--iterations must exceed completed iteration {completed_iterations}")

    starting_total_transitions = total_transitions
    starting_metrics_count = len(metrics)

    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    configured_reward = reward_configuration(
        args.invalid_kick_penalty, args.repeated_invalid_kick_penalty)
    env = make_training_env(
        n_envs, action_repeat=args.action_repeat, seed=args.seed, randomized_starts=True,
        invalid_kick_penalty=args.invalid_kick_penalty,
        repeated_invalid_kick_penalty=args.repeated_invalid_kick_penalty)
    observations = env.reset()
    previous_models = (
        (load_checkpoint(args.previous_policy_checkpoint)[0],)
        if args.previous_policy_checkpoint else ()
    )
    pool_manager: FrozenSelfPlayPool | None = None
    self_play_labels: tuple[str, ...] = ()
    if args.curriculum_stage == 4:
        anchors = [parse_frozen_checkpoint(value) for value in args.frozen_checkpoint]
        pool_manager = FrozenSelfPlayPool(
            checkpoint_root, anchors=anchors,
            seed_snapshots=[parse_frozen_checkpoint(value) for value in args.seed_self_play_checkpoint],
            cap=args.self_play_pool_cap,
            snapshot_every=args.self_play_snapshot_every)
        self_play_models, self_play_labels = pool_manager.load_active()
        self_play_weights = pool_manager.active_sampling_weights()
    else:
        self_play_models = tuple(load_checkpoint(path)[0] for path in args.self_play_checkpoint)
        self_play_labels = tuple(f"self_play_{index}" for index in range(len(self_play_models)))
        self_play_weights = None
    opponent_pool = OpponentPool(
        n_envs,
        seed=args.seed + 1,
        configuration=configuration_for_stage(args.curriculum_stage),
        previous_models=previous_models,
        self_play_models=self_play_models,
        self_play_labels=self_play_labels,
        self_play_weights=self_play_weights,
    )
    config["active_opponent_pool"] = (
        pool_manager.metadata() if pool_manager is not None else None)
    config["git"] = _git_metadata(Path(__file__).parents[1])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(checkpoint_root / "run_configuration.json", config)
    rows = np.arange(n_envs)
    side_rng = np.random.default_rng(args.seed + 2)
    learner_sides = (
        side_rng.integers(0, 2, n_envs, dtype=np.int64)
        if args.curriculum_stage == 4 else np.zeros(n_envs, dtype=np.int64)
    )
    previous_invalid = np.zeros(n_envs, dtype=bool)
    attribution = GoalAttributionTracker(n_envs)
    if trainer_state is not None and not curriculum_changed:
        reset_mutator = getattr(env.state_mutator, "base", env.state_mutator)
        if "reset_rng_state" in trainer_state:
            reset_mutator.rng.bit_generator.state = trainer_state["reset_rng_state"]
        if "opponent_rng_state" in trainer_state:
            opponent_pool.rng.bit_generator.state = trainer_state["opponent_rng_state"]
        if "side_rng_state" in trainer_state:
            side_rng.bit_generator.state = trainer_state["side_rng_state"]
        if "learner_sides" in trainer_state:
            restored_sides = np.asarray(trainer_state["learner_sides"], dtype=np.int64)
            if restored_sides.shape == learner_sides.shape:
                learner_sides[:] = restored_sides
        if "previous_invalid" in trainer_state:
            restored_invalid = np.asarray(trainer_state["previous_invalid"], dtype=bool)
            if restored_invalid.shape == previous_invalid.shape:
                previous_invalid[:] = restored_invalid
        opponent_pool.reset(np.ones(n_envs, dtype=bool), env.state_mutator.kickoff_delay)
    started = time.perf_counter()
    optimizer_updates = 0
    interrupted = False

    def save_at(target: Path) -> tuple[Path, Path]:
        reset_mutator = getattr(env.state_mutator, "base", env.state_mutator)
        result = save_training_checkpoint_atomic(
            target,
            model,
            optimizer=optimizer,
            completed_iterations=completed_iterations,
            upstream_revision=upstream_revision,
            action_repeat=args.action_repeat,
            seed=args.seed,
            n_envs=n_envs,
            total_transitions=total_transitions,
            best_goal_difference=best_goal_difference,
            best_evaluation_reward=best_evaluation_reward,
            metrics=metrics,
            curriculum_stage=args.curriculum_stage,
            reward_configuration_value=configured_reward,
            experiment_configuration=config,
            extra_trainer_state={
                "reset_rng_state": reset_mutator.rng.bit_generator.state,
                "opponent_rng_state": opponent_pool.rng.bit_generator.state,
                "side_rng_state": side_rng.bit_generator.state,
                "learner_sides": learner_sides.copy(),
                "previous_invalid": previous_invalid.copy(),
                "self_play_pool_metadata": (
                    pool_manager.metadata() if pool_manager is not None else None),
                "champion_evaluation": champion_evaluation,
            },
        )
        if pool_manager is not None:
            _atomic_json(target / "self_play_pool_metadata.json", pool_manager.metadata())
        return result

    champion_evaluation: dict[str, object] | None = None
    if args.promotion_report:
        supplied_promotion = json.loads(Path(args.promotion_report).read_text())
        if isinstance(supplied_promotion, dict) and "strongest" in supplied_promotion:
            champion_evaluation = supplied_promotion["strongest"].get("metrics")
    if trainer_state is not None and trainer_state.get("champion_evaluation"):
        champion_evaluation = trainer_state["champion_evaluation"]
    if args.curriculum_stage == 4 and not (checkpoint_root / "best" / "model.pt").exists():
        save_at(checkpoint_root / "best")

    try:
        for iteration in range(completed_iterations + 1, iterations + 1):
            obs_buffer = torch.empty(rollout_steps, n_envs, OBSERVATION_SIZE)
            action_buffer = torch.empty(rollout_steps, n_envs, 3, dtype=torch.long)
            logp_buffer = torch.empty(rollout_steps, n_envs)
            value_buffer = torch.empty(rollout_steps, n_envs)
            reward_buffer = torch.empty(rollout_steps, n_envs)
            done_buffer = torch.empty(rollout_steps, n_envs)
            entropy_sum = kick_count = valid_kick_count = invalid_kick_count = 0.0
            held_invalid_kick_count = successful_contact_count = reward_sum = 0.0
            invalid_streak = np.zeros(n_envs, dtype=np.int64)
            maximum_invalid_streak = 0
            agent_normal_goals = opponent_normal_goals = 0
            goals_for = goals_against = agent_own_goals = opponent_own_goals = 0
            opponent_results: dict[str, dict[str, int]] = {}
            action_array = np.empty((n_envs, 2, 3), dtype=np.int64)
            rollout_started = time.perf_counter()

            for step in range(rollout_steps):
                opponent_sides = 1 - learner_sides
                learner_obs = observations[rows, learner_sides]
                opponent_obs = observations[rows, opponent_sides]
                with torch.no_grad():
                    learner, log_probability, entropy, values = sample_actions(model, learner_obs)
                opponent = opponent_pool.actions(
                    env.prev_state, opponent_sides, opponent_obs)
                learner_actions = learner.numpy()
                requested, valid, invalid, held_invalid = kick_request_masks(
                    learner_actions, env.prev_state, player_index=learner_sides,
                    previous_invalid=previous_invalid)
                opponent_requested, opponent_valid, _, _ = kick_request_masks(
                    opponent, env.prev_state, player_index=opponent_sides)
                opponent_valid = opponent_requested & opponent_valid
                attribution.record_masks(valid, opponent_valid, learner_sides)
                before_ball_velocity = env.prev_state.ball_vel.copy()
                kick_count += int(requested.sum())
                valid_kick_count += int(valid.sum())
                invalid_kick_count += int(invalid.sum())
                held_invalid_kick_count += int(held_invalid.sum())
                invalid_streak = np.where(invalid, invalid_streak + 1, 0)
                maximum_invalid_streak = max(maximum_invalid_streak, int(invalid_streak.max()))
                action_array[rows, learner_sides] = learner_actions
                action_array[rows, opponent_sides] = opponent
                next_observations, rewards, terminated, truncated = env.step(action_array)
                velocity_change = np.linalg.norm(
                    env.prev_state.ball_vel - before_ball_velocity, axis=1)
                successful_contact_count += int((valid & (velocity_change > 1e-5)).sum())
                done = terminated | truncated
                learner_rewards = rewards[rows, learner_sides]
                obs_buffer[step] = torch.as_tensor(learner_obs)
                action_buffer[step] = learner
                logp_buffer[step] = log_probability
                value_buffer[step] = values
                reward_buffer[step] = torch.as_tensor(learner_rewards)
                done_buffer[step] = torch.as_tensor(done.astype(np.float32))
                entropy_sum += float(entropy.sum())
                reward_sum += float(learner_rewards.sum())
                if done.any():
                    assignments = opponent_pool.labels()
                    conceding = env.prev_state.scored
                    learner_team = np.where(learner_sides == 0, 2, 4)
                    opponent_team = np.where(learner_sides == 0, 4, 2)
                    learner_scored = done & (conceding == opponent_team)
                    opponent_scored = done & (conceding == learner_team)
                    attributed = attribution.attribute(conceding, learner_sides)
                    agent_normal = done & attributed["agent_normal_goals"]
                    opponent_normal = done & attributed["opponent_normal_goals"]
                    agent_own = done & attributed["agent_own_goals"]
                    opponent_own = done & attributed["opponent_own_goals"]
                    goals_for += int(learner_scored.sum())
                    goals_against += int(opponent_scored.sum())
                    agent_own_goals += int(agent_own.sum())
                    opponent_own_goals += int(opponent_own.sum())
                    agent_normal_goals += int(agent_normal.sum())
                    opponent_normal_goals += int(opponent_normal.sum())
                    for index in np.flatnonzero(done):
                        label = str(assignments[index])
                        result = opponent_results.setdefault(label, {
                            "episodes": 0, "wins": 0, "goals_for": 0,
                            "goals_against": 0, "agent_own_goals": 0,
                            "opponent_own_goals": 0, "agent_normal_goals": 0,
                            "opponent_normal_goals": 0,
                        })
                        result["episodes"] += 1
                        result["wins"] += int(learner_scored[index])
                        result["goals_for"] += int(learner_scored[index])
                        result["goals_against"] += int(opponent_scored[index])
                        result["agent_own_goals"] += int(agent_own[index])
                        result["opponent_own_goals"] += int(opponent_own[index])
                        result["agent_normal_goals"] += int(agent_normal[index])
                        result["opponent_normal_goals"] += int(opponent_normal[index])
                    opponent_pool.reset(done, env.state_mutator.kickoff_delay)
                    if args.curriculum_stage == 4:
                        learner_sides[done] = side_rng.integers(
                            0, 2, int(done.sum()), dtype=np.int64)
                    attribution.reset(done)
                previous_invalid = invalid
                previous_invalid[done] = False
                observations = next_observations

            rollout_seconds = time.perf_counter() - rollout_started

            with torch.no_grad():
                next_values = model(torch.as_tensor(
                    observations[rows, learner_sides], dtype=torch.float32))[3]
            advantages = torch.zeros_like(reward_buffer)
            last_advantage = torch.zeros(n_envs)
            for step in reversed(range(rollout_steps)):
                not_done = 1.0 - done_buffer[step]
                following = next_values if step == rollout_steps - 1 else value_buffer[step + 1]
                delta = reward_buffer[step] + GAMMA * following * not_done - value_buffer[step]
                last_advantage = delta + GAMMA * GAE_LAMBDA * not_done * last_advantage
                advantages[step] = last_advantage
            returns = advantages + value_buffer

            batch_obs = obs_buffer.reshape(-1, OBSERVATION_SIZE)
            batch_actions = action_buffer.reshape(-1, 3)
            batch_logp = logp_buffer.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_advantages = (
                batch_advantages - batch_advantages.mean()
            ) / (batch_advantages.std() + 1e-8)
            indices = np.arange(len(batch_obs))
            minibatch_size = min(1024, len(indices))
            iteration_losses: list[float] = []
            optimization_started = time.perf_counter()
            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), minibatch_size):
                    selection = torch.as_tensor(indices[start : start + minibatch_size])
                    log_probability, entropy, values = evaluate_actions(
                        model, batch_obs[selection], batch_actions[selection])
                    ratio = (log_probability - batch_logp[selection]).exp()
                    advantage = batch_advantages[selection]
                    policy_loss = -torch.min(
                        ratio * advantage,
                        torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage,
                    ).mean()
                    value_loss = 0.5 * (values - batch_returns[selection]).pow(2).mean()
                    loss = (
                        policy_loss + VALUE_COEFFICIENT * value_loss
                        - ENTROPY_COEFFICIENT * entropy.mean()
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite PPO loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
                    optimizer.step()
                    iteration_losses.append(float(loss.detach()))
                    optimizer_updates += 1
            optimization_seconds = time.perf_counter() - optimization_started

            completed_iterations = iteration
            iteration_transitions = rollout_steps * n_envs
            total_transitions += iteration_transitions
            record: dict[str, object] = {
                "iteration": iteration,
                "total_transitions": total_transitions,
                "mean_training_reward": reward_sum / iteration_transitions,
                "action_entropy": entropy_sum / iteration_transitions,
                "kick_fraction": kick_count / iteration_transitions,
                "kick_request_fraction": kick_count / iteration_transitions,
                "valid_kick_fraction_of_requests": (
                    valid_kick_count / kick_count if kick_count else 0.0),
                "valid_kick_fraction_of_decisions": valid_kick_count / iteration_transitions,
                "invalid_kick_fraction_of_requests": (
                    invalid_kick_count / kick_count if kick_count else 0.0),
                "invalid_kick_fraction_of_decisions": invalid_kick_count / iteration_transitions,
                "held_invalid_kick_fraction_of_decisions": (
                    held_invalid_kick_count / iteration_transitions),
                "held_invalid_kick_fraction": held_invalid_kick_count / iteration_transitions,
                "successful_contact_fraction": (
                    successful_contact_count / kick_count if kick_count else 0.0),
                "consecutive_invalid_kick_streak": maximum_invalid_streak,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "agent_normal_goals": agent_normal_goals,
                "opponent_normal_goals": opponent_normal_goals,
                "agent_own_goals": agent_own_goals,
                "opponent_own_goals": opponent_own_goals,
                "win_rate_by_opponent": {
                    label: {
                        **result,
                        "win_rate": result["wins"] / result["episodes"],
                    }
                    for label, result in sorted(opponent_results.items())
                },
                "mean_loss": float(np.mean(iteration_losses)),
                "rollout_seconds": rollout_seconds,
                "optimization_seconds": optimization_seconds,
                "optimizer_time_seconds": optimization_seconds,
                "transitions_per_second": iteration_transitions / rollout_seconds,
                "evaluation_seconds": 0.0,
                "checkpoint_seconds": 0.0,
            }
            if (
                pool_manager is not None
                and iteration % args.self_play_snapshot_every == 0
            ):
                snapshot_started = time.perf_counter()

                def save_snapshot(directory: Path) -> None:
                    save_training_checkpoint(
                        directory, model, upstream_revision=upstream_revision,
                        action_repeat=args.action_repeat, seed=args.seed,
                        n_envs=n_envs, iterations=completed_iterations,
                        total_transitions=total_transitions, curriculum_stage=4,
                        reward_configuration_value=configured_reward,
                        experiment_configuration=config)

                pool_manager.add_snapshot(iteration, save_snapshot)
                active_models, active_labels = pool_manager.load_active()
                opponent_pool.set_self_play_models(
                    active_models, active_labels, pool_manager.active_sampling_weights())
                record["self_play_snapshot_added"] = f"self_play_iter_{iteration:06d}"
                record["active_self_play_generations"] = list(active_labels)
                record["checkpoint_seconds"] = (
                    float(record["checkpoint_seconds"])
                    + time.perf_counter() - snapshot_started)
            if iteration % args.eval_every == 0 or iteration == iterations:
                evaluation_started = time.perf_counter()
                evaluation_models: tuple[object, ...] = ()
                evaluation_labels: tuple[str, ...] = ()
                evaluation_configuration = configuration_for_stage(1)
                random_evaluation_side = False
                if pool_manager is not None:
                    evaluation_models, evaluation_labels = pool_manager.load_active()
                    evaluation_configuration = configuration_for_stage(4)
                    random_evaluation_side = True
                evaluation = evaluate_model(
                    model,
                    episodes=args.eval_episodes,
                    n_envs=min(args.eval_n_envs, n_envs),
                    action_repeat=args.action_repeat,
                    max_decisions=args.eval_max_decisions,
                    random_learner_side=random_evaluation_side,
                    opponent_configuration=evaluation_configuration,
                    opponent_models=evaluation_models,
                    opponent_labels=evaluation_labels,
                )
                record["evaluation_seconds"] = time.perf_counter() - evaluation_started
                record["evaluation"] = evaluation
                should_promote = _is_better(
                    evaluation, best_goal_difference, best_evaluation_reward)
                if pool_manager is not None:
                    checkpoint_rows = evaluation["score_by_checkpoint_generation"]
                    stage2 = checkpoint_rows.get("stage2", {"win_rate": 0.0})
                    hard = evaluation["score_vs_hard_rule_based"]
                    promotion_input = {
                        **evaluation,
                        "stage2_win_rate": stage2["win_rate"],
                        "hard_rule_win_rate": hard["win_rate"],
                        "worst_checkpoint_win_rate": min(
                            (row["win_rate"] for row in checkpoint_rows.values()),
                            default=0.0,
                        ),
                    }
                    decision = promotion_decision(
                        promotion_input, champion_report=champion_evaluation)
                    record["promotion"] = decision
                    should_promote = bool(decision["promoted"])
                if should_promote:
                    best_goal_difference = int(evaluation["goal_difference"])
                    best_evaluation_reward = float(evaluation["mean_reward"])
                    if pool_manager is not None:
                        champion_evaluation = promotion_input
                        if record.get("self_play_snapshot_added"):
                            pool_manager.promote_snapshot(
                                iteration, float(record["promotion"]["promotion_score"]))
                            active_models, active_labels = pool_manager.load_active()
                            opponent_pool.set_self_play_models(
                                active_models, active_labels,
                                pool_manager.active_sampling_weights())
                            record["active_self_play_generations"] = list(active_labels)
                    metrics.append(record)
                    checkpoint_started = time.perf_counter()
                    save_at(checkpoint_root / "best")
                    record["checkpoint_seconds"] = (
                        float(record["checkpoint_seconds"])
                        + time.perf_counter() - checkpoint_started)
                    metrics.pop()
            metrics.append(record)
            _atomic_json(checkpoint_root / "training_metrics.json", metrics)
            if iteration % args.save_every == 0 or iteration == iterations:
                checkpoint_started = time.perf_counter()
                save_at(checkpoint_root / "periodic" / f"iter_{iteration:06d}")
                record["checkpoint_seconds"] = (
                    float(record["checkpoint_seconds"])
                    + time.perf_counter() - checkpoint_started)
                _atomic_json(checkpoint_root / "training_metrics.json", metrics)
    except KeyboardInterrupt:
        interrupted = True

    if completed_iterations > 0:
        final_checkpoint_started = time.perf_counter()
        latest_model, latest_metadata = save_at(checkpoint_root / "latest")
        final_checkpoint_seconds = time.perf_counter() - final_checkpoint_started
    else:
        raise RuntimeError("training stopped before completing an iteration")
    elapsed = time.perf_counter() - started
    new_metrics = metrics[starting_metrics_count:]
    new_transitions = total_transitions - starting_total_transitions
    measured_rollout_seconds = sum(
        float(row["rollout_seconds"]) for row in new_metrics if "rollout_seconds" in row)
    parameters_changed = any(
        not torch.equal(initial_parameters[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    if not parameters_finite or (optimizer_updates > 0 and not parameters_changed):
        raise RuntimeError("PPO update did not leave a valid changed model")
    report: dict[str, object] = {
        **config,
        "dry_run": False,
        "interrupted": interrupted,
        "completed_iterations": completed_iterations,
        "total_transitions": total_transitions,
        "new_optimizer_updates": optimizer_updates,
        "parameters_changed": parameters_changed,
        "parameters_finite": parameters_finite,
        "elapsed_seconds": elapsed,
        "rollout_seconds": measured_rollout_seconds,
        "optimization_seconds": sum(float(row["optimization_seconds"]) for row in new_metrics if "optimization_seconds" in row),
        "evaluation_seconds": sum(float(row.get("evaluation_seconds", 0.0)) for row in new_metrics),
        "checkpoint_seconds": (
            sum(float(row.get("checkpoint_seconds", 0.0)) for row in new_metrics)
            + final_checkpoint_seconds),
        "transitions_per_second": new_transitions / elapsed,
        "rollout_transitions_per_second": (
            new_transitions / measured_rollout_seconds if measured_rollout_seconds else 0.0),
        "latest_model": str(latest_model),
        "latest_metadata": str(latest_metadata),
        "best_model": str(checkpoint_root / "best" / "model.pt"),
        "metrics_path": str(checkpoint_root / "training_metrics.json"),
    }
    _atomic_json(checkpoint_root / "training_report.json", report)
    del env
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_DEFAULTS), default="full")
    parser.add_argument("--iterations", "--iters", dest="iterations", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo_vs_chase")
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-from",
        help="load checkpoint weights into a new run with iteration/optimizer counters reset")
    parser.add_argument("--opponent", choices=("pool",), default="pool")
    parser.add_argument("--curriculum-stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--previous-policy-checkpoint")
    parser.add_argument("--self-play-checkpoint", action="append", default=[])
    parser.add_argument(
        "--frozen-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--seed-self-play-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--self-play-snapshot-every", type=int, default=16)
    parser.add_argument("--self-play-pool-cap", type=int, default=8)
    parser.add_argument("--promotion-report")
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument(
        "--invalid-kick-penalty", type=float,
        default=DEFAULT_INVALID_KICK_PENALTY)
    parser.add_argument(
        "--repeated-invalid-kick-penalty", type=float,
        default=DEFAULT_REPEATED_INVALID_KICK_PENALTY)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-n-envs", type=int, default=64)
    parser.add_argument("--eval-max-decisions", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from policy_contract.checkpoint_contract import (
    PortablePolicy,
    checkpoint_metadata,
    save_checkpoint,
)
from sim_training.reward_config import REWARD_CONFIGURATION
from sim_training.randomized_reset import RESET_DISTRIBUTION, RESET_DISTRIBUTION_VERSION
from sim_training.curriculum import CURRICULUM_VERSION


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def save_training_checkpoint(
    directory: str | Path,
    model: PortablePolicy,
    *,
    upstream_revision: str,
    action_repeat: int,
    seed: int,
    n_envs: int,
    iterations: int,
    total_transitions: int,
    curriculum_stage: int = 1,
    action_execution: str = "movement-repeat-kick-pulse-v1",
    reward_configuration_value: dict[str, object] | None = None,
    experiment_configuration: dict[str, object] | None = None,
):
    configured_reward = reward_configuration_value or REWARD_CONFIGURATION
    metadata = checkpoint_metadata(
        model,
        upstream_revision=upstream_revision,
        action_repeat=action_repeat,
        seed=seed,
        opponent="episode_sampled_pool",
        reward_configuration=configured_reward,
        n_envs=n_envs,
        iterations=iterations,
        total_transitions=total_transitions,
        device="cpu",
        package_versions={
            name: _version(name)
            for name in ("haxball-core", "haxballgym", "numpy", "torch")
        },
        training_environment={
            "version": "randomized-pool-reward-v2",
            "reset_distribution_version": RESET_DISTRIBUTION_VERSION,
            "reset_distribution": RESET_DISTRIBUTION,
            "reward_configuration_version": configured_reward["version"],
            "curriculum_version": CURRICULUM_VERSION,
            "curriculum_stage": curriculum_stage,
            "action_execution": action_execution,
            "held_out_validation": "pending_or_recorded_in_metrics",
            "experiment_configuration": experiment_configuration,
        },
    )
    return (*save_checkpoint(directory, model, metadata), metadata)


def save_training_checkpoint_atomic(
    directory: str | Path,
    model: PortablePolicy,
    *,
    optimizer: torch.optim.Optimizer,
    completed_iterations: int,
    upstream_revision: str,
    action_repeat: int,
    seed: int,
    n_envs: int,
    total_transitions: int,
    best_goal_difference: int,
    best_evaluation_reward: float,
    metrics: list[dict[str, object]],
    curriculum_stage: int = 1,
    extra_trainer_state: dict[str, object] | None = None,
    reward_configuration_value: dict[str, object] | None = None,
    experiment_configuration: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write every checkpoint file through a same-filesystem atomic replace."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        model_path, metadata_path, _ = save_training_checkpoint(
            temporary,
            model,
            upstream_revision=upstream_revision,
            action_repeat=action_repeat,
            seed=seed,
            n_envs=n_envs,
            iterations=completed_iterations,
            total_transitions=total_transitions,
            curriculum_stage=curriculum_stage,
            reward_configuration_value=reward_configuration_value,
            experiment_configuration=experiment_configuration,
        )
        trainer_state = {
            "completed_iterations": completed_iterations,
            "total_transitions": total_transitions,
            "optimizer_state": optimizer.state_dict(),
            "best_goal_difference": best_goal_difference,
            "best_evaluation_reward": best_evaluation_reward,
            "numpy_random_state": __import__("numpy").random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if extra_trainer_state:
            trainer_state.update(extra_trainer_state)
        torch.save(trainer_state, temporary / "trainer_state.pt")
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        for source in (
            model_path,
            metadata_path,
            temporary / "trainer_state.pt",
            temporary / "metrics.json",
        ):
            os.replace(source, target / source.name)
        return target / "model.pt", target / "policy_metadata.json"
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def load_trainer_state(directory_or_model: str | Path) -> dict[str, object] | None:
    supplied = Path(directory_or_model)
    directory = supplied.parent if supplied.suffix == ".pt" else supplied
    path = directory / "trainer_state.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)

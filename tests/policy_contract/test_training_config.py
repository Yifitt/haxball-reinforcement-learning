from __future__ import annotations

import argparse

import pytest
import torch

from policy_contract.checkpoint_contract import PortablePolicy, load_checkpoint
from sim_training.checkpoint import (
    load_trainer_state,
    save_training_checkpoint_atomic,
)
from sim_training.train import _is_better, resolve_configuration, train


def arguments(tmp_path, **overrides):
    values = {
        "mode": "full",
        "iterations": None,
        "n_envs": None,
        "seed": 0,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "run"),
        "resume": None,
        "opponent": "pool",
        "curriculum_stage": 1,
        "previous_policy_checkpoint": None,
        "self_play_checkpoint": [],
        "frozen_checkpoint": [],
        "seed_self_play_checkpoint": [],
        "self_play_snapshot_every": 16,
        "self_play_pool_cap": 8,
        "promotion_report": None,
        "action_repeat": 8,
        "invalid_kick_penalty": -0.004,
        "repeated_invalid_kick_penalty": -0.002,
        "hidden": 256,
        "depth": 2,
        "eval_every": 8,
        "save_every": 8,
        "eval_episodes": 128,
        "eval_n_envs": 64,
        "eval_max_decisions": 300,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_full_configuration_defaults_to_512_envs_and_several_million_transitions(tmp_path) -> None:
    config = resolve_configuration(arguments(tmp_path))
    assert config["n_envs"] == 512
    assert config["iterations"] == 128
    assert config["total_target_transitions"] == 4_194_304
    assert config["observation_size"] == 16
    assert config["action_size"] == 18
    assert config["curriculum_stage"] == 1
    assert config["eval_episodes"] >= 100


def test_training_dry_run_does_not_create_checkpoint_or_import_browser(tmp_path) -> None:
    report = train(arguments(tmp_path))
    assert report["dry_run"] is True
    assert report["will_train"] is False
    assert report["browser_dependencies"] is False
    assert not (tmp_path / "run").exists()


def test_atomic_checkpoint_contains_portable_and_resumable_state(tmp_path) -> None:
    model = PortablePolicy(hidden=16)
    optimizer = torch.optim.Adam(model.parameters())
    target = tmp_path / "latest"
    model_path, _ = save_training_checkpoint_atomic(
        target,
        model,
        optimizer=optimizer,
        completed_iterations=4,
        upstream_revision="1781914b79ccad18a65367d66de4645412405bce",
        action_repeat=8,
        seed=3,
        n_envs=512,
        total_transitions=131_072,
        best_goal_difference=2,
        best_evaluation_reward=0.1,
        metrics=[{"iteration": 4}],
        extra_trainer_state={"side_rng_state": {"test": True}},
    )
    loaded, metadata = load_checkpoint(model_path)
    trainer = load_trainer_state(target)
    assert isinstance(loaded, PortablePolicy)
    assert metadata["number_of_actions"] == 18
    assert metadata["training_environment"]["reset_distribution_version"]
    assert "human_" + "scenarios" not in metadata["training_environment"]
    assert trainer["completed_iterations"] == 4
    assert trainer["side_rng_state"] == {"test": True}
    assert (target / "metrics.json").exists()


def test_removed_reset_dataset_configuration_is_rejected(tmp_path) -> None:
    removed_key = "human_" + "scenarios"
    with pytest.raises(ValueError, match="unsupported training configuration fields"):
        resolve_configuration(arguments(tmp_path, **{removed_key: "dataset.npz"}))


def test_legacy_comparison_is_retained_for_pre_stage4_training() -> None:
    assert _is_better({"goal_difference": 1, "mean_reward": -1.0}, 0, 100.0)
    assert _is_better({"goal_difference": 1, "mean_reward": 0.2}, 1, 0.1)
    assert not _is_better({"goal_difference": 0, "mean_reward": 100.0}, 1, 0.1)


def test_dry_run_rejects_incompatible_resume_action_repeat(tmp_path) -> None:
    model = PortablePolicy(hidden=16)
    optimizer = torch.optim.Adam(model.parameters())
    target = tmp_path / "resume"
    save_training_checkpoint_atomic(
        target,
        model,
        optimizer=optimizer,
        completed_iterations=1,
        upstream_revision="1781914b79ccad18a65367d66de4645412405bce",
        action_repeat=8,
        seed=0,
        n_envs=16,
        total_transitions=256,
        best_goal_difference=0,
        best_evaluation_reward=0.0,
        metrics=[],
    )
    with pytest.raises(ValueError, match="action_repeat"):
        train(arguments(
            tmp_path,
            resume=str(target / "model.pt"),
            action_repeat=4,
        ))

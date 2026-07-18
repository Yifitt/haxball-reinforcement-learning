from __future__ import annotations

import json

import pytest

from sim_training.curriculum import configuration_for_stage, validate_stage_requirements


def test_curriculum_introduces_policy_then_self_play_opponents() -> None:
    assert "previous_policy" not in configuration_for_stage(2).names
    assert "previous_policy" in configuration_for_stage(3).names
    assert "self_play" in configuration_for_stage(4).names


def test_self_play_requires_consistent_randomized_pool_wins(tmp_path) -> None:
    report = tmp_path / "metrics.json"
    report.write_text(json.dumps([
        {"evaluation": {"episodes": 100, "win_rate": value}}
        for value in (0.61, 0.70, 0.65)
    ]))
    validate_stage_requirements(
        4, previous_checkpoint="previous/model.pt",
        self_play_checkpoints=[], promotion_report=str(report),
        frozen_checkpoints=["stage1=model.pt", "stage2=model.pt"])
    report.write_text(json.dumps([
        {"evaluation": {"episodes": 100, "win_rate": value}}
        for value in (0.61, 0.59, 0.65)
    ]))
    with pytest.raises(ValueError, match="three consecutive"):
        validate_stage_requirements(
            4, previous_checkpoint="previous/model.pt",
            self_play_checkpoints=[], promotion_report=str(report),
            frozen_checkpoints=["stage1=model.pt", "stage2=model.pt"])


def test_existing_stage4_checkpoint_can_resume_without_repeating_promotion_gate() -> None:
    validate_stage_requirements(
        4, previous_checkpoint=None, self_play_checkpoints=[], promotion_report=None,
        frozen_checkpoints=["stage1=model.pt", "stage2=model.pt"],
        validated_stage4_resume=True)
    with pytest.raises(ValueError, match="promotion-report"):
        validate_stage_requirements(
            4, previous_checkpoint=None, self_play_checkpoints=[], promotion_report=None,
            frozen_checkpoints=["stage1=model.pt", "stage2=model.pt"])

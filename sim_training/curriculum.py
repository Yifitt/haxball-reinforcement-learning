from __future__ import annotations

import json
from pathlib import Path

from sim_training.opponent_pool import OpponentPoolConfiguration

CURRICULUM_VERSION = "ppo-curriculum-v2"

STAGES = {
    1: OpponentPoolConfiguration(
        names=("standard_chase", "delayed_chase", "noisy_chase", "defensive_chase",
               "aggressive_chase", "mixed_persistent_random_chase"),
        weights=(0.25, 0.15, 0.15, 0.15, 0.15, 0.15),
    ),
    2: OpponentPoolConfiguration(
        names=("standard_chase", "noisy_chase", "defensive_chase", "aggressive_chase",
               "mixed_persistent_random_chase"),
        weights=(0.05, 0.20, 0.25, 0.35, 0.15),
    ),
    3: OpponentPoolConfiguration(
        names=("noisy_chase", "defensive_chase", "aggressive_chase", "previous_policy"),
        weights=(0.15, 0.20, 0.30, 0.35),
    ),
    4: OpponentPoolConfiguration(
        names=("defensive_chase", "aggressive_chase", "self_play"),
        weights=(0.10, 0.10, 0.80),
    ),
}


def configuration_for_stage(stage: int) -> OpponentPoolConfiguration:
    if stage not in STAGES:
        raise ValueError("curriculum stage must be 1, 2, 3, or 4")
    return STAGES[stage]


def validate_stage_requirements(
    stage: int,
    *,
    previous_checkpoint: str | None,
    self_play_checkpoints: list[str],
    promotion_report: str | None,
    frozen_checkpoints: list[str] | None = None,
    validated_stage4_resume: bool = False,
) -> None:
    if stage == 3 and not previous_checkpoint:
        raise ValueError("curriculum stage 3 requires --previous-policy-checkpoint")
    if stage < 4:
        return
    if frozen_checkpoints is None:
        frozen_checkpoints = self_play_checkpoints
    if len(frozen_checkpoints) < 2:
        raise ValueError("curriculum stage 4 requires at least two --frozen-checkpoint anchors")
    if not promotion_report:
        if validated_stage4_resume:
            return
        raise ValueError("curriculum stage 4 requires --promotion-report")
    records = json.loads(Path(promotion_report).read_text())
    if isinstance(records, dict) and records.get("protocol") == "held-out-balanced-stage4-tournament-v1":
        strongest = records.get("strongest", {})
        metrics = strongest.get("metrics", {})
        if strongest.get("promoted") and metrics.get("episodes", 0) >= 100:
            return
        raise ValueError("tournament recovery report has no healthy promoted checkpoint")
    evaluations = [record["evaluation"] for record in records if "evaluation" in record]
    recent = evaluations[-3:]
    if len(recent) < 3 or any(
        report.get("episodes", 0) < 100 or report.get("win_rate", 0.0) < 0.60
        for report in recent
    ):
        raise ValueError(
            "self-play promotion requires three consecutive 100+ episode randomized-pool "
            "evaluations with win_rate >= 0.60")

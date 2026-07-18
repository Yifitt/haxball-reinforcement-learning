from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from policy_contract.checkpoint_contract import PortablePolicy
from sim_training.curriculum import configuration_for_stage
from sim_training.opponent_pool import OpponentPool
from sim_training.self_play_pool import FrozenSelfPlayPool, parse_frozen_checkpoint

ROOT = Path(__file__).parents[2]
STAGE1 = ROOT / "checkpoints/ppo_randomized_curriculum_v1/best/model.pt"
STAGE2 = ROOT / "checkpoints/ppo_randomized_curriculum_stage2_v1/best/model.pt"


def test_stage4_sampling_is_80_percent_checkpoint_and_20_percent_hard_rule() -> None:
    pool = OpponentPool(
        20_000, seed=81, configuration=configuration_for_stage(4),
        self_play_models=(PortablePolicy(hidden=8), PortablePolicy(hidden=8)),
        self_play_labels=("stage1", "stage2"),
    )
    names = pool.names()
    checkpoint_fraction = float((names == "self_play").mean())
    assert 0.78 < checkpoint_fraction < 0.82
    assert set(pool.labels()[names == "self_play"]) == {"stage1", "stage2"}
    assert set(names[names != "self_play"]) == {"defensive_chase", "aggressive_chase"}


def test_frozen_pool_keeps_anchors_immutable_and_caps_active_generations(tmp_path) -> None:
    anchors = [parse_frozen_checkpoint(f"stage1={STAGE1}"), parse_frozen_checkpoint(f"stage2={STAGE2}")]
    manager = FrozenSelfPlayPool(
        tmp_path / "run", anchors=anchors, cap=4, snapshot_every=1)

    def save_copy(directory: Path) -> None:
        shutil.copy2(STAGE2, directory / "model.pt")
        shutil.copy2(STAGE2.parent / "policy_metadata.json", directory / "policy_metadata.json")

    for iteration in (1, 2, 3):
        assert manager.add_snapshot(iteration, save_copy)
        manager.promote_snapshot(iteration, float(iteration))
    metadata = manager.metadata()
    active = [entry for entry in metadata["entries"] if entry["active"]]
    assert [entry["label"] for entry in active[:2]] == ["stage1", "stage2"]
    assert {entry["label"] for entry in active[2:]} == {
        "self_play_iter_000002", "self_play_iter_000003"}
    assert len(active) == 4
    inactive = [entry for entry in metadata["entries"] if not entry["active"]]
    assert inactive[0]["label"] == "self_play_iter_000001"
    assert Path(inactive[0]["path"]).is_file()
    assert manager.add_snapshot(3, save_copy) is False


def test_parse_frozen_checkpoint_requires_label_and_valid_checkpoint() -> None:
    label, path = parse_frozen_checkpoint(f"stage2={STAGE2}")
    assert label == "stage2"
    assert path == STAGE2


def test_health_cleanup_deactivates_without_deleting_and_weights_anchors(tmp_path) -> None:
    anchors = [parse_frozen_checkpoint(f"stage1={STAGE1}"), parse_frozen_checkpoint(f"stage2={STAGE2}")]
    manager = FrozenSelfPlayPool(
        tmp_path / "run", anchors=anchors,
        seed_snapshots=[parse_frozen_checkpoint(f"healthy={STAGE2}"),
                        parse_frozen_checkpoint(f"bad={STAGE1}")],
        cap=4, snapshot_every=16)
    rejected = manager.apply_health_report({
        "healthy": {"promotion_score": 2.0, "rejection_reasons": []},
        "bad": {"promotion_score": 0.0, "rejection_reasons": ["frequent own goals"]},
    }, minimum_score=1.0)
    assert rejected == ["bad"]
    inactive = next(row for row in manager.metadata()["entries"] if row["label"] == "bad")
    assert Path(inactive["path"]).is_file()
    assert [row["label"] for row in manager.active_entries()] == ["stage1", "stage2", "healthy"]
    np.testing.assert_allclose(manager.active_sampling_weights(), [0.1875, 0.1875, 0.625])

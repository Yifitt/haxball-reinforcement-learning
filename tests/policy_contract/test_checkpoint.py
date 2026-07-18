from __future__ import annotations

import json
import sys
import time

import numpy as np
import pytest

from integration.controller.policies import CheckpointPolicy
from policy_contract.checkpoint_contract import (
    PortablePolicy,
    checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
)
from policy_contract.observation_contract import OBSERVATION_SIZE
from policy_contract.versions import OBSERVATION_VERSION

from .test_observations import equivalent_states


def metadata_for(model: PortablePolicy):
    return checkpoint_metadata(
        model,
        upstream_revision="1781914b79ccad18a65367d66de4645412405bce",
        action_repeat=8,
        seed=0,
        opponent="fixed_chase",
        reward_configuration={"goal": 10.0},
        n_envs=16,
        iterations=1,
        total_transitions=256,
        device="cpu",
        package_versions={"torch": "test"},
    )


def test_portable_checkpoint_save_load_and_valid_prediction(tmp_path) -> None:
    model = PortablePolicy(hidden=32, depth=2)
    model_path, metadata_path = save_checkpoint(tmp_path, model, metadata_for(model))
    loaded, metadata = load_checkpoint(model_path)
    action = loaded.predict_action(np.zeros(OBSERVATION_SIZE, dtype=np.float32), team=1)
    assert 0 <= action < 18
    assert metadata["observation_version"] == OBSERVATION_VERSION
    assert model_path.name == "model.pt"
    assert metadata_path.name == "policy_metadata.json"


def test_loader_rejects_incompatible_contract(tmp_path) -> None:
    model = PortablePolicy(hidden=16)
    _, metadata_path = save_checkpoint(tmp_path, model, metadata_for(model))
    metadata = json.loads(metadata_path.read_text())
    metadata["observation_size"] = 99
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="observation_size"):
        load_checkpoint(tmp_path)


def test_loader_keeps_browser_compatibility_with_v1_portable_checkpoint(tmp_path) -> None:
    model = PortablePolicy(hidden=16)
    _, metadata_path = save_checkpoint(tmp_path, model, metadata_for(model))
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_version"] = "portable-policy-v1"
    metadata.pop("training_environment", None)
    metadata_path.write_text(json.dumps(metadata))
    loaded, loaded_metadata = load_checkpoint(tmp_path)
    assert loaded.predict_bins(np.zeros((1, OBSERVATION_SIZE), dtype=np.float32)).shape == (1, 3)
    assert loaded_metadata["checkpoint_version"] == "portable-policy-v1"


def test_browser_checkpoint_policy_needs_no_simulator_import_and_is_fast(tmp_path) -> None:
    model = PortablePolicy(hidden=32)
    save_checkpoint(tmp_path, model, metadata_for(model))
    sys.modules.pop("haxball_core", None)
    sys.modules.pop("haxballgym", None)
    policy = CheckpointPolicy("controlled", tmp_path / "model.pt")
    _, browser = equivalent_states()
    started = time.perf_counter()
    actions = [policy.select_action(browser) for _ in range(100)]
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert all(0 <= action < 18 for action in actions)
    assert policy.median_inference_ms() is not None
    assert elapsed_ms / 100 < 50.0
    assert "haxball_core" not in sys.modules
    assert "haxballgym" not in sys.modules
    policy.reset()
    assert policy.median_inference_ms() is not None

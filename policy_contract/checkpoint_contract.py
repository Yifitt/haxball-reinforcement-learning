from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .action_contract import policy_bins_to_canonical
from .observation_contract import OBSERVATION_SIZE
from .versions import (
    ACTION_VERSION,
    CHECKPOINT_VERSION,
    OBSERVATION_VERSION,
    SUPPORTED_CHECKPOINT_VERSIONS,
)


class PortablePolicy(nn.Module):
    """Upstream-compatible three-head MLP without simulator imports."""

    def __init__(self, obs_dim: int = OBSERVATION_SIZE, hidden: int = 256, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden), nn.Tanh()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_x = nn.Linear(hidden, 3)
        self.head_y = nn.Linear(hidden, 3)
        self.head_k = nn.Linear(hidden, 2)
        self.value = nn.Linear(hidden, 1)
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.depth = depth

    def forward(self, observation: torch.Tensor):
        hidden = self.trunk(observation)
        return (
            self.head_x(hidden), self.head_y(hidden),
            self.head_k(hidden), self.value(hidden).squeeze(-1),
        )

    @torch.no_grad()
    def predict_bins(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape[-1] != self.obs_dim or not np.isfinite(array).all():
            raise ValueError("checkpoint received an incompatible observation")
        tensor = torch.as_tensor(array, dtype=torch.float32)
        logits_x, logits_y, logits_k, _ = self(tensor)
        bins = torch.stack(
            (logits_x.argmax(-1), logits_y.argmax(-1), logits_k.argmax(-1)), dim=-1)
        return bins.cpu().numpy()

    def predict_action(self, observation: np.ndarray, *, team: int) -> int:
        bins = self.predict_bins(observation)
        if bins.shape != (3,):
            raise ValueError("predict_action expects one observation")
        return policy_bins_to_canonical(bins, team=team)


def checkpoint_metadata(
    model: PortablePolicy,
    *,
    upstream_revision: str,
    action_repeat: int,
    seed: int,
    opponent: str,
    reward_configuration: dict[str, Any],
    n_envs: int,
    iterations: int,
    total_transitions: int,
    device: str,
    package_versions: dict[str, str],
    training_environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "upstream_haxballgym_revision": upstream_revision,
        "observation_version": OBSERVATION_VERSION,
        "observation_size": model.obs_dim,
        "action_version": ACTION_VERSION,
        "number_of_actions": 18,
        "network_architecture": {
            "type": "three_head_mlp",
            "hidden": model.hidden,
            "depth": model.depth,
            "heads": {"x": 3, "y": 3, "kick": 2},
        },
        "model_parameter_names": list(model.state_dict()),
        "action_repeat": action_repeat,
        "training_seed": seed,
        "opponent_type": opponent,
        "reward_configuration": reward_configuration,
        "environment_count": n_envs,
        "iterations": iterations,
        "total_transitions": total_transitions,
        "training_device": device,
        "package_versions": package_versions,
        "simulator": True,
        "training_environment": training_environment or {"version": "legacy-deterministic-v1"},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("checkpoint_version") not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            "incompatible checkpoint checkpoint_version: expected one of "
            f"{SUPPORTED_CHECKPOINT_VERSIONS!r}, got {metadata.get('checkpoint_version')!r}")
    expected = {
        "observation_version": OBSERVATION_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_version": ACTION_VERSION,
        "number_of_actions": 18,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"incompatible checkpoint {key}: expected {value!r}, got {metadata.get(key)!r}")
    architecture = metadata.get("network_architecture")
    if not isinstance(architecture, dict) or architecture.get("type") != "three_head_mlp":
        raise ValueError("unsupported checkpoint network architecture")


def save_checkpoint(directory: str | Path, model: PortablePolicy, metadata: dict[str, Any]) -> tuple[Path, Path]:
    validate_metadata(metadata)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.pt"
    metadata_path = target / "policy_metadata.json"
    torch.save(model.state_dict(), model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return model_path, metadata_path


def load_checkpoint(directory_or_model: str | Path) -> tuple[PortablePolicy, dict[str, Any]]:
    supplied = Path(directory_or_model)
    directory = supplied.parent if supplied.suffix == ".pt" else supplied
    model_path = supplied if supplied.suffix == ".pt" else directory / "model.pt"
    metadata_path = directory / "policy_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    validate_metadata(metadata)
    architecture = metadata["network_architecture"]
    model = PortablePolicy(
        obs_dim=metadata["observation_size"],
        hidden=int(architecture["hidden"]),
        depth=int(architecture["depth"]),
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if list(state_dict) != metadata["model_parameter_names"]:
        raise ValueError("checkpoint parameter names do not match metadata")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from policy_contract.checkpoint_contract import load_checkpoint

SELF_PLAY_POOL_VERSION = "frozen-self-play-pool-v2"


def parse_frozen_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("frozen checkpoints must use LABEL=PATH")
    label, supplied = value.split("=", 1)
    if not label or not supplied or not label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("frozen checkpoint labels must be non-empty letters, numbers, '_' or '-'")
    path = Path(supplied)
    model_path = path if path.suffix == ".pt" else path / "model.pt"
    if not model_path.is_file() or not (model_path.parent / "policy_metadata.json").is_file():
        raise ValueError(f"frozen checkpoint is incomplete: {model_path}")
    load_checkpoint(model_path)
    return label, model_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class FrozenSelfPlayPool:
    """Immutable checkpoint generations with anchors and a capped active window."""

    def __init__(
        self,
        checkpoint_root: str | Path,
        *,
        anchors: list[tuple[str, Path]],
        seed_snapshots: list[tuple[str, Path]] | None = None,
        cap: int,
        snapshot_every: int,
    ) -> None:
        if cap < len(anchors) + 1:
            raise ValueError("self-play pool cap must leave room for anchors and a snapshot")
        if snapshot_every < 1:
            raise ValueError("self-play snapshot interval must be positive")
        self.checkpoint_root = Path(checkpoint_root)
        self.pool_root = self.checkpoint_root / "self_play_pool"
        self.manifest_path = self.checkpoint_root / "self_play_pool_metadata.json"
        if self.manifest_path.exists():
            self._metadata = json.loads(self.manifest_path.read_text())
            if self._metadata.get("version") == "frozen-self-play-pool-v1":
                self._metadata["version"] = SELF_PLAY_POOL_VERSION
                self._metadata["anchor_probability"] = 0.30
                self._metadata["healthy_self_play_probability"] = 0.50
                for entry in self._metadata["entries"]:
                    entry.setdefault("health", "unrated")
                    entry.setdefault("deactivation_reason", None)
                    entry.setdefault("promotion_score", None)
                self._write()
            elif self._metadata.get("version") != SELF_PLAY_POOL_VERSION:
                raise ValueError("incompatible self-play pool metadata version")
            if int(self._metadata["cap"]) != cap:
                raise ValueError("resume self-play pool cap does not match metadata")
            if int(self._metadata["snapshot_every"]) != snapshot_every:
                raise ValueError("resume snapshot interval does not match metadata")
            self._verify_entries()
        else:
            labels = [label for label, _ in anchors]
            if len(labels) != len(set(labels)) or len(anchors) < 2:
                raise ValueError("Stage 4 requires at least two uniquely labelled frozen anchors")
            self._metadata = {
                "version": SELF_PLAY_POOL_VERSION,
                "checkpoint_probability": 0.8,
                "hard_rule_based_probability": 0.2,
                "anchor_probability": 0.30,
                "healthy_self_play_probability": 0.50,
                "cap": cap,
                "snapshot_every": snapshot_every,
                "entries": [],
            }
            for label, source in anchors:
                self._copy_anchor(label, source)
            for label, source in (seed_snapshots or []):
                self._copy_seed_snapshot(label, source)
            self._enforce_cap()
            self._write()

    def _copy_anchor(self, label: str, source_model: Path) -> None:
        target = self.pool_root / f"anchor_{label}"
        if target.exists():
            raise FileExistsError(f"refusing to update frozen checkpoint in place: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            shutil.copy2(source_model, temporary / "model.pt")
            shutil.copy2(source_model.parent / "policy_metadata.json", temporary / "policy_metadata.json")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._metadata["entries"].append({
            "label": label,
            "kind": "anchor",
            "created_iteration": None,
            "path": str(target / "model.pt"),
            "sha256": _sha256(target / "model.pt"),
            "active": True,
            "health": "permanent_anchor",
            "deactivation_reason": None,
            "promotion_score": None,
        })

    def _verify_entries(self) -> None:
        for entry in self._metadata["entries"]:
            path = Path(entry["path"])
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"frozen self-play checkpoint changed or is missing: {path}")

    def _copy_seed_snapshot(self, label: str, source_model: Path) -> None:
        if any(entry["label"] == label for entry in self._metadata["entries"]):
            raise ValueError(f"duplicate frozen checkpoint label: {label}")
        target = self.pool_root / label
        if target.exists():
            raise FileExistsError(f"refusing to update frozen checkpoint in place: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            shutil.copy2(source_model, temporary / "model.pt")
            shutil.copy2(source_model.parent / "policy_metadata.json", temporary / "policy_metadata.json")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._metadata["entries"].append({
            "label": label, "kind": "self_play", "created_iteration": 0,
            "path": str(target / "model.pt"), "sha256": _sha256(target / "model.pt"),
            "active": True, "health": "tournament_approved_seed",
            "deactivation_reason": None, "promotion_score": None,
        })

    def _write(self) -> None:
        _atomic_json(self.manifest_path, self._metadata)

    def metadata(self) -> dict[str, object]:
        return json.loads(json.dumps(self._metadata))

    def active_entries(self) -> list[dict[str, object]]:
        return [entry for entry in self._metadata["entries"] if entry["active"]]

    def load_active(self) -> tuple[tuple[object, ...], tuple[str, ...]]:
        entries = self.active_entries()
        models = tuple(load_checkpoint(entry["path"])[0] for entry in entries)
        labels = tuple(str(entry["label"]) for entry in entries)
        return models, labels

    def active_sampling_weights(self) -> tuple[float, ...]:
        """Conditional checkpoint weights: 30% anchors, 50% healthy self-play overall."""
        entries = self.active_entries()
        anchors = [entry for entry in entries if entry["kind"] == "anchor"]
        snapshots = [entry for entry in entries if entry["kind"] == "self_play"]
        if not anchors:
            raise ValueError("self-play pool has no permanent anchors")
        anchor_share = 0.30 / 0.80
        snapshot_share = 1.0 - anchor_share if snapshots else 0.0
        if not snapshots:
            anchor_share = 1.0
        return tuple(
            anchor_share / len(anchors) if entry["kind"] == "anchor"
            else snapshot_share / len(snapshots)
            for entry in entries
        )

    def apply_health_report(
        self, health_by_label: dict[str, dict[str, object]], *, minimum_score: float
    ) -> list[str]:
        """Deactivate unhealthy generations while preserving every immutable file."""
        rejected: list[str] = []
        for entry in self._metadata["entries"]:
            if entry["kind"] == "anchor":
                entry["active"] = True
                entry["health"] = "permanent_anchor"
                continue
            health = health_by_label.get(str(entry["label"]))
            if health is None:
                entry["active"] = False
                entry["health"] = "unrated"
                entry["deactivation_reason"] = "not present in deterministic tournament"
                rejected.append(str(entry["label"]))
                continue
            score = float(health["promotion_score"])
            reasons = list(health.get("rejection_reasons", []))
            healthy = not reasons and score >= minimum_score
            entry["active"] = healthy
            entry["health"] = "healthy" if healthy else "rejected"
            entry["promotion_score"] = score
            entry["deactivation_reason"] = None if healthy else "; ".join(reasons) or "low tournament score"
            if not healthy:
                rejected.append(str(entry["label"]))
        self._enforce_cap()
        self._write()
        return rejected

    def _enforce_cap(self) -> None:
        active_snapshots = [
            entry for entry in self._metadata["entries"]
            if entry["kind"] == "self_play" and entry["active"]
        ]
        anchor_count = sum(entry["kind"] == "anchor" for entry in self._metadata["entries"])
        excess = max(0, len(active_snapshots) - (int(self._metadata["cap"]) - anchor_count))
        for entry in sorted(
            active_snapshots,
            key=lambda row: (float(row.get("promotion_score") or -1e9), int(row["created_iteration"])),
        )[:excess]:
            entry["active"] = False
            entry["health"] = "cap_pruned"
            entry["deactivation_reason"] = "lower-ranked healthy snapshot outside active cap"

    def add_snapshot(
        self,
        iteration: int,
        save_callback: Callable[[Path], None],
    ) -> bool:
        label = f"self_play_iter_{iteration:06d}"
        if any(entry["label"] == label for entry in self._metadata["entries"]):
            return False
        target = self.pool_root / label
        if target.exists():
            raise FileExistsError(f"refusing to update frozen checkpoint in place: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{label}-", dir=target.parent))
        try:
            save_callback(temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._metadata["entries"].append({
            "label": label,
            "kind": "self_play",
            "created_iteration": iteration,
            "path": str(target / "model.pt"),
            "sha256": _sha256(target / "model.pt"),
            "active": False,
            "health": "pending_tournament",
            "deactivation_reason": None,
            "promotion_score": None,
        })
        self._enforce_cap()
        self._write()
        return True

    def promote_snapshot(self, iteration: int, promotion_score: float) -> None:
        label = f"self_play_iter_{iteration:06d}"
        matches = [entry for entry in self._metadata["entries"] if entry["label"] == label]
        if len(matches) != 1 or matches[0]["kind"] != "self_play":
            raise ValueError(f"unknown self-play snapshot: {label}")
        entry = matches[0]
        entry["active"] = True
        entry["health"] = "promoted"
        entry["promotion_score"] = float(promotion_score)
        entry["deactivation_reason"] = None
        self._enforce_cap()
        self._write()


def load_active_pool(manifest_path: str | Path) -> tuple[tuple[object, ...], tuple[str, ...], dict[str, object]]:
    metadata = json.loads(Path(manifest_path).read_text())
    if metadata.get("version") not in (SELF_PLAY_POOL_VERSION, "frozen-self-play-pool-v1"):
        raise ValueError("incompatible self-play pool metadata version")
    entries = [entry for entry in metadata["entries"] if entry["active"]]
    for entry in entries:
        path = Path(entry["path"])
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"frozen self-play checkpoint changed: {path}")
    return (
        tuple(load_checkpoint(entry["path"])[0] for entry in entries),
        tuple(str(entry["label"]) for entry in entries),
        metadata,
    )

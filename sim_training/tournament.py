from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from policy_contract.checkpoint_contract import load_checkpoint
from sim_training.evaluate import HELD_OUT_SEEDS, _evaluate_batch
from sim_training.opponent_pool import OpponentPoolConfiguration
from sim_training.promotion import promotion_decision
from sim_training.self_play_pool import FrozenSelfPlayPool, parse_frozen_checkpoint

HARD_OPPONENTS = ("defensive_chase", "aggressive_chase")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_candidates(root: Path) -> list[dict[str, object]]:
    manifest = json.loads((root / "self_play_pool_metadata.json").read_text())
    paths = sorted(root.glob("**/model.pt"))
    by_hash: dict[str, dict[str, object]] = {}
    for path in paths:
        sha = _sha256(path)
        row = by_hash.setdefault(sha, {"sha256": sha, "aliases": []})
        row["aliases"].append(str(path))
    label_by_sha = {str(entry["sha256"]): str(entry["label"]) for entry in manifest["entries"]}
    for row in by_hash.values():
        row["label"] = label_by_sha.get(str(row["sha256"]), Path(row["aliases"][0]).parent.name)
        preferred = next((
            alias for alias in row["aliases"]
            if "/periodic/" in alias and (Path(alias).parent / "trainer_state.pt").is_file()
        ), next((
            alias for alias in row["aliases"] if "/self_play_pool/" in alias
        ), row["aliases"][0]))
        row["path"] = preferred
    return sorted(by_hash.values(), key=lambda row: str(row["label"]))


def _candidate_report(
    candidate_path: Path,
    opponent_models: tuple[object, ...],
    opponent_labels: tuple[str, ...],
    *,
    repetitions: int,
    max_decisions: int,
) -> dict[str, object]:
    model, metadata = load_checkpoint(candidate_path)
    matchup_labels = opponent_labels + HARD_OPPONENTS
    totals: dict[str, float] = defaultdict(float)
    groups: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for seed in HELD_OUT_SEEDS:
        labels = np.asarray([
            label for label in matchup_labels for side in (0, 1)
            for _ in range(repetitions)
        ], dtype=object)
        sides = np.asarray([
            side for _label in matchup_labels for side in (0, 1)
            for _ in range(repetitions)
        ], dtype=np.int64)
        batch = _evaluate_batch(
            model, count=len(labels), seed=seed,
            action_repeat=int(metadata["action_repeat"]), max_decisions=max_decisions,
            random_learner_side=False,
            opponent_configuration=OpponentPoolConfiguration(("self_play",), (1.0,)),
            opponent_models=opponent_models, opponent_labels=opponent_labels,
            forced_learner_sides=sides, forced_opponent_labels=labels,
        )
        for key in (
            "episodes", "goals_for", "goals_against", "agent_normal_goals",
            "opponent_normal_goals", "agent_own_goals", "opponent_own_goals",
            "wins", "decision_total", "episode_length_total", "requested_total",
            "valid_total", "invalid_total", "held_invalid_total", "successful_total",
            "entropy_total",
        ):
            totals[key] += float(batch[key])
        totals["maximum_invalid_streak"] = max(
            totals["maximum_invalid_streak"], float(batch["maximum_invalid_streak"]))
        for label, values in batch["by_opponent"].items():
            for key, value in values.items():
                groups[str(label)][key] += float(value)
    episodes = int(totals["episodes"])
    decisions = int(totals["decision_total"])
    requests = int(totals["requested_total"])
    by_opponent: dict[str, dict[str, float | int]] = {}
    for label, values in sorted(groups.items()):
        n = int(values["episodes"])
        by_opponent[label] = {
            "episodes": n,
            "agent_normal_goals": int(values["agent_normal_goals"]),
            "opponent_normal_goals": int(values["opponent_normal_goals"]),
            "agent_own_goals": int(values["agent_own_goals"]),
            "opponent_own_goals": int(values["opponent_own_goals"]),
            "win_rate": values["wins"] / n,
        }
    checkpoint_rates = [float(by_opponent[label]["win_rate"]) for label in opponent_labels]
    hard_episodes = sum(int(by_opponent[label]["episodes"]) for label in HARD_OPPONENTS)
    hard_wins = sum(float(groups[label]["wins"]) for label in HARD_OPPONENTS)
    return {
        "episodes": episodes,
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "both_learner_sides": True,
        "agent_normal_goals": int(totals["agent_normal_goals"]),
        "opponent_normal_goals": int(totals["opponent_normal_goals"]),
        "goals_conceded": int(totals["goals_against"]),
        "agent_own_goals": int(totals["agent_own_goals"]),
        "opponent_own_goals": int(totals["opponent_own_goals"]),
        "win_rate": totals["wins"] / episodes,
        "worst_checkpoint_win_rate": min(checkpoint_rates),
        "stage2_win_rate": float(by_opponent["stage2"]["win_rate"]),
        "hard_rule_win_rate": hard_wins / hard_episodes,
        "mean_episode_length": totals["episode_length_total"] / episodes,
        "policy_entropy": totals["entropy_total"] / decisions,
        "kick_request_fraction": requests / decisions,
        "valid_kick_fraction_of_requests": totals["valid_total"] / requests if requests else 0.0,
        "invalid_kick_fraction_of_requests": totals["invalid_total"] / requests if requests else 0.0,
        "invalid_kick_fraction_of_decisions": totals["invalid_total"] / decisions,
        "held_invalid_kick_fraction": totals["held_invalid_total"] / decisions,
        "successful_contact_fraction": totals["successful_total"] / requests if requests else 0.0,
        "consecutive_invalid_kick_streak": int(totals["maximum_invalid_streak"]),
        "by_opponent": by_opponent,
    }


def run_tournament(root: Path, *, repetitions: int = 2, max_decisions: int = 400,
                   apply_pool_cleanup: bool = False) -> dict[str, object]:
    manifest_path = root / "self_play_pool_metadata.json"
    manifest = json.loads(manifest_path.read_text())
    original_active = [
        entry for entry in manifest["entries"]
        if entry["active"] and entry["kind"] == "self_play"
    ]
    opponent_entries = [
        entry for entry in manifest["entries"]
        if entry["kind"] == "anchor" or entry in original_active
    ]
    opponent_models = tuple(load_checkpoint(entry["path"])[0] for entry in opponent_entries)
    opponent_labels = tuple(str(entry["label"]) for entry in opponent_entries)
    candidates = discover_candidates(root)
    results: list[dict[str, object]] = []
    for candidate in candidates:
        report = _candidate_report(
            Path(candidate["path"]), opponent_models, opponent_labels,
            repetitions=repetitions, max_decisions=max_decisions,
        )
        decision = promotion_decision(report)
        results.append({**candidate, "metrics": report, **decision})
    eligible = [row for row in results if row["promoted"]]
    ranked = sorted(eligible or results, key=lambda row: float(row["promotion_score"]), reverse=True)
    strongest = ranked[0]

    snapshot_results = {
        str(row["label"]): row for row in results if str(row["label"]).startswith("self_play_iter_")
    }
    scores = sorted(float(row["promotion_score"]) for row in snapshot_results.values())
    health_floor = scores[max(0, len(scores) // 4 - 1)] if scores else float("-inf")
    health_report: dict[str, dict[str, object]] = {}
    for label, row in snapshot_results.items():
        metrics = row["metrics"]
        reasons: list[str] = []
        if int(metrics["agent_own_goals"]) / int(metrics["episodes"]) > 0.05:
            reasons.append("frequent own goals")
        if float(metrics["win_rate"]) < 0.45:
            reasons.append("trivially exploitable")
        if float(metrics["stage2_win_rate"]) < 0.55:
            reasons.append("catastrophic Stage 2 forgetting")
        if float(metrics["invalid_kick_fraction_of_requests"]) > 0.80:
            reasons.append("degenerate raw-kick behavior")
        health_report[label] = {
            "promotion_score": row["promotion_score"],
            "rejection_reasons": reasons,
        }
    rejected: list[str] = []
    if apply_pool_cleanup:
        anchors = [
            parse_frozen_checkpoint(f"{entry['label']}={entry['path']}")
            for entry in manifest["entries"] if entry["kind"] == "anchor"
        ]
        manager = FrozenSelfPlayPool(
            root, anchors=anchors, cap=int(manifest["cap"]),
            snapshot_every=int(manifest["snapshot_every"]),
        )
        rejected = manager.apply_health_report(health_report, minimum_score=health_floor)
    report = {
        "protocol": "held-out-balanced-stage4-tournament-v1",
        "candidate_count": len(results),
        "episodes_per_candidate": int(results[0]["metrics"]["episodes"]),
        "opponents": list(opponent_labels + HARD_OPPONENTS),
        "strongest": strongest,
        "ranked_candidates": ranked,
        "all_candidates": results,
        "pool_cleanup_applied": apply_pool_cleanup,
        "health_score_floor": health_floor,
        "rejected_snapshots": rejected,
    }
    output = root / "tournament_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--apply-pool-cleanup", action="store_true")
    args = parser.parse_args()
    report = run_tournament(
        args.checkpoint_root, repetitions=args.repetitions,
        max_decisions=args.max_decisions, apply_pool_cleanup=args.apply_pool_cleanup,
    )
    print(json.dumps({
        "report": str(args.checkpoint_root / "tournament_report.json"),
        "strongest": report["strongest"],
        "rejected_snapshots": report["rejected_snapshots"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

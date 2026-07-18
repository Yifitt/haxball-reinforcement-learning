from __future__ import annotations

import pytest

from policy_contract.checkpoint_contract import PortablePolicy
from sim_training.evaluate import HELD_OUT_SEEDS, evaluate_model
from sim_training.curriculum import configuration_for_stage


def test_heldout_evaluation_requires_100_episodes() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_model(
            PortablePolicy(hidden=8), episodes=99, n_envs=16,
            action_repeat=8, max_decisions=300)


def test_heldout_evaluation_reports_generalization_metrics() -> None:
    report = evaluate_model(
        PortablePolicy(hidden=8), episodes=100, n_envs=32,
        action_repeat=8, max_decisions=300)
    assert report["episodes"] == 100
    assert set(report["held_out_seeds"]) == set(HELD_OUT_SEEDS)
    assert report["goals_for"] + report["goals_against"] <= 100
    assert 0.0 <= report["win_rate"] <= 1.0
    assert 0 <= report["own_goals"] <= 100
    assert 0 <= report["unique_action_count"] <= 18
    assert report["mean_episode_length"] > 0
    assert 0.0 <= report["invalid_kick_fraction_of_requests"] <= 1.0
    assert 0.0 <= report["invalid_kick_fraction_of_decisions"] <= 1.0
    assert set(report["score_by_opponent_type"]) == {
        "standard_chase", "delayed_chase", "noisy_chase", "defensive_chase",
        "aggressive_chase", "mixed_persistent_random_chase",
    }
    assert report["score_by_kickoff_configuration"]


def test_stage4_evaluation_randomizes_side_and_scores_checkpoint_generations() -> None:
    report = evaluate_model(
        PortablePolicy(hidden=8), episodes=100, n_envs=32,
        action_repeat=8, max_decisions=300, random_learner_side=True,
        opponent_configuration=configuration_for_stage(4),
        opponent_models=(PortablePolicy(hidden=8), PortablePolicy(hidden=8)),
        opponent_labels=("stage1", "stage2"))
    assert set(report["score_by_learner_side"]) == {"red", "blue"}
    assert set(report["score_by_checkpoint_generation"]) == {"stage1", "stage2"}
    assert report["score_vs_hard_rule_based"]["episodes"] > 0
    assert "agent_own_goals" in report
    assert "opponent_own_goals" in report

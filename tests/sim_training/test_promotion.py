from __future__ import annotations

import pytest

from sim_training.promotion import promotion_decision, promotion_score


def healthy(**updates):
    report = {
        "episodes": 100, "agent_normal_goals": 62, "opponent_normal_goals": 30,
        "agent_own_goals": 2, "opponent_own_goals": 5,
        "stage2_win_rate": 0.65, "hard_rule_win_rate": 0.90,
        "worst_checkpoint_win_rate": 0.40,
        "invalid_kick_fraction_of_requests": 0.60, "policy_entropy": 1.0,
    }
    report.update(updates)
    return report


@pytest.mark.parametrize("update", [
    {"agent_own_goals": 6}, {"stage2_win_rate": 0.54},
    {"hard_rule_win_rate": 0.79}, {"worst_checkpoint_win_rate": 0.14},
    {"invalid_kick_fraction_of_requests": 0.76}, {"policy_entropy": 0.24},
])
def test_each_promotion_safety_gate_rejects(update) -> None:
    assert not promotion_decision(healthy(**update))["promoted"]


def test_opponent_own_goals_do_not_increase_score_and_champion_is_retained() -> None:
    assert promotion_score(healthy(opponent_own_goals=99)) == promotion_score(healthy())
    weak = healthy(agent_normal_goals=58, opponent_normal_goals=34)
    decision = promotion_decision(weak, champion_report=healthy())
    assert not decision["promoted"]
    assert any("champion" in reason for reason in decision["rejection_reasons"])

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PromotionThresholds:
    maximum_agent_own_goal_rate: float = 0.05
    minimum_stage2_win_rate: float = 0.55
    minimum_hard_rule_win_rate: float = 0.80
    minimum_worst_checkpoint_win_rate: float = 0.15
    maximum_invalid_kick_fraction_of_requests: float = 0.75
    minimum_policy_entropy: float = 0.25
    maximum_score_regression_from_champion: float = 0.05


DEFAULT_THRESHOLDS = PromotionThresholds()


def promotion_score(report: dict[str, object]) -> float:
    """Earned goals and breadth matter; opponent own goals deliberately contribute zero."""
    episodes = max(1, int(report["episodes"]))
    return (
        2.0 * int(report["agent_normal_goals"]) / episodes
        - 2.0 * int(report["opponent_normal_goals"]) / episodes
        - 6.0 * int(report["agent_own_goals"]) / episodes
        + 1.5 * float(report["stage2_win_rate"])
        + 1.0 * float(report["hard_rule_win_rate"])
        + 2.0 * float(report["worst_checkpoint_win_rate"])
        - 0.5 * float(report["invalid_kick_fraction_of_requests"])
    )


def promotion_decision(
    report: dict[str, object],
    *,
    champion_report: dict[str, object] | None = None,
    thresholds: PromotionThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    episodes = max(1, int(report["episodes"]))
    own_goal_rate = int(report["agent_own_goals"]) / episodes
    score = promotion_score(report)
    reasons: list[str] = []
    gates = (
        (own_goal_rate <= thresholds.maximum_agent_own_goal_rate,
         f"agent own-goal rate {own_goal_rate:.4f} exceeds {thresholds.maximum_agent_own_goal_rate:.4f}"),
        (float(report["stage2_win_rate"]) >= thresholds.minimum_stage2_win_rate,
         f"Stage 2 win rate {float(report['stage2_win_rate']):.4f} below {thresholds.minimum_stage2_win_rate:.4f}"),
        (float(report["hard_rule_win_rate"]) >= thresholds.minimum_hard_rule_win_rate,
         f"hard-rule win rate {float(report['hard_rule_win_rate']):.4f} below {thresholds.minimum_hard_rule_win_rate:.4f}"),
        (float(report["worst_checkpoint_win_rate"]) >= thresholds.minimum_worst_checkpoint_win_rate,
         f"worst-checkpoint win rate {float(report['worst_checkpoint_win_rate']):.4f} below {thresholds.minimum_worst_checkpoint_win_rate:.4f}"),
        (float(report["invalid_kick_fraction_of_requests"]) <= thresholds.maximum_invalid_kick_fraction_of_requests,
         "invalid kick requests dominate policy behavior"),
        (float(report["policy_entropy"]) >= thresholds.minimum_policy_entropy,
         f"policy entropy {float(report['policy_entropy']):.4f} collapsed"),
    )
    reasons.extend(message for passed, message in gates if not passed)
    champion_score = None
    if champion_report is not None:
        champion_score = promotion_score(champion_report)
        if score < champion_score - thresholds.maximum_score_regression_from_champion:
            reasons.append(
                f"promotion score {score:.4f} substantially worse than champion {champion_score:.4f}")
    return {
        "promoted": not reasons,
        "promotion_score": score,
        "champion_score": champion_score,
        "agent_own_goal_rate": own_goal_rate,
        "rejection_reasons": reasons,
        "thresholds": asdict(thresholds),
    }

"""Isolated reward calculation for easy later tuning."""

from __future__ import annotations

from .config import EnvConfig


def calculate_reward(
    *,
    side: int,
    goal: int | None,
    previous_ball_x: float,
    current_ball_x: float,
    touched_or_kicked: bool,
    config: EnvConfig,
) -> tuple[float, dict[str, float]]:
    coefficients = config.rewards
    components = {
        "goal": coefficients.goal if goal == side else 0.0,
        "concede": coefficients.concede if goal == 1 - side else 0.0,
        "ball_progress": 0.0,
        "touch": coefficients.touch if touched_or_kicked else 0.0,
        "inactivity": 0.0,
    }
    oriented_delta = (current_ball_x - previous_ball_x) * (1.0 if side == 0 else -1.0)
    normalized_delta = oriented_delta / config.field_width
    coefficient = coefficients.ball_progress if normalized_delta >= 0.0 else coefficients.ball_regress
    components["ball_progress"] = coefficient * normalized_delta
    if abs(oriented_delta) < 1e-9 and not touched_or_kicked:
        components["inactivity"] = coefficients.inactivity
    return float(sum(components.values())), components

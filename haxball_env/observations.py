"""Normalized, side-invariant observations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import EnvConfig
from .physics import GameState

OBSERVATION_SIZE = 21


def mirror_for_side(vector: NDArray[np.floating], side: int) -> NDArray[np.float64]:
    """Mirror x for the right-side player so it always attacks toward +x."""
    if side not in (0, 1):
        raise ValueError("side must be 0 or 1")
    result = np.asarray(vector, dtype=np.float64).copy()
    if result.shape != (2,):
        raise ValueError("vector must have shape (2,)")
    if side == 1:
        result[0] *= -1.0
    return result


def build_observation(
    state: GameState,
    side: int,
    scores: NDArray[np.integer],
    remaining_time_fraction: float,
    config: EnvConfig,
) -> NDArray[np.float32]:
    other = 1 - side
    position_scale = np.array([config.field_width / 2.0, config.field_height / 2.0])
    distance_scale = np.array([config.field_width, config.field_height])

    player_position = mirror_for_side(state.player_positions[side], side)
    player_velocity = mirror_for_side(state.player_velocities[side], side)
    opponent_position = mirror_for_side(state.player_positions[other], side)
    opponent_velocity = mirror_for_side(state.player_velocities[other], side)
    ball_position = mirror_for_side(state.ball_position, side)
    ball_velocity = mirror_for_side(state.ball_velocity, side)
    opponent_goal = np.array([config.field_width / 2.0, 0.0])
    own_goal = np.array([-config.field_width / 2.0, 0.0])

    observation = np.concatenate(
        (
            player_position / position_scale,
            player_velocity / config.player_max_speed,
            opponent_position / position_scale,
            opponent_velocity / config.player_max_speed,
            ball_position / position_scale,
            ball_velocity / config.ball_max_speed,
            (ball_position - player_position) / distance_scale,
            (opponent_goal - ball_position) / distance_scale,
            (own_goal - player_position) / distance_scale,
            np.array([1.0 if state.kick_cooldowns[side] <= 0.0 else 0.0]),
            np.array([(int(scores[side]) - int(scores[other])) / config.score_limit]),
            np.array([remaining_time_fraction]),
        )
    )
    return np.clip(observation, -1.0, 1.0).astype(np.float32)

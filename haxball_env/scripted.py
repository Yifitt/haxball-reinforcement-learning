"""A dependency-free chase/defend scripted opponent."""

from __future__ import annotations

import numpy as np

from .actions import direction_to_action
from .config import EnvConfig
from .physics import GameState


class ScriptedOpponent:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def act(self, state: GameState, side: int) -> int:
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        player = state.player_positions[side]
        ball = state.ball_position
        own_goal = np.array(
            [-self.config.field_width / 2.0 if side == 0 else self.config.field_width / 2.0, 0.0]
        )
        distance_to_ball = float(np.linalg.norm(ball - player))

        defensive_half = ball[0] < 0.0 if side == 0 else ball[0] > 0.0
        if defensive_half and distance_to_ball > self.config.kick_range:
            goalward = own_goal - ball
            goalward /= max(float(np.linalg.norm(goalward)), 1e-9)
            target = ball + goalward * (2.0 * self.config.player_radius)
        else:
            target = ball
        kick = distance_to_ball <= self.config.kick_range
        return direction_to_action(target - player, kick=kick)

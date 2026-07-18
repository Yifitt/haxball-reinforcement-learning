from __future__ import annotations

from typing import Any

import numpy as np

from integration.controller.protocol import StateMessage

OBSERVATION_SIZE = 16
POSITION_COEFFICIENT = np.asarray((1.0 / 420.0, 1.0 / 200.0), dtype=np.float64)
VELOCITY_COEFFICIENT = 1.0 / 10.0
GOAL_X = 370.0


def _finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _build(
    player_pos: np.ndarray,
    player_vel: np.ndarray,
    opponent_pos: np.ndarray,
    opponent_vel: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    team: np.ndarray,
) -> np.ndarray:
    player_pos = _finite("player_pos", player_pos)
    player_vel = _finite("player_vel", player_vel)
    opponent_pos = _finite("opponent_pos", opponent_pos)
    opponent_vel = _finite("opponent_vel", opponent_vel)
    ball_pos = _finite("ball_pos", ball_pos)
    ball_vel = _finite("ball_vel", ball_vel)
    team = np.asarray(team, dtype=np.int64)
    if player_pos.shape[-1:] != (2,) or team.shape != player_pos.shape[:-1]:
        raise ValueError("observation bodies must end in xy and team must match their leading shape")
    if not np.isin(team, (1, 2)).all():
        raise ValueError("team must use browser IDs 1 (Red) or 2 (Blue)")
    for name, array in (
        ("player_vel", player_vel), ("opponent_pos", opponent_pos),
        ("opponent_vel", opponent_vel), ("ball_pos", ball_pos), ("ball_vel", ball_vel),
    ):
        if array.shape != player_pos.shape:
            raise ValueError(f"{name} shape does not match player_pos")

    target_x = np.where(team == 1, GOAL_X, -GOAL_X)
    target = np.stack((target_x, np.zeros_like(target_x)), axis=-1)
    own = -target
    out = np.concatenate(
        (
            player_pos * POSITION_COEFFICIENT,
            player_vel * VELOCITY_COEFFICIENT,
            (ball_pos - player_pos) * POSITION_COEFFICIENT,
            ball_vel * VELOCITY_COEFFICIENT,
            (target - player_pos) * POSITION_COEFFICIENT,
            (own - player_pos) * POSITION_COEFFICIENT,
            (opponent_pos - player_pos) * POSITION_COEFFICIENT,
            opponent_vel * VELOCITY_COEFFICIENT,
        ),
        axis=-1,
    )
    mirror = np.where(team == 1, 1.0, -1.0)[..., None]
    out[..., 0::2] *= mirror
    result = out.astype(np.float32)
    if result.shape[-1] != OBSERVATION_SIZE or not np.isfinite(result).all():
        raise ValueError("invalid shared policy observation")
    return result


def build_sim_observation(state: Any) -> np.ndarray:
    """Build contract observations from an upstream HaxballGym `GameState`."""
    positions = _finite("state.player_pos", state.player_pos)
    velocities = _finite("state.player_vel", state.player_vel)
    teams = np.asarray(state.team)
    if positions.ndim != 3 or positions.shape[1] != 2:
        raise ValueError("the v1 contract supports exactly 1v1 simulator states")
    opponent_positions = positions[:, ::-1, :]
    opponent_velocities = velocities[:, ::-1, :]
    ball_positions = np.broadcast_to(_finite("state.ball_pos", state.ball_pos)[:, None, :], positions.shape)
    ball_velocities = np.broadcast_to(_finite("state.ball_vel", state.ball_vel)[:, None, :], positions.shape)
    # Upstream team flags are RED=2 and BLUE=4; browser protocol uses 1 and 2.
    browser_teams = np.where(teams == 2, 1, np.where(teams == 4, 2, 0))
    return _build(
        positions, velocities, opponent_positions, opponent_velocities,
        ball_positions, ball_velocities, browser_teams,
    )


def build_browser_observation(state: StateMessage, *, role: str = "controlled") -> np.ndarray:
    """Build the same observation from real Headless Host numeric state.

    Real HaxBall host y grows downward; the Rust simulator y grows upward. Only y
    positions and y velocities are negated here. Horizontal team mirroring is the
    shared transformation in `_build`.
    """
    if role not in ("controlled", "opponent"):
        raise ValueError(f"unknown browser role: {role}")
    player = state.controlled_player if role == "controlled" else state.opponent_player
    opponent = state.opponent_player if role == "controlled" else state.controlled_player
    ball = state.ball
    if not state.game_active or player is None or opponent is None or ball is None:
        raise ValueError("active mapped players and ball are required for policy inference")

    def canonical_body(body: Any) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray((body.x, -body.y), dtype=np.float64),
            np.asarray((body.vx, -body.vy), dtype=np.float64),
        )

    player_pos, player_vel = canonical_body(player)
    opponent_pos, opponent_vel = canonical_body(opponent)
    ball_pos, ball_vel = canonical_body(ball)
    return _build(
        player_pos, player_vel, opponent_pos, opponent_vel,
        ball_pos, ball_vel, np.asarray(player.team),
    )

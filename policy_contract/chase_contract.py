from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RED_TEAM = 1
BLUE_TEAM = 2
ACTUAL_KICK_RANGE = 29.0


@dataclass(frozen=True)
class ChaseDecision:
    dx: np.ndarray
    dy: np.ndarray
    kick: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    opponent_goal_x: np.ndarray


def goal_directed_chase(
    player_pos: np.ndarray,
    ball_pos: np.ndarray,
    team: np.ndarray | int,
    *,
    behind_ball_distance: float = 24.0,
    alignment_tolerance: float = 15.0,
    kick_range: float = ACTUAL_KICK_RANGE,
    dead_zone: float = 2.5,
) -> ChaseDecision:
    """Return physical-field chase directions for browser team IDs 1/2.

    Red attacks the Blue goal at +x and Blue attacks the Red goal at -x. A
    player on the wrong (opponent-goal) side of the ball first routes around to
    a goal-side staging point. This prevents a naive ball chase from pushing
    the ball into the acting player's own goal.
    """
    player = np.asarray(player_pos, dtype=np.float64)
    ball = np.asarray(ball_pos, dtype=np.float64)
    teams = np.asarray(team, dtype=np.int64)
    if player.shape[-1:] != (2,) or ball.shape != player.shape:
        raise ValueError("chase player_pos and ball_pos must have matching (..., 2) shapes")
    if teams.shape != player.shape[:-1] or not np.isin(teams, (RED_TEAM, BLUE_TEAM)).all():
        raise ValueError("chase team must use browser IDs 1 (Red) or 2 (Blue)")

    attack = np.where(teams == RED_TEAM, 1.0, -1.0)
    opponent_goal_x = attack * 370.0
    to_ball = ball - player
    distance = np.linalg.norm(to_ball, axis=-1)
    aligned = np.abs(to_ball[..., 1]) <= alignment_tolerance
    goal_side = (player[..., 0] - ball[..., 0]) * attack <= 0.0

    staging_x = ball[..., 0] - attack * behind_ball_distance
    route_clearance = ACTUAL_KICK_RANGE + 7.0
    close_to_ball_line = np.abs(to_ball[..., 1]) < route_clearance
    route_sign = np.where(to_ball[..., 1] >= 0.0, -1.0, 1.0)
    staging_y = ball[..., 1] + route_sign * route_clearance
    target_x = np.where(
        goal_side,
        np.where(aligned, ball[..., 0] + attack * 12.0, ball[..., 0]),
        np.where(close_to_ball_line, player[..., 0], staging_x),
    )
    target_y = np.where(goal_side, ball[..., 1], staging_y)
    target_delta = np.stack((target_x, target_y), axis=-1) - player
    dx = np.where(np.abs(target_delta[..., 0]) <= dead_zone, 0, np.sign(target_delta[..., 0])).astype(np.int64)
    dy = np.where(np.abs(target_delta[..., 1]) <= dead_zone, 0, np.sign(target_delta[..., 1])).astype(np.int64)
    kick = ((distance < kick_range) & aligned & goal_side).astype(np.int64)
    return ChaseDecision(dx, dy, kick, target_x, target_y, opponent_goal_x)


def kick_request_counts(
    actions: np.ndarray,
    state: object,
    *,
    player_index: int,
    kick_range: float = ACTUAL_KICK_RANGE,
) -> tuple[int, int]:
    """Return `(kick requests, requests outside actual pre-action kick range)`."""
    action_array = np.asarray(actions)
    requested = action_array[..., 2] > 0
    rows = np.arange(len(np.asarray(state.ball_pos)))
    indices = np.broadcast_to(np.asarray(player_index, dtype=np.int64), rows.shape)
    distance = np.linalg.norm(
        np.asarray(state.ball_pos) - np.asarray(state.player_pos)[rows, indices], axis=-1)
    invalid = requested & (distance >= kick_range)
    return int(requested.sum()), int(invalid.sum())


def kick_request_masks(
    actions: np.ndarray,
    state: object,
    *,
    player_index: int,
    previous_invalid: np.ndarray | None = None,
    kick_range: float = ACTUAL_KICK_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify requests as valid, invalid, or held-invalid in pre-action state."""
    action_array = np.asarray(actions)
    requested = action_array[..., 2] > 0
    rows = np.arange(len(np.asarray(state.ball_pos)))
    indices = np.broadcast_to(np.asarray(player_index, dtype=np.int64), rows.shape)
    distance = np.linalg.norm(
        np.asarray(state.ball_pos) - np.asarray(state.player_pos)[rows, indices], axis=-1)
    valid = requested & (distance < kick_range)
    invalid = requested & ~valid
    held_invalid = invalid & (
        np.zeros_like(invalid) if previous_invalid is None else np.asarray(previous_invalid)
    )
    return requested, valid, invalid, held_invalid

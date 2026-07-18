from __future__ import annotations

import numpy as np

from policy_contract.chase_contract import ACTUAL_KICK_RANGE


class GoalAttributionTracker:
    """Attribute goals to the last valid kick without treating opponent own goals as earned."""

    def __init__(self, n_envs: int) -> None:
        self.last_kicker_side = np.full(n_envs, -1, dtype=np.int8)

    def reset(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            self.last_kicker_side.fill(-1)
        else:
            self.last_kicker_side[np.asarray(mask, dtype=bool)] = -1

    def record_actions(self, state: object, actions: np.ndarray) -> np.ndarray:
        action_array = np.asarray(actions)
        distance = np.linalg.norm(
            np.asarray(state.ball_pos)[:, None, :] - np.asarray(state.player_pos), axis=-1)
        valid = (action_array[..., 2] > 0) & (distance < ACTUAL_KICK_RANGE)
        any_valid = valid.any(axis=1)
        if any_valid.any():
            nearest = np.argmin(np.where(valid, distance, np.inf), axis=1)
            self.last_kicker_side[any_valid] = nearest[any_valid].astype(np.int8)
        return valid

    def record_masks(
        self,
        learner_valid: np.ndarray,
        opponent_valid: np.ndarray,
        learner_sides: np.ndarray,
    ) -> None:
        opponent_sides = 1 - learner_sides
        self.last_kicker_side[np.asarray(learner_valid, dtype=bool)] = learner_sides[learner_valid]
        self.last_kicker_side[np.asarray(opponent_valid, dtype=bool)] = opponent_sides[opponent_valid]

    def attribute(
        self, conceding_team: np.ndarray, learner_sides: np.ndarray
    ) -> dict[str, np.ndarray]:
        conceding = np.asarray(conceding_team)
        learner = np.asarray(learner_sides, dtype=np.int64)
        opponent = 1 - learner
        learner_team = np.where(learner == 0, 2, 4)
        opponent_team = np.where(learner == 0, 4, 2)
        has_goal = conceding != -1
        conceding_side = np.where(conceding == 2, 0, 1)
        own_goal = has_goal & (self.last_kicker_side == conceding_side)
        agent_own = own_goal & (conceding_side == learner)
        opponent_own = own_goal & (conceding_side == opponent)
        agent_normal = has_goal & (conceding == opponent_team) & ~opponent_own
        opponent_normal = has_goal & (conceding == learner_team) & ~agent_own
        return {
            "agent_normal_goals": agent_normal,
            "opponent_normal_goals": opponent_normal,
            "agent_own_goals": agent_own,
            "opponent_own_goals": opponent_own,
            "agent_wins": has_goal & (conceding == opponent_team),
            "opponent_wins": has_goal & (conceding == learner_team),
        }

from __future__ import annotations

import numpy as np

from policy_contract.chase_contract import ACTUAL_KICK_RANGE
from sim_training.goal_attribution import GoalAttributionTracker
from sim_training.tracking_action import TrackingDiscreteAction

DEFAULT_INVALID_KICK_PENALTY = -0.004
DEFAULT_REPEATED_INVALID_KICK_PENALTY = -0.002


def reward_configuration(
    invalid_kick_penalty: float = DEFAULT_INVALID_KICK_PENALTY,
    repeated_invalid_kick_penalty: float = DEFAULT_REPEATED_INVALID_KICK_PENALTY,
) -> dict[str, float | str]:
    if invalid_kick_penalty >= 0 or repeated_invalid_kick_penalty > 0:
        raise ValueError("kick penalties must be negative (or zero for repeated)")
    return {
        "version": "own-goal-and-kick-discipline-v3",
        "normal_goal": 10.0,
        "normal_concede": -10.0,
        "own_goal": -20.0,
        "velocity_ball_to_goal": 0.02,
        "velocity_player_to_ball": 0.005,
        "invalid_kick": float(invalid_kick_penalty),
        "repeated_held_invalid_kick": float(repeated_invalid_kick_penalty),
        "actual_kick_range": ACTUAL_KICK_RANGE,
    }


REWARD_CONFIGURATION = reward_configuration()


class AttributedGoalReward:
    """Normal goals are zero-sum; own goals only punish the responsible team."""

    def __init__(self, action_parser: TrackingDiscreteAction, configuration: dict[str, object]) -> None:
        self.action_parser = action_parser
        self.configuration = configuration
        self.tracker: GoalAttributionTracker | None = None

    def reset(self, state) -> None:
        self.tracker = GoalAttributionTracker(state.n_envs)

    def get_rewards(self, state, prev, terminated, truncated):
        if self.tracker is None:
            self.reset(state)
        assert self.tracker is not None
        actions = self.action_parser.requested_actions
        if actions is not None:
            self.tracker.record_actions(state if prev is None else prev, actions)
        rewards = np.zeros((state.n_envs, state.n_players), dtype=np.float32)
        goal_rows = np.flatnonzero(state.scored != -1)
        for row in goal_rows:
            conceding_team = state.scored[row]
            conceding_side = int(np.flatnonzero(state.team[row] == conceding_team)[0])
            own_goal = self.tracker.last_kicker_side[row] == conceding_side
            if own_goal:
                rewards[row, state.team[row] == conceding_team] = float(
                    self.configuration["own_goal"])
            else:
                rewards[row, state.team[row] == conceding_team] = float(
                    self.configuration["normal_concede"])
                rewards[row, state.team[row] != conceding_team] = float(
                    self.configuration["normal_goal"])
        done = np.asarray(terminated) | np.asarray(truncated)
        self.tracker.reset(done)
        return rewards


class InvalidKickPenalty:
    """Direct configurable penalty for raw out-of-range and repeated kick requests."""

    def __init__(
        self,
        action_parser: TrackingDiscreteAction,
        *,
        invalid_penalty: float = DEFAULT_INVALID_KICK_PENALTY,
        repeated_penalty: float = DEFAULT_REPEATED_INVALID_KICK_PENALTY,
    ) -> None:
        self.action_parser = action_parser
        self.invalid_penalty = float(invalid_penalty)
        self.repeated_penalty = float(repeated_penalty)
        self.previous_invalid: np.ndarray | None = None
        self.valid_kick: np.ndarray | None = None
        self.invalid_kick: np.ndarray | None = None
        self.repeated_held_invalid_kick: np.ndarray | None = None

    def reset(self, state) -> None:
        shape = (state.n_envs, state.n_players)
        self.previous_invalid = np.zeros(shape, dtype=bool)
        self.valid_kick = np.zeros(shape, dtype=bool)
        self.invalid_kick = np.zeros(shape, dtype=bool)
        self.repeated_held_invalid_kick = np.zeros(shape, dtype=bool)

    def get_rewards(self, state, prev, terminated, truncated):
        actions = self.action_parser.requested_actions
        if actions is None:
            return np.zeros((state.n_envs, state.n_players), dtype=np.float32)
        before = state if prev is None else prev
        distance = np.linalg.norm(
            before.ball_pos[:, None, :] - before.player_pos, axis=-1)
        requested = actions[..., 2] > 0
        valid = requested & (distance < ACTUAL_KICK_RANGE)
        invalid = requested & ~valid
        previous = self.previous_invalid
        if previous is None or previous.shape != invalid.shape:
            previous = np.zeros_like(invalid)
        repeated = invalid & previous
        penalty = (
            invalid.astype(np.float32) * self.invalid_penalty
            + repeated.astype(np.float32) * self.repeated_penalty
        )
        self.valid_kick = valid
        self.invalid_kick = invalid
        self.repeated_held_invalid_kick = repeated
        self.previous_invalid = invalid.copy()
        done = np.asarray(terminated) | np.asarray(truncated)
        self.previous_invalid[done] = False
        return penalty


def build_reward(
    action_parser: TrackingDiscreteAction,
    *,
    invalid_kick_penalty: float = DEFAULT_INVALID_KICK_PENALTY,
    repeated_invalid_kick_penalty: float = DEFAULT_REPEATED_INVALID_KICK_PENALTY,
):
    from haxballgym.reward import CombinedReward, VelocityBallToGoal, VelocityPlayerToBall

    configuration = reward_configuration(
        invalid_kick_penalty, repeated_invalid_kick_penalty)
    return CombinedReward(
        (AttributedGoalReward(action_parser, configuration), 1.0),
        (VelocityBallToGoal(), float(configuration["velocity_ball_to_goal"])),
        (VelocityPlayerToBall(), float(configuration["velocity_player_to_ball"])),
        (InvalidKickPenalty(
            action_parser, invalid_penalty=invalid_kick_penalty,
            repeated_penalty=repeated_invalid_kick_penalty), 1.0),
    )

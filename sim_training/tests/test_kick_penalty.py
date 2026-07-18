from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from policy_contract.chase_contract import ACTUAL_KICK_RANGE, kick_request_counts
from sim_training.reward_config import InvalidKickPenalty, REWARD_CONFIGURATION
from sim_training.tracking_action import TrackingDiscreteAction


def fake_state(distances: list[float]):
    count = len(distances)
    return SimpleNamespace(
        n_envs=count,
        n_players=2,
        ball_pos=np.zeros((count, 2)),
        player_pos=np.asarray([
            [[-distance, 0.0], [100.0, 0.0]] for distance in distances
        ]),
    )


def test_invalid_kick_penalty_uses_exact_pre_action_range_and_preserves_valid_kicks() -> None:
    parser = TrackingDiscreteAction(kick_values=2)
    actions = np.ones((3, 2, 3), dtype=np.int64)
    actions[..., 2] = 0
    actions[:, 0, 2] = 1
    parser.parse_actions(actions)
    before = fake_state([ACTUAL_KICK_RANGE - 0.01, ACTUAL_KICK_RANGE, 200.0])
    after = fake_state([200.0, 1.0, 1.0])
    reward = InvalidKickPenalty(parser).get_rewards(
        after, before, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))
    np.testing.assert_allclose(reward[:, 0], [0.0, -0.004, -0.004])
    assert reward[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_kick_metrics_track_total_and_invalid_fractions_independently() -> None:
    actions = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 0]])
    requested, invalid = kick_request_counts(
        actions, fake_state([10.0, 100.0, 100.0]), player_index=0)
    assert (requested, invalid) == (2, 1)


def test_goal_reward_remains_dominant_over_invalid_kick_penalty() -> None:
    assert REWARD_CONFIGURATION["invalid_kick"] == -0.004
    assert REWARD_CONFIGURATION["repeated_held_invalid_kick"] == -0.002
    assert REWARD_CONFIGURATION["normal_goal"] >= 1_000 * (
        abs(REWARD_CONFIGURATION["invalid_kick"])
        + abs(REWARD_CONFIGURATION["repeated_held_invalid_kick"]))


def test_repeated_held_kick_is_distinguished_without_penalizing_valid_kick() -> None:
    parser = TrackingDiscreteAction(kick_values=2)
    actions = np.ones((2, 2, 3), dtype=np.int64)
    actions[..., 2] = 0
    actions[:, 0, 2] = 1
    parser.parse_actions(actions)
    penalty = InvalidKickPenalty(parser)
    state = fake_state([100.0, ACTUAL_KICK_RANGE - 0.01])
    penalty.reset(state)
    first = penalty.get_rewards(state, state, np.zeros(2, bool), np.zeros(2, bool))
    second = penalty.get_rewards(state, state, np.zeros(2, bool), np.zeros(2, bool))
    np.testing.assert_allclose(first[:, 0], [-0.004, 0.0])
    np.testing.assert_allclose(second[:, 0], [-0.006, 0.0])
    assert penalty.invalid_kick[:, 0].tolist() == [True, False]
    assert penalty.repeated_held_invalid_kick[:, 0].tolist() == [True, False]
    assert penalty.valid_kick[:, 0].tolist() == [False, True]

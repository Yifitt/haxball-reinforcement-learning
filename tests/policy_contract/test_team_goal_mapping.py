from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from policy_contract.action_contract import policy_bins_to_canonical
from policy_contract.chase_contract import goal_directed_chase
from policy_contract.observation_contract import build_sim_observation
from sim_training.policy import chase_bins


def mirrored_states() -> SimpleNamespace:
    return SimpleNamespace(
        player_pos=np.asarray([
            [[-100.0, 20.0], [80.0, -10.0]],
            [[100.0, 20.0], [-80.0, -10.0]],
        ]),
        player_vel=np.asarray([
            [[1.0, 0.5], [-2.0, -0.25]],
            [[-1.0, 0.5], [2.0, -0.25]],
        ]),
        ball_pos=np.asarray([[30.0, 5.0], [-30.0, 5.0]]),
        ball_vel=np.asarray([[3.0, -1.0], [-3.0, -1.0]]),
        team=np.asarray([[2, 4], [4, 2]]),
    )


def physical_x(bins: np.ndarray, upstream_team: int) -> int:
    return int(bins[0] - 1) * (1 if upstream_team == 2 else -1)


def test_red_and_blue_select_their_opponent_goal() -> None:
    decision = goal_directed_chase(
        np.asarray([[-100.0, 0.0], [100.0, 0.0]]),
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        np.asarray([1, 2]),
    )
    assert decision.opponent_goal_x.tolist() == [370.0, -370.0]
    assert decision.dx.tolist() == [1, -1]


def test_mirrored_red_blue_states_have_equal_policy_observations_and_actions() -> None:
    state = mirrored_states()
    observations = build_sim_observation(state)
    np.testing.assert_allclose(observations[0, 0], observations[1, 0], atol=1e-7)

    bins = chase_bins(state, 0)
    np.testing.assert_array_equal(bins[0], bins[1])
    assert physical_x(bins[0], 2) == -physical_x(bins[1], 4)

    canonical = np.asarray((2, 1, 0))
    red_action = policy_bins_to_canonical(canonical, team=1)
    blue_action = policy_bins_to_canonical(canonical, team=2)
    assert (red_action, blue_action) == (4, 3)


def test_wrong_side_chase_does_not_kick_or_drive_through_ball_toward_own_goal() -> None:
    players = np.asarray([[20.0, 0.0], [-20.0, 0.0]])
    balls = np.zeros((2, 2))
    decision = goal_directed_chase(players, balls, np.asarray([1, 2]))
    assert decision.kick.tolist() == [0, 0]
    assert decision.dx.tolist() == [0, 0]
    assert decision.dy.tolist() == [-1, -1]


def test_ball_near_own_and_opponent_goals_preserves_attack_direction() -> None:
    players = np.asarray([[320.0, 0.0], [-320.0, 0.0], [-320.0, 0.0], [320.0, 0.0]])
    balls = np.asarray([[350.0, 0.0], [-350.0, 0.0], [-350.0, 0.0], [350.0, 0.0]])
    teams = np.asarray([1, 2, 1, 2])
    decision = goal_directed_chase(players, balls, teams)
    assert decision.dx[:2].tolist() == [1, -1]
    assert decision.kick[:2].tolist() == [0, 0]
    assert decision.dx[2:].tolist() == [0, 0]
    assert decision.kick[2:].tolist() == [0, 0]

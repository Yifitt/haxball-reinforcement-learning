from __future__ import annotations

import numpy as np

from haxballgym.action import DiscreteAction
from haxballgym.obs import DefaultObs

from policy_contract.action_contract import canonical_to_sim_bins
from policy_contract.observation_contract import build_sim_observation
from sim_training.env_factory import make_training_env
from sim_training.reward_config import REWARD_CONFIGURATION


def test_contract_observation_matches_upstream_default_obs() -> None:
    env = make_training_env(4)
    env.reset()
    state = env.prev_state
    expected = DefaultObs().build_obs(state)
    actual = build_sim_observation(state)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_all_canonical_actions_parse_to_valid_native_engine_inputs() -> None:
    parser = DiscreteAction(kick_values=2)
    bins = np.stack([canonical_to_sim_bins(action) for action in range(18)])
    native = parser.parse_actions(bins)
    assert native.shape == (18, 3)
    assert np.isin(native[:, :2], (-1, 0, 1)).all()
    assert np.isin(native[:, 2], (0, 1)).all()
    assert len({tuple(row) for row in native.tolist()}) == 18


def test_goal_weight_dominates_dense_reward_weights() -> None:
    non_goal_reward_scale = sum(abs(REWARD_CONFIGURATION[name]) for name in (
        "velocity_ball_to_goal", "velocity_player_to_ball", "invalid_kick"))
    assert REWARD_CONFIGURATION["normal_goal"] > 100 * non_goal_reward_scale


def test_stationary_kick_free_step_is_not_positive() -> None:
    # This reward invariant needs the legacy zero-velocity kickoff; randomized
    # starts can legitimately have positive initial ball/player velocity shaping.
    env = make_training_env(1, randomized_starts=False)
    env.reset()
    stationary = np.ones((1, 2, 3), dtype=np.int64)
    stationary[..., 2] = 0
    _, reward, _, _ = env.step(stationary)
    assert reward[0, 0] <= 0.0

from __future__ import annotations

import numpy as np

from sim_training.curriculum import configuration_for_stage
from sim_training.env_factory import make_training_env
from sim_training.opponent_pool import (
    BASE_OPPONENTS,
    OpponentPool,
    OpponentPoolConfiguration,
)
from policy_contract.checkpoint_contract import PortablePolicy


def test_pool_samples_every_scripted_opponent_reproducibly() -> None:
    config = configuration_for_stage(1)
    left = OpponentPool(4096, seed=17, configuration=config)
    right = OpponentPool(4096, seed=17, configuration=config)
    np.testing.assert_array_equal(left.names(), right.names())
    assert set(left.names()) == set(BASE_OPPONENTS)


def test_pool_actions_are_batched_valid_and_do_not_kick_outside_range() -> None:
    env = make_training_env(96, seed=22)
    observations = env.reset()
    pool = OpponentPool(96, seed=23, configuration=configuration_for_stage(1))
    actions = pool.actions(env.prev_state, 1, observations[:, 1])
    assert actions.shape == (96, 3)
    assert ((actions[:, :2] >= 0) & (actions[:, :2] <= 2)).all()
    assert ((actions[:, 2] >= 0) & (actions[:, 2] <= 1)).all()
    distance = np.linalg.norm(
        env.prev_state.ball_pos - env.prev_state.player_pos[:, 1], axis=-1)
    assert not ((actions[:, 2] == 1) & (distance >= 29.0)).any()


def test_episode_resets_resample_members_independently() -> None:
    pool = OpponentPool(64, seed=5, configuration=configuration_for_stage(1))
    before = pool.names()
    mask = np.arange(64) % 2 == 0
    pool.reset(mask)
    after = pool.names()
    np.testing.assert_array_equal(after[~mask], before[~mask])
    assert np.any(after[mask] != before[mask])


def test_checkpoint_opponents_use_the_same_batched_policy_contract() -> None:
    env = make_training_env(12, seed=44)
    observations = env.reset()
    pool = OpponentPool(
        12, seed=45,
        configuration=OpponentPoolConfiguration(("previous_policy",), (1.0,)),
        previous_models=(PortablePolicy(hidden=8),),
    )
    actions = pool.actions(env.prev_state, 1, observations[:, 1])
    assert actions.shape == (12, 3)
    assert ((actions[:, :2] >= 0) & (actions[:, :2] <= 2)).all()
    assert ((actions[:, 2] >= 0) & (actions[:, 2] <= 1)).all()


def test_defensive_chase_routes_around_ball_instead_of_retreating_through_it() -> None:
    env = make_training_env(1, seed=55)
    observations = env.reset()
    state = env.engine.snapshot()
    state.player_pos[0, 1] = (-100.0, 0.0)  # Blue is beyond the ball.
    state.ball_pos[0] = (-50.0, 0.0)        # Ball is outside Blue's own half.
    env.engine.set_state(
        state.ball_pos, state.ball_vel, state.player_pos, state.player_vel, state.steps)
    env.prev_state = env.engine.snapshot()
    pool = OpponentPool(
        1, seed=56,
        configuration=OpponentPoolConfiguration(("defensive_chase",), (1.0,)),
    )
    action = pool.actions(env.prev_state, 1, observations[:, 1])[0]
    physical_dx = int(action[0] - 1) * -1  # Blue policy x is mirrored.
    assert physical_dx == 0
    assert action[1] != 1
    assert action[2] == 0

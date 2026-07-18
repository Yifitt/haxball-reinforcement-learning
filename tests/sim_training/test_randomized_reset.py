from __future__ import annotations

import numpy as np

from sim_training.randomized_reset import RandomizedKickoffMutator


def test_randomized_resets_are_seeded_independent_and_physically_valid() -> None:
    first = RandomizedKickoffMutator(73)._sample(512)
    second = RandomizedKickoffMutator(73)._sample(512)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    ball_pos, ball_vel, player_pos, player_vel, closest, delay = first
    assert np.unique(player_pos[:, 0, 1]).size > 500
    assert np.unique(player_pos[:, 1, 1]).size > 500
    assert np.unique(ball_pos[:, 1]).size > 500
    assert set(np.unique(closest)) == {1, 2}
    assert set(np.unique(delay)) == {0, 1, 2}
    assert np.max(np.abs(ball_pos[:, 0])) <= 18.0
    assert np.max(np.abs(ball_pos[:, 1])) <= 65.0
    assert np.max(np.abs(player_pos[..., 0])) <= 305.0
    assert np.max(np.abs(player_pos[..., 1])) <= 68.0
    assert np.max(np.abs(player_vel)) <= 0.08
    assert np.max(np.abs(ball_vel)) <= 0.35
    distances = np.linalg.norm(player_pos - ball_pos[:, None], axis=-1)
    np.testing.assert_array_equal(np.argmin(distances, axis=1) + 1, closest)
    assert np.min(distances) > 100.0


def test_masked_reset_changes_only_finished_environments() -> None:
    from sim_training.env_factory import make_training_env

    env = make_training_env(8, seed=91)
    env.reset()
    before = env.engine.snapshot()
    generations = env.state_mutator.reset_generation.copy()
    mask = np.array([True, False, False, True, False, True, False, False])
    env.state_mutator.reset_mask(env.engine, mask)
    after = env.engine.snapshot()
    np.testing.assert_array_equal(after.ball_pos[~mask], before.ball_pos[~mask])
    assert np.any(after.ball_pos[mask] != before.ball_pos[mask])
    np.testing.assert_array_equal(
        env.state_mutator.reset_generation, generations + mask.astype(np.int64))

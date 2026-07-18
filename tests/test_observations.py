import numpy as np

from haxball_env.config import EnvConfig
from haxball_env.observations import build_observation, mirror_for_side
from haxball_env.physics import GameState


def test_mirror_for_right_side_flips_only_x() -> None:
    np.testing.assert_array_equal(mirror_for_side(np.array([2.0, -3.0]), 0), [2.0, -3.0])
    np.testing.assert_array_equal(mirror_for_side(np.array([2.0, -3.0]), 1), [-2.0, -3.0])


def test_swapped_mirrored_state_has_same_observation() -> None:
    config = EnvConfig()
    state_left = GameState(
        player_positions=np.array([[-3.0, 1.0], [4.0, -2.0]]),
        player_velocities=np.array([[0.5, -0.2], [-0.7, 0.3]]),
        ball_position=np.array([-0.5, 0.4]),
        ball_velocity=np.array([1.2, -0.8]),
        kick_cooldowns=np.array([0.0, 0.2]),
    )
    state_right = GameState(
        player_positions=np.array([[-4.0, -2.0], [3.0, 1.0]]),
        player_velocities=np.array([[0.7, 0.3], [-0.5, -0.2]]),
        ball_position=np.array([0.5, 0.4]),
        ball_velocity=np.array([-1.2, -0.8]),
        kick_cooldowns=np.array([0.2, 0.0]),
    )
    left = build_observation(state_left, 0, np.array([2, 1]), 0.6, config)
    right = build_observation(state_right, 1, np.array([1, 2]), 0.6, config)
    np.testing.assert_allclose(left, right)

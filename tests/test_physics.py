import numpy as np

from haxball_env.config import EnvConfig
from haxball_env.physics import (
    GameState,
    accelerate_player,
    apply_ball_friction,
    collide_with_walls,
    resolve_player_ball_collision,
    simulate_substep,
    try_kick,
)


def make_state() -> GameState:
    return GameState(
        player_positions=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        player_velocities=np.zeros((2, 2)),
        ball_position=np.zeros(2),
        ball_velocity=np.zeros(2),
        kick_cooldowns=np.zeros(2),
    )


def test_player_velocity_respects_maximum() -> None:
    config = EnvConfig()
    velocity = np.zeros(2)
    for _ in range(1_000):
        accelerate_player(velocity, np.array([1.0, 0.0]), config)
    assert np.linalg.norm(velocity) <= config.player_max_speed + 1e-12


def test_ball_friction_reduces_speed() -> None:
    velocity = np.array([4.0, -3.0])
    before = np.linalg.norm(velocity)
    apply_ball_friction(velocity, EnvConfig())
    assert np.linalg.norm(velocity) < before


def test_wall_collision_keeps_body_inside() -> None:
    config = EnvConfig()
    position = np.array([50.0, 50.0])
    velocity = np.array([2.0, 3.0])
    collide_with_walls(position, velocity, config.ball_radius, config)
    assert position[0] <= config.field_width / 2 - config.ball_radius
    assert position[1] <= config.field_height / 2 - config.ball_radius
    assert np.all(velocity <= 0.0)


def test_player_ball_collision_transfers_momentum() -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[0] = np.array([0.0, 0.0])
    state.player_velocities[0] = np.array([3.0, 0.0])
    state.ball_position = np.array([config.player_radius + config.ball_radius - 0.01, 0.0])
    assert resolve_player_ball_collision(state, 0, config)
    assert state.ball_velocity[0] > 0.0


def test_kick_only_activates_in_range_and_has_cooldown() -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[0] = np.array([-config.kick_range - 0.01, 0.0])
    assert not try_kick(state, 0, config)
    state.player_positions[0] = np.array([-config.kick_range + 0.01, 0.0])
    assert try_kick(state, 0, config)
    first_velocity = state.ball_velocity.copy()
    assert not try_kick(state, 0, config)
    np.testing.assert_array_equal(state.ball_velocity, first_velocity)


def test_kick_cooldown_expires() -> None:
    config = EnvConfig(kick_cooldown=2 * EnvConfig().physics_timestep)
    state = make_state()
    state.player_positions[0] = np.array([-1.0, 0.0])
    assert try_kick(state, 0, config)
    simulate_substep(state, np.zeros((2, 2)), (False, False), config)
    assert not try_kick(state, 0, config)
    simulate_substep(state, np.zeros((2, 2)), (False, False), config)
    assert try_kick(state, 0, config)


def test_ball_crossing_goal_line_scores() -> None:
    config = EnvConfig()
    position = np.array([config.field_width / 2 + 0.01, 0.0])
    velocity = np.array([2.0, 0.0])
    assert collide_with_walls(
        position, velocity, config.ball_radius, config, can_score=True
    ) == 0

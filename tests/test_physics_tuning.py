import numpy as np
import pytest

from haxball_env import EnvConfig, HaxBallEnv
from haxball_env.physics import (
    GameState,
    collide_with_walls,
    resolve_player_collision,
    simulate_substep,
    try_kick,
)


def make_state() -> GameState:
    return GameState(
        player_positions=np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        player_velocities=np.zeros((2, 2), dtype=np.float64),
        ball_position=np.zeros(2, dtype=np.float64),
        ball_velocity=np.zeros(2, dtype=np.float64),
        kick_cooldowns=np.zeros(2, dtype=np.float64),
    )


def player_bounds(config: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            -config.field_width / 2.0 + config.player_radius,
            -config.field_height / 2.0 + config.player_radius,
        ]
    )
    return lower, -lower


def assert_players_are_valid(state: GameState, config: EnvConfig) -> None:
    lower, upper = player_bounds(config)
    assert np.all(state.player_positions >= lower - 1e-12)
    assert np.all(state.player_positions <= upper + 1e-12)
    assert np.all(np.isfinite(state.player_positions))
    assert np.all(np.isfinite(state.player_velocities))
    assert np.max(np.linalg.norm(state.player_velocities, axis=1)) <= (
        config.player_max_speed + 1e-12
    )


def test_player_wall_collision_has_no_rebound_and_preserves_tangent() -> None:
    config = EnvConfig()
    position = np.array([-config.field_width, 1.0])
    velocity = np.array([-6.0, 2.5])

    collide_with_walls(
        position,
        velocity,
        config.player_radius,
        config,
        restitution=config.player_wall_restitution,
    )

    assert velocity[0] == pytest.approx(0.0, abs=1e-12)
    assert velocity[1] == pytest.approx(2.5)


def test_ball_wall_restitution_remains_unchanged() -> None:
    config = EnvConfig()
    position = np.array([0.0, config.field_height])
    velocity = np.array([1.5, 6.0])

    collide_with_walls(position, velocity, config.ball_radius, config)

    assert velocity[0] == pytest.approx(1.5)
    assert velocity[1] == pytest.approx(-4.8)


def test_repeated_movement_into_player_wall_does_not_oscillate() -> None:
    config = EnvConfig()
    state = make_state()
    lower, _ = player_bounds(config)
    state.player_positions[0] = [lower[0], 0.0]
    directions = np.array([[-1.0, 0.0], [0.0, 0.0]])

    normal_velocities = []
    for _ in range(600):
        simulate_substep(state, directions, (False, False), config)
        normal_velocities.append(float(state.player_velocities[0, 0]))
        assert state.player_positions[0, 0] == pytest.approx(lower[0])

    assert max(abs(value) for value in normal_velocities) <= 1e-12
    assert_players_are_valid(state, config)


def test_overlapping_players_separate_and_remain_in_bounds() -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[:] = [[0.0, 0.0], [0.4, 0.0]]

    resolve_player_collision(state, config)

    assert np.linalg.norm(
        state.player_positions[1] - state.player_positions[0]
    ) == pytest.approx(2.0 * config.player_radius)
    assert_players_are_valid(state, config)


def test_coincident_players_resolve_deterministically() -> None:
    config = EnvConfig()
    first = make_state()
    second = make_state()
    first.player_positions[:] = 0.0
    second.player_positions[:] = 0.0

    resolve_player_collision(first, config)
    resolve_player_collision(second, config)

    np.testing.assert_array_equal(first.player_positions, second.player_positions)
    np.testing.assert_array_equal(first.player_velocities, second.player_velocities)
    assert first.player_positions[0, 0] < first.player_positions[1, 0]
    assert np.linalg.norm(
        first.player_positions[1] - first.player_positions[0]
    ) == pytest.approx(2.0 * config.player_radius)
    assert_players_are_valid(first, config)


def test_players_wedged_against_different_corner_walls_separate() -> None:
    config = EnvConfig()
    state = make_state()
    # Player 0 can only move right along the bottom wall, while player 1 can
    # only move up along the left wall. Pure normal correction is blocked.
    state.player_positions[:] = [[-8.46, -5.4], [-8.90, -5.17]]

    resolve_player_collision(state, config)

    assert np.linalg.norm(state.player_positions[1] - state.player_positions[0]) >= (
        2.0 * config.player_radius - 1e-9
    )
    assert_players_are_valid(state, config)


@pytest.mark.parametrize(
    ("corner_sign", "push_direction"),
    [((-1.0, 1.0), (-1.0, 1.0)), ((1.0, -1.0), (1.0, -1.0))],
    ids=("top-left", "bottom-right"),
)
def test_players_pushed_into_corner_separate_safely(
    corner_sign: tuple[float, float], push_direction: tuple[float, float]
) -> None:
    config = EnvConfig()
    state = make_state()
    _, upper = player_bounds(config)
    corner = np.asarray(corner_sign) * upper
    state.player_positions[:] = corner
    direction = np.asarray(push_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    directions = np.repeat(direction[None, :], 2, axis=0)
    minimum = 2.0 * config.player_radius

    for _ in range(600):
        simulate_substep(state, directions, (False, False), config)
        assert np.linalg.norm(state.player_positions[1] - state.player_positions[0]) >= (
            minimum - 1e-9
        )
        assert_players_are_valid(state, config)


def test_default_ball_is_smaller() -> None:
    assert EnvConfig().ball_radius == pytest.approx(0.30)


def test_smaller_ball_remains_kickable() -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[0] = [-(config.player_radius + config.ball_radius), 0.0]

    assert try_kick(state, 0, config)
    assert state.ball_velocity[0] > 0.0
    assert np.all(np.isfinite(state.ball_velocity))


@pytest.mark.parametrize(("side", "direction"), [(0, 1.0), (1, -1.0)])
def test_smaller_ball_can_score_in_both_goals(side: int, direction: float) -> None:
    config = EnvConfig()
    state = make_state()
    state.ball_position[:] = [direction * (config.field_width / 2.0 - 0.01), 0.0]
    state.ball_velocity[:] = [direction * 2.0, 0.0]

    result = simulate_substep(state, np.zeros((2, 2)), (False, False), config)

    assert result.goal == side


def test_default_reset_positions_do_not_overlap() -> None:
    env = HaxBallEnv()
    try:
        env.reset(seed=0)
        player_distance = np.linalg.norm(
            env.state.player_positions[1] - env.state.player_positions[0]
        )
        assert player_distance >= 2.0 * env.config.player_radius
        for position in env.state.player_positions:
            assert np.linalg.norm(position - env.state.ball_position) >= (
                env.config.player_radius + env.config.ball_radius
            )
    finally:
        env.close()

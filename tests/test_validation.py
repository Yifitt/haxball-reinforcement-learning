import numpy as np
import pytest

from haxball_env import EnvConfig, HaxBallEnv
from haxball_env.actions import decode_action
from haxball_env.observations import build_observation
from haxball_env.physics import (
    GameState,
    apply_ball_friction,
    collide_with_walls,
    resolve_player_ball_collision,
    simulate_substep,
    try_kick,
)
from haxball_env.scripted import ScriptedOpponent


def make_state() -> GameState:
    return GameState(
        player_positions=np.array([[-3.0, 0.0], [3.0, 0.0]]),
        player_velocities=np.zeros((2, 2)),
        ball_position=np.zeros(2),
        ball_velocity=np.zeros(2),
        kick_cooldowns=np.zeros(2),
    )


def assert_valid_reset(env: HaxBallEnv) -> None:
    config = env.config
    assert np.all(np.abs(env.state.player_positions[:, 0]) <= config.field_width / 2 - config.player_radius)
    assert np.all(np.abs(env.state.player_positions[:, 1]) <= config.field_height / 2 - config.player_radius)
    assert np.all(np.abs(env.state.ball_position) <= [config.field_width / 2, config.field_height / 2])
    assert np.linalg.norm(env.state.player_positions[0] - env.state.player_positions[1]) >= 2 * config.player_radius
    for position in env.state.player_positions:
        assert np.linalg.norm(position - env.state.ball_position) >= config.player_radius + config.ball_radius


def test_stationary_ball_remains_stationary() -> None:
    state = make_state()
    for _ in range(20):
        result = simulate_substep(state, np.zeros((2, 2)), (False, False), EnvConfig())
        assert result.goal is None
    np.testing.assert_array_equal(state.ball_position, [0.0, 0.0])
    np.testing.assert_array_equal(state.ball_velocity, [0.0, 0.0])


def test_tiny_ball_velocity_stays_finite_and_does_not_grow() -> None:
    velocity = np.array([1e-8, -1e-8])
    previous_speed = np.linalg.norm(velocity)
    for _ in range(20):
        apply_ball_friction(velocity, EnvConfig())
        speed = np.linalg.norm(velocity)
        assert np.all(np.isfinite(velocity))
        assert speed <= previous_speed
        previous_speed = speed


@pytest.mark.parametrize("start_y, velocity_y, expected_sign", [(6.0, 4.0, -1), (-6.0, -4.0, 1)])
def test_ball_bounces_from_horizontal_walls(
    start_y: float, velocity_y: float, expected_sign: int
) -> None:
    config = EnvConfig()
    position = np.array([0.0, start_y])
    velocity = np.array([1.0, velocity_y])
    assert collide_with_walls(position, velocity, config.ball_radius, config, can_score=True) is None
    assert abs(position[1]) <= config.field_height / 2 - config.ball_radius
    assert np.sign(velocity[1]) == expected_sign
    assert np.all(np.isfinite(velocity))
    assert np.linalg.norm(velocity) <= np.linalg.norm([1.0, velocity_y])


def test_end_wall_outside_goal_mouth_does_not_score() -> None:
    config = EnvConfig()
    position = np.array([config.field_width / 2 + 1.0, config.goal_size / 2 + 0.5])
    velocity = np.array([5.0, 0.0])
    assert collide_with_walls(position, velocity, config.ball_radius, config, can_score=True) is None
    assert position[0] == pytest.approx(config.field_width / 2 - config.ball_radius)
    assert velocity[0] < 0.0


@pytest.mark.parametrize("side, x", [(0, 10.01), (1, -10.01)])
def test_goal_crossing_scores_correct_side_once_and_resets(side: int, x: float) -> None:
    config = EnvConfig(score_limit=2)
    env = HaxBallEnv(config)
    env.reset(seed=9)
    env.state.ball_position[:] = [x, 0.0]
    _, _, terminated, truncated, info = env.step(0)
    assert info["goal"] == side
    assert info["scores"][side] == 1
    assert not terminated and not truncated
    assert_valid_reset(env)
    _, _, _, _, next_info = env.step(0)
    assert next_info["goal"] is None
    assert next_info["scores"][side] == 1


def test_repeated_player_movement_against_wall_is_stable() -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[0] = [-config.field_width / 2 + config.player_radius, 4.0]
    directions = np.array([[-1.0, 1.0], [0.0, 0.0]])
    directions[0] /= np.linalg.norm(directions[0])
    for _ in range(2_000):
        simulate_substep(state, directions, (False, False), config)
    lower = [-config.field_width / 2 + config.player_radius, -config.field_height / 2 + config.player_radius]
    upper = [config.field_width / 2 - config.player_radius, config.field_height / 2 - config.player_radius]
    assert np.all(state.player_positions[0] >= lower)
    assert np.all(state.player_positions[0] <= upper)
    assert np.all(np.isfinite(state.player_positions[0]))
    assert np.linalg.norm(state.player_velocities[0]) <= config.player_max_speed + 1e-12


@pytest.mark.parametrize("ball_offset", [(0.0, 0.0), (1e-14, -1e-14), (0.2, 0.1)])
def test_deep_player_ball_overlap_resolves_safely(ball_offset: tuple[float, float]) -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[0] = [0.0, 0.0]
    state.player_velocities[0] = [config.player_max_speed, 0.0]
    state.ball_position[:] = ball_offset
    assert resolve_player_ball_collision(state, 0, config)
    assert np.all(np.isfinite(state.ball_position))
    assert np.all(np.isfinite(state.ball_velocity))
    assert np.linalg.norm(state.ball_velocity) <= config.ball_max_speed + 1e-12
    assert np.linalg.norm(state.ball_position - state.player_positions[0]) >= (
        config.player_radius + config.ball_radius
    )


def test_kick_direction_range_and_near_overlap_stability() -> None:
    config = EnvConfig()
    state = make_state()
    state.player_positions[0] = [-1.0, -0.5]
    state.ball_position[:] = [0.0, 0.0]
    delta = state.ball_position - state.player_positions[0]
    assert try_kick(state, 0, config)
    assert np.dot(state.ball_velocity, delta) > 0.0
    assert not try_kick(state, 0, config)

    overlap = make_state()
    overlap.player_positions[0] = overlap.ball_position
    assert try_kick(overlap, 0, config)
    assert np.all(np.isfinite(overlap.ball_velocity))
    assert 0.0 < np.linalg.norm(overlap.ball_velocity) <= config.ball_max_speed


def test_reset_is_valid_reproducible_and_seed_sensitive() -> None:
    env = HaxBallEnv(EnvConfig(spawn_jitter=0.5))
    first, _ = env.reset(seed=10)
    first_positions = env.state.player_positions.copy()
    assert_valid_reset(env)
    repeated, _ = env.reset(seed=10)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first_positions, env.state.player_positions)
    env.reset(seed=11)
    assert not np.array_equal(first_positions, env.state.player_positions)
    assert_valid_reset(env)


def test_finished_episode_rejects_steps_and_reset_starts_fresh() -> None:
    config = EnvConfig(score_limit=1)
    env = HaxBallEnv(config)
    env.reset(seed=1)
    env.state.ball_position[:] = [config.field_width / 2 + 0.01, 0.0]
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    finished_state = env.state.copy()
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(0)
    np.testing.assert_array_equal(env.state.player_positions, finished_state.player_positions)
    observation, info = env.reset(seed=1)
    assert info["scores"].tolist() == [0, 0]
    assert observation[-1] == pytest.approx(1.0)
    env.step(0)


def test_reward_components_do_not_leak_between_steps() -> None:
    config = EnvConfig(score_limit=2)
    env = HaxBallEnv(config)
    env.reset(seed=3)
    env.state.ball_position[:] = [config.field_width / 2 + 0.01, 0.0]
    _, _, _, _, goal_info = env.step(0)
    assert goal_info["reward_components"]["goal"] == config.rewards.goal
    _, reward, _, _, next_info = env.step(0)
    assert next_info["reward_components"]["goal"] == 0.0
    assert next_info["reward_components"]["concede"] == 0.0
    assert reward == pytest.approx(sum(next_info["reward_components"].values()))


def test_extreme_valid_observation_is_finite_and_contained() -> None:
    config = EnvConfig()
    env = HaxBallEnv(config, controlled_side=1)
    env.reset(seed=0)
    env.state.player_positions[:] = [
        [-config.field_width / 2 + config.player_radius, config.field_height / 2 - config.player_radius],
        [config.field_width / 2 - config.player_radius, -config.field_height / 2 + config.player_radius],
    ]
    player_component = config.player_max_speed / np.sqrt(2.0)
    ball_component = config.ball_max_speed / np.sqrt(2.0)
    env.state.player_velocities[:] = [
        [-player_component, player_component],
        [player_component, -player_component],
    ]
    env.state.ball_position[:] = [config.field_width / 2, config.goal_size / 2]
    env.state.ball_velocity[:] = [ball_component, -ball_component]
    observation = env._observation()
    assert env.observation_space.contains(observation)
    assert np.all(np.isfinite(observation))


def test_score_goal_vectors_and_time_use_controlled_perspective() -> None:
    config = EnvConfig(score_limit=4, episode_time_limit=10.0)
    state = make_state()
    state.player_positions[:] = [[-4.0, 0.0], [4.0, 0.0]]
    state.ball_position[:] = [2.0, 0.0]
    scores = np.array([3, 1])
    left = build_observation(state, 0, scores, 0.25, config)
    right = build_observation(state, 1, scores, 0.25, config)
    assert left[14] == pytest.approx((10.0 - 2.0) / 20.0)
    assert right[14] == pytest.approx((10.0 - -2.0) / 20.0)
    assert left[16] == pytest.approx(-0.3)
    assert right[16] == pytest.approx(-0.3)
    assert left[19] == pytest.approx(0.5)
    assert right[19] == pytest.approx(-0.5)
    assert left[20] == pytest.approx(0.25)
    assert right[20] == pytest.approx(0.25)


def test_environment_remaining_time_fraction_matches_elapsed_time() -> None:
    config = EnvConfig(episode_time_limit=10.0)
    env = HaxBallEnv(config)
    env.reset(seed=0)
    observation, _, _, _, _ = env.step(0)
    expected = 1.0 - config.action_repeat * config.physics_timestep / config.episode_time_limit
    assert observation[20] == pytest.approx(expected)


def test_scripted_policy_is_deterministic_and_approaches_ball() -> None:
    config = EnvConfig()
    policy = ScriptedOpponent(config)
    state = make_state()
    state.player_positions[0] = [-4.0, 0.0]
    state.ball_position[:] = [1.0, 1.0]
    first = policy.act(state, 0)
    assert first == policy.act(state.copy(), 0)
    direction, kick = decode_action(first)
    assert direction[0] > 0.0 and direction[1] > 0.0
    assert not kick


def test_scripted_policy_moves_to_defensive_position_and_kicks_only_close() -> None:
    config = EnvConfig()
    policy = ScriptedOpponent(config)
    state = make_state()
    state.player_positions[0] = [-3.0, 0.0]
    state.ball_position[:] = [-6.0, 1.0]
    direction, kick = decode_action(policy.act(state, 0))
    assert direction[0] < 0.0
    assert not kick
    state.player_positions[0] = state.ball_position + [config.kick_range - 0.01, 0.0]
    _, kick = decode_action(policy.act(state, 0))
    assert kick


def test_random_opponent_is_seed_deterministic() -> None:
    first = HaxBallEnv(opponent_mode="random")
    second = HaxBallEnv(opponent_mode="random")
    first.reset(seed=22)
    second.reset(seed=22)
    for _ in range(20):
        obs_a, reward_a, done_a, truncated_a, _ = first.step(0)
        obs_b, reward_b, done_b, truncated_b, _ = second.step(0)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert (reward_a, done_a, truncated_a) == (reward_b, done_b, truncated_b)

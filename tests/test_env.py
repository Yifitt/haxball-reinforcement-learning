import numpy as np

from haxball_env import EnvConfig, HaxBallEnv
from haxball_env.actions import NUM_ACTIONS
from haxball_env.scripted import ScriptedOpponent


def test_same_seed_produces_same_initial_state() -> None:
    first = HaxBallEnv()
    second = HaxBallEnv()
    obs_a, _ = first.reset(seed=123)
    obs_b, _ = second.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(first.state.player_positions, second.state.player_positions)


def test_observation_is_inside_declared_space() -> None:
    env = HaxBallEnv(controlled_side=1)
    observation, _ = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    observation, _, _, _, _ = env.step(8)
    assert env.observation_space.contains(observation)


def test_goal_updates_score_and_resets_positions() -> None:
    config = EnvConfig(score_limit=2)
    env = HaxBallEnv(config)
    env.reset(seed=1)
    env.state.ball_position[:] = [config.field_width / 2 + 0.01, 0.0]
    observation, _, terminated, truncated, info = env.step(0)
    assert info["goal"] == 0
    assert info["scores"].tolist() == [1, 0]
    assert not terminated and not truncated
    np.testing.assert_array_equal(env.state.ball_position, [0.0, 0.0])
    assert np.all(np.isfinite(observation))


def test_score_limit_terminates_episode() -> None:
    config = EnvConfig(score_limit=1)
    env = HaxBallEnv(config)
    env.reset(seed=1)
    env.state.ball_position[:] = [config.field_width / 2 + 0.01, 0.0]
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated


def test_time_limit_truncates_episode() -> None:
    base = EnvConfig()
    config = EnvConfig(episode_time_limit=base.physics_timestep)
    env = HaxBallEnv(config)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and truncated


def test_reward_components_are_reported() -> None:
    env = HaxBallEnv()
    env.reset(seed=2)
    _, reward, _, _, info = env.step(4)
    assert set(info["reward_components"]) == {
        "goal", "concede", "ball_progress", "touch", "inactivity"
    }
    assert np.isclose(reward, sum(info["reward_components"].values()))


def test_scripted_opponent_always_returns_valid_action() -> None:
    env = HaxBallEnv()
    opponent = ScriptedOpponent(env.config)
    env.reset(seed=3)
    for side in (0, 1):
        action = opponent.act(env.state, side)
        assert 0 <= action < NUM_ACTIONS


def test_short_random_episode_has_no_nan_or_exceptions() -> None:
    env = HaxBallEnv(EnvConfig(episode_time_limit=2.0))
    observation, _ = env.reset(seed=42)
    rng = np.random.default_rng(42)
    for _ in range(100):
        observation, reward, terminated, truncated, _ = env.step(
            int(rng.integers(NUM_ACTIONS))
        )
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        if terminated or truncated:
            observation, _ = env.reset(seed=42)

"""Gymnasium wrapper around the deterministic 1v1 simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import NUM_ACTIONS, decode_action
from .config import EnvConfig
from .observations import OBSERVATION_SIZE, build_observation
from .physics import GameState, simulate_substep
from .rewards import calculate_reward
from .scripted import ScriptedOpponent

if TYPE_CHECKING:
    from .renderer import PygameRenderer


class HaxBallEnv(gym.Env[np.ndarray, int]):
    """A compact 1v1 environment with one controlled player."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        config: EnvConfig | None = None,
        controlled_side: int = 0,
        render_mode: Literal["human"] | None = None,
        opponent_mode: Literal["scripted", "random"] = "scripted",
    ) -> None:
        super().__init__()
        if controlled_side not in (0, 1):
            raise ValueError("controlled_side must be 0 or 1")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        if opponent_mode not in ("scripted", "random"):
            raise ValueError(f"unsupported opponent_mode: {opponent_mode!r}")
        self.config = config or EnvConfig()
        self.controlled_side = controlled_side
        self.render_mode = render_mode
        self.opponent_mode = opponent_mode
        self.opponent = ScriptedOpponent(self.config)
        self.metadata = {**self.metadata, "render_fps": self.config.render_fps}
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.state = self._initial_state(jitter=False)
        self.scores = np.zeros(2, dtype=np.int32)
        self.elapsed_time = 0.0
        self._episode_done = False
        self._renderer: PygameRenderer | None = None
        self.window_closed = False

    def _initial_state(self, *, jitter: bool) -> GameState:
        positions = np.array(
            [
                [-self.config.field_width * 0.25, 0.0],
                [self.config.field_width * 0.25, 0.0],
            ],
            dtype=np.float64,
        )
        if jitter and self.config.spawn_jitter > 0.0:
            positions[:, 1] += self.np_random.uniform(
                -self.config.spawn_jitter,
                self.config.spawn_jitter,
                size=2,
            )
        return GameState(
            player_positions=positions,
            player_velocities=np.zeros((2, 2), dtype=np.float64),
            ball_position=np.zeros(2, dtype=np.float64),
            ball_velocity=np.zeros(2, dtype=np.float64),
            kick_cooldowns=np.zeros(2, dtype=np.float64),
        )

    def _reset_positions(self) -> None:
        self.state = self._initial_state(jitter=True)

    def _remaining_fraction(self) -> float:
        return float(np.clip(1.0 - self.elapsed_time / self.config.episode_time_limit, 0.0, 1.0))

    def _observation(self) -> np.ndarray:
        return build_observation(
            self.state,
            self.controlled_side,
            self.scores,
            self._remaining_fraction(),
            self.config,
        )

    def _info(
        self,
        *,
        goal: int | None = None,
        reward_components: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "scores": self.scores.copy(),
            "elapsed_time": self.elapsed_time,
            "goal": goal,
            "reward_components": reward_components
            or {"goal": 0.0, "concede": 0.0, "ball_progress": 0.0, "touch": 0.0, "inactivity": 0.0},
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.scores.fill(0)
        self.elapsed_time = 0.0
        self._episode_done = False
        self.window_closed = False
        self._reset_positions()
        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _opponent_action(self, side: int) -> int:
        if self.opponent_mode == "random":
            return int(self.np_random.integers(NUM_ACTIONS))
        return self.opponent.act(self.state, side)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        controlled_direction, controlled_kick = decode_action(action)
        opponent_side = 1 - self.controlled_side
        opponent_action = self._opponent_action(opponent_side)
        opponent_direction, opponent_kick = decode_action(opponent_action)
        directions = np.zeros((2, 2), dtype=np.float64)
        directions[self.controlled_side] = controlled_direction
        directions[opponent_side] = opponent_direction
        kick_requests = [False, False]
        kick_requests[self.controlled_side] = controlled_kick
        kick_requests[opponent_side] = opponent_kick

        previous_ball_x = float(self.state.ball_position[0])
        reward_ball_x = previous_ball_x
        goal: int | None = None
        controlled_contact = False
        for _ in range(self.config.action_repeat):
            result = simulate_substep(
                self.state,
                directions,
                (kick_requests[0], kick_requests[1]),
                self.config,
            )
            self.elapsed_time += self.config.physics_timestep
            controlled_contact |= (
                result.touches[self.controlled_side] or result.kicks[self.controlled_side]
            )
            reward_ball_x = float(self.state.ball_position[0])
            if result.goal is not None:
                goal = result.goal
                self.scores[goal] += 1
                break
            if self.elapsed_time >= self.config.episode_time_limit:
                break

        reward, components = calculate_reward(
            side=self.controlled_side,
            goal=goal,
            previous_ball_x=previous_ball_x,
            current_ball_x=reward_ball_x,
            touched_or_kicked=controlled_contact,
            config=self.config,
        )
        terminated = bool(np.any(self.scores >= self.config.score_limit))
        truncated = bool(not terminated and self.elapsed_time >= self.config.episode_time_limit)
        self._episode_done = terminated or truncated
        if goal is not None:
            self._reset_positions()

        result = self._observation(), reward, terminated, truncated, self._info(
            goal=goal,
            reward_components=components,
        )
        if self.render_mode == "human":
            self.render()
        return result

    def render(self) -> None:
        if self.render_mode is None:
            return
        if self._renderer is None:
            from .renderer import PygameRenderer

            self._renderer = PygameRenderer(self.config)
        self.window_closed = not self._renderer.render(
            state=self.state,
            scores=self.scores,
            remaining_time=self.config.episode_time_limit - self.elapsed_time,
            controlled_side=self.controlled_side,
            episode_done=self._episode_done,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

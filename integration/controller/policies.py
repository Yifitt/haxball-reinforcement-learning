from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from policy_contract.chase_contract import goal_directed_chase

from .protocol import PlayerState, StateMessage

CONTROLLED_POLICY_NAMES = ("random", "persistent-random", "chase", "checkpoint")
POLICY_NAMES = (*CONTROLLED_POLICY_NAMES, "stationary")


class Policy(Protocol):
    def select_action(self, state: StateMessage) -> int: ...
    def reset(self) -> None: ...


class StationaryPolicy:
    def select_action(self, state: StateMessage) -> int:
        return 0

    def reset(self) -> None:
        pass


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def select_action(self, state: StateMessage) -> int:
        return self._rng.randrange(18)

    def reset(self) -> None:
        self._rng.seed(self._seed)


class PersistentRandomPolicy:
    def __init__(
        self,
        seed: int,
        *,
        action_hold_steps: int = 6,
        kick_probability: float = 0.10,
    ) -> None:
        if action_hold_steps < 1:
            raise ValueError("action_hold_steps must be positive")
        if not 0.0 <= kick_probability <= 1.0:
            raise ValueError("kick_probability must be between 0 and 1")
        self._seed = seed
        self._rng = random.Random(seed)
        self.action_hold_steps = action_hold_steps
        self.kick_probability = kick_probability
        self._movement_action = 0
        self._remaining = 0

    def select_action(self, state: StateMessage) -> int:
        if self._remaining <= 0:
            self._movement_action = self._rng.randrange(9)
            self._remaining = self.action_hold_steps
        self._remaining -= 1
        kick = self._rng.random() < self.kick_probability
        return self._movement_action + (9 if kick else 0)

    def reset(self) -> None:
        self._rng.seed(self._seed)
        self._movement_action = 0
        self._remaining = 0


def _movement_action(dx: float, dy: float, dead_zone: float) -> int:
    horizontal = 0 if abs(dx) <= dead_zone else (1 if dx > 0 else -1)
    vertical = 0 if abs(dy) <= dead_zone else (1 if dy > 0 else -1)
    return {
        (0, 0): 0,
        (0, -1): 1,
        (0, 1): 2,
        (-1, 0): 3,
        (1, 0): 4,
        (-1, -1): 5,
        (1, -1): 6,
        (-1, 1): 7,
        (1, 1): 8,
    }[(horizontal, vertical)]


class ChasePolicy:
    def __init__(
        self,
        role: str,
        *,
        behind_ball_distance: float = 24.0,
        alignment_tolerance: float = 15.0,
        kick_distance: float = 28.0,
        dead_zone: float = 2.5,
    ) -> None:
        if role not in ("controlled", "opponent"):
            raise ValueError(f"unknown policy role: {role}")
        self.role = role
        self.behind_ball_distance = behind_ball_distance
        self.alignment_tolerance = alignment_tolerance
        self.kick_distance = kick_distance
        self.dead_zone = dead_zone

    def _player(self, state: StateMessage) -> PlayerState | None:
        return state.controlled if self.role == "controlled" else state.opponent

    def select_action(self, state: StateMessage) -> int:
        player = self._player(state)
        ball = state.ball
        if not state.game_active or player is None or ball is None:
            return 0
        decision = goal_directed_chase(
            np.asarray((player.x, player.y)),
            np.asarray((ball.x, ball.y)),
            np.asarray(player.team),
            behind_ball_distance=self.behind_ball_distance,
            alignment_tolerance=self.alignment_tolerance,
            kick_range=self.kick_distance,
            dead_zone=self.dead_zone,
        )
        movement = _movement_action(float(decision.dx), float(decision.dy), 0.0)
        kick = bool(decision.kick)
        return movement + (9 if kick else 0)

    def reset(self) -> None:
        pass


class CheckpointPolicy:
    """Portable simulator-trained policy for real browser state inference."""

    def __init__(self, role: str, checkpoint_path: str | Path) -> None:
        if role not in ("controlled", "opponent"):
            raise ValueError(f"unknown policy role: {role}")
        from policy_contract.checkpoint_contract import load_checkpoint

        self.role = role
        self.checkpoint_path = str(checkpoint_path)
        self.model, self.metadata = load_checkpoint(checkpoint_path)
        self.inference_ms: list[float] = []

    def select_action(self, state: StateMessage) -> int:
        from policy_contract.observation_contract import build_browser_observation

        player = state.controlled_player if self.role == "controlled" else state.opponent_player
        if player is None:
            raise ValueError(f"checkpoint policy missing {self.role} player")
        observation = build_browser_observation(state, role=self.role)
        started = time.perf_counter()
        action = self.model.predict_action(observation, team=player.team)
        self.inference_ms.append((time.perf_counter() - started) * 1000.0)
        if len(self.inference_ms) > 10_000:
            del self.inference_ms[: len(self.inference_ms) - 10_000]
        return action

    def median_inference_ms(self) -> float | None:
        if not self.inference_ms:
            return None
        ordered = sorted(self.inference_ms)
        return ordered[len(ordered) // 2]

    def reset(self) -> None:
        # The network is feed-forward and has no recurrent policy state.
        pass


def create_policy(
    name: str,
    role: str,
    *,
    seed: int,
    action_hold_steps: int = 6,
    kick_probability: float = 0.10,
    checkpoint_path: str | Path | None = None,
) -> Policy:
    if name == "stationary":
        return StationaryPolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "persistent-random":
        return PersistentRandomPolicy(
            seed,
            action_hold_steps=action_hold_steps,
            kick_probability=kick_probability,
        )
    if name == "chase":
        return ChasePolicy(role)
    if name == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint policy requires a checkpoint path")
        return CheckpointPolicy(role, checkpoint_path)
    raise ValueError(f"unknown policy: {name}")

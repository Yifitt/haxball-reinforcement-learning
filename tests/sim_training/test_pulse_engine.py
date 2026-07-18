from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sim_training.pulse_engine import PulseTransitionEngine


@dataclass(frozen=True)
class FakeState:
    scored: np.ndarray


class FakeEngine:
    def __init__(self) -> None:
        self.n_envs = 2
        self.n_players = 2
        self._teams = np.asarray([[2, 4], [2, 4]])
        self._e = object()
        self.actions: list[np.ndarray] = []
        self.state = FakeState(np.full(2, -1, dtype=np.int8))

    def step(self, actions: np.ndarray) -> FakeState:
        self.actions.append(actions.copy())
        return self.state


def test_kick_is_one_tick_pulse_while_movement_repeats() -> None:
    delegate = FakeEngine()
    engine = PulseTransitionEngine(delegate, action_repeat=8)
    actions = np.asarray([
        [[2, 1, 1], [0, 1, 0]],
        [[1, 2, 1], [1, 0, 1]],
    ])
    engine.step(actions)
    assert len(delegate.actions) == 8
    np.testing.assert_array_equal(delegate.actions[0], actions)
    for repeated in delegate.actions[1:]:
        np.testing.assert_array_equal(repeated[..., :2], actions[..., :2])
        assert not repeated[..., 2].any()

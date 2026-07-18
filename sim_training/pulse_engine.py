from __future__ import annotations

from dataclasses import replace

import numpy as np


class PulseTransitionEngine:
    """Repeat movement for a decision while pulsing kick on its first physics tick."""

    def __init__(self, delegate: object, *, action_repeat: int) -> None:
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self._delegate = delegate
        self.action_repeat = int(action_repeat)
        self.n_envs = delegate.n_envs
        self.n_players = delegate.n_players
        self._teams = delegate._teams
        self._e = delegate._e

    def reset(self):
        return self._delegate.reset()

    def reset_mask(self, mask: np.ndarray) -> None:
        self._delegate.reset_mask(mask)

    def snapshot(self):
        return self._delegate.snapshot()

    def set_state(self, *args, **kwargs):
        return self._delegate.set_state(*args, **kwargs)

    def set_kick_rate_limit(self, *args, **kwargs) -> None:
        self._delegate.set_kick_rate_limit(*args, **kwargs)

    def step(self, engine_actions: np.ndarray):
        pulse = np.ascontiguousarray(engine_actions, dtype=np.int64)
        movement_only = pulse.copy()
        movement_only[..., 2] = 0
        scored = np.full(self.n_envs, -1, dtype=np.int8)
        state = None
        for tick in range(self.action_repeat):
            state = self._delegate.step(pulse if tick == 0 else movement_only)
            new_goal = (scored == -1) & (state.scored != -1)
            scored[new_goal] = state.scored[new_goal]
        assert state is not None
        return replace(state, scored=scored)

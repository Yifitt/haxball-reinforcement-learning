from __future__ import annotations

import numpy as np


class TrackingDiscreteAction:
    """Discrete parser that exposes the current policy kick requests to rewards."""

    def __init__(self, kick_values: int = 2) -> None:
        from haxballgym.action import DiscreteAction

        self._delegate = DiscreteAction(kick_values=kick_values)
        self.requested_actions: np.ndarray | None = None

    def action_space(self) -> tuple[int, ...]:
        return self._delegate.action_space()

    def parse_actions(self, actions: np.ndarray) -> np.ndarray:
        self.requested_actions = np.asarray(actions, dtype=np.int64).copy()
        return self._delegate.parse_actions(actions)

    def reset(self, state) -> None:
        self.requested_actions = None

"""Mapping between the 18 discrete actions and legal movement/kick pairs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


MOVEMENT_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
)
NUM_ACTIONS = 18


def decode_action(action: int) -> tuple[NDArray[np.float64], bool]:
    """Decode ``action``; actions 0..8 move, and 9..17 add kick."""
    if isinstance(action, np.ndarray):
        if action.shape != ():
            raise ValueError("action must be a scalar")
        action = int(action.item())
    if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < NUM_ACTIONS:
        raise ValueError(f"action must be an integer in [0, {NUM_ACTIONS - 1}]")

    action = int(action)
    movement_index = action % len(MOVEMENT_DIRECTIONS)
    direction = np.asarray(MOVEMENT_DIRECTIONS[movement_index], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length > 0.0:
        direction /= length
    return direction, action >= len(MOVEMENT_DIRECTIONS)


def encode_action(movement_index: int, kick: bool = False) -> int:
    if not 0 <= movement_index < len(MOVEMENT_DIRECTIONS):
        raise ValueError("movement_index must be in [0, 8]")
    return int(movement_index + (9 if kick else 0))


def direction_to_action(direction: NDArray[np.floating], kick: bool = False) -> int:
    """Quantize a vector to the closest of the nine legal movement choices."""
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("direction must have shape (2,)")
    if float(np.linalg.norm(vector)) < 1e-9:
        return encode_action(0, kick)
    signs = (int(np.sign(vector[0])), int(np.sign(vector[1])))
    return encode_action(MOVEMENT_DIRECTIONS.index(signs), kick)

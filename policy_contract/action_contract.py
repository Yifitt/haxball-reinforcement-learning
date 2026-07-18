from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from integration.controller.actions import ACTIONS, get_action


def canonical_to_sim_bins(action: int) -> np.ndarray:
    """Canonical browser action -> HaxballGym policy-frame `(x,y,kick)` bins.

    Browser/HaxBall `up` decreases host y, while HaxballGym `up` increases sim y;
    both are represented here by the semantic `up` bin (y=2). The simulator's
    action parser handles its numeric world axis; Playwright keeps the key name.
    """
    definition = get_action(action)
    keys = definition["keys"]
    x = 0 if "left" in keys else 2 if "right" in keys else 1
    y = 2 if "up" in keys else 0 if "down" in keys else 1
    return np.asarray((x, y, int(definition["kick"])), dtype=np.int64)


_BINS_TO_CANONICAL = {
    tuple(canonical_to_sim_bins(action).tolist()): action
    for action in range(len(ACTIONS))
}


def sim_bins_to_canonical(bins: Sequence[int] | np.ndarray) -> int:
    values = tuple(int(value) for value in bins)
    if len(values) != 3 or values[0] not in range(3) or values[1] not in range(3) or values[2] not in range(2):
        raise ValueError(f"invalid HaxballGym action bins: {values!r}")
    return _BINS_TO_CANONICAL[values]


def policy_bins_to_canonical(
    bins: Sequence[int] | np.ndarray,
    *,
    team: int,
) -> int:
    """Convert normalized policy-frame bins to physical browser key semantics.

    Observations always attack +x. HaxballGym explicitly un-mirrors Blue's x
    action before physics; real-browser deployment must perform the same explicit
    conversion. Vertical input is semantic and is never team-swapped.
    """
    values = np.asarray(bins, dtype=np.int64).copy()
    if values.shape != (3,) or team not in (1, 2):
        raise ValueError("policy action requires three bins and team 1 or 2")
    if team == 2:
        values[0] = 2 - values[0]
    return sim_bins_to_canonical(values)

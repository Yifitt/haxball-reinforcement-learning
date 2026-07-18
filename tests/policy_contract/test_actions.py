from __future__ import annotations

import numpy as np

from integration.controller.actions import ACTIONS
from policy_contract.action_contract import (
    canonical_to_sim_bins,
    policy_bins_to_canonical,
    sim_bins_to_canonical,
)


def test_all_18_actions_round_trip_with_diagonals_and_kick() -> None:
    seen: set[tuple[int, int, int]] = set()
    for action in range(18):
        bins = canonical_to_sim_bins(action)
        assert bins.shape == (3,)
        assert sim_bins_to_canonical(bins) == action
        seen.add(tuple(bins.tolist()))
    assert len(seen) == 18
    assert canonical_to_sim_bins(5).tolist() == [0, 2, 0]
    assert canonical_to_sim_bins(17).tolist() == [2, 0, 1]
    assert len(ACTIONS) == 18


def test_blue_policy_frame_only_unmirrors_horizontal_input() -> None:
    bins = np.asarray((2, 2, 1))  # attack-right + up + kick in normalized frame
    red = policy_bins_to_canonical(bins, team=1)
    blue = policy_bins_to_canonical(bins, team=2)
    assert ACTIONS[red]["name"] == "up-right+kick"
    assert ACTIONS[blue]["name"] == "up-left+kick"

from __future__ import annotations

import numpy as np

from sim_training.goal_attribution import GoalAttributionTracker


def test_red_agent_own_goal_is_not_an_opponent_normal_goal() -> None:
    tracker = GoalAttributionTracker(1)
    tracker.last_kicker_side[0] = 0
    result = tracker.attribute(np.asarray([2]), np.asarray([0]))
    assert result["agent_own_goals"].tolist() == [True]
    assert result["opponent_normal_goals"].tolist() == [False]
    assert result["agent_normal_goals"].tolist() == [False]


def test_blue_agent_and_opponent_own_goals_are_attributed_symmetrically() -> None:
    tracker = GoalAttributionTracker(2)
    tracker.last_kicker_side[:] = [1, 0]
    result = tracker.attribute(np.asarray([4, 2]), np.asarray([1, 1]))
    assert result["agent_own_goals"].tolist() == [True, False]
    assert result["opponent_own_goals"].tolist() == [False, True]
    assert result["agent_normal_goals"].tolist() == [False, False]
    assert result["opponent_normal_goals"].tolist() == [False, False]

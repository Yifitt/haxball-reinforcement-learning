from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from integration.controller.protocol import BodyState, MatchState, PlayerState, StateMessage
from policy_contract.observation_contract import (
    OBSERVATION_SIZE,
    build_browser_observation,
    build_sim_observation,
)


def equivalent_states():
    sim = SimpleNamespace(
        player_pos=np.asarray([[[-100.0, 10.0], [100.0, -5.0]]]),
        player_vel=np.asarray([[[1.0, 0.5], [-1.0, -0.25]]]),
        ball_pos=np.asarray([[20.0, 3.0]]),
        ball_vel=np.asarray([[2.0, -1.0]]),
        team=np.asarray([[2, 4]]),
    )
    browser = StateMessage(
        tick_id=3,
        timestamp=1000.0,
        game_active=True,
        controlled=PlayerState(id=8, team=1, x=-100, y=-10, vx=1, vy=-0.5),
        opponent=PlayerState(id=2, team=2, x=100, y=5, vx=-1, vy=0.25),
        ball=BodyState(x=20, y=-3, vx=2, vy=1),
        match=MatchState(
            controlled_side="red", controlled_score=0, opponent_score=0,
            elapsed_time=0, last_goal_event=None,
        ),
    )
    return sim, browser


def test_simulator_and_browser_observations_are_semantically_equal() -> None:
    sim, browser = equivalent_states()
    simulator = build_sim_observation(sim)
    assert simulator.shape == (1, 2, OBSERVATION_SIZE)
    np.testing.assert_allclose(
        simulator[0, 0], build_browser_observation(browser, role="controlled"), atol=1e-7)
    np.testing.assert_allclose(
        simulator[0, 1], build_browser_observation(browser, role="opponent"), atol=1e-7)


def test_vertical_conversion_and_red_blue_mirroring_are_explicit() -> None:
    sim, browser = equivalent_states()
    observations = build_sim_observation(sim)
    assert observations[0, 0, 1] == pytest.approx(10 / 200)
    assert observations[0, 1, 0] == pytest.approx(-100 / 420)
    assert build_browser_observation(browser, role="controlled")[1] == pytest.approx(10 / 200)


def test_missing_or_nonfinite_browser_state_fails_clearly() -> None:
    _, browser = equivalent_states()
    inactive = StateMessage(
        tick_id=browser.tick_id, timestamp=browser.timestamp, game_active=False,
        controlled=browser.controlled, opponent=browser.opponent, ball=browser.ball,
        match=browser.match,
    )
    with pytest.raises(ValueError, match="active mapped"):
        build_browser_observation(inactive)
    bad = SimpleNamespace(**vars(equivalent_states()[0]))
    bad.ball_pos = np.asarray([[np.nan, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        build_sim_observation(bad)

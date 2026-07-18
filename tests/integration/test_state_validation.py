import copy

import pytest

from integration.controller.protocol import validate_state_message


def state_message() -> dict[str, object]:
    body = {"x": 1.0, "y": 2.0, "vx": 0.5, "vy": -0.25}
    return {
        "protocol_version": 1,
        "type": "state",
        "tick_id": 10,
        "timestamp": 1_700_000_000_000.0,
        "game_active": True,
        "controlled": {"id": 1, "team": 1, **body},
        "opponent": {"id": 2, "team": 2, **body},
        "ball": body,
        "match": {
            "controlled_side": "red",
            "controlled_score": 1,
            "opponent_score": 2,
            "elapsed_time": 12.5,
            "last_goal_event": {"team": "blue", "tick_id": 9},
        },
    }


def test_valid_state_schema() -> None:
    state = validate_state_message(state_message())
    assert state.tick_id == 10
    assert state.controlled is not None and state.controlled.id == 1
    assert state.controlled.team == 1
    assert state.ball is not None and state.ball.vx == 0.5


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_body_values_are_rejected(bad_value: float) -> None:
    message = state_message()
    message["ball"] = {"x": bad_value, "y": 0.0, "vx": 0.0, "vy": 0.0}
    with pytest.raises(ValueError, match="finite"):
        validate_state_message(message)


def test_missing_players_are_cleanly_represented_while_inactive() -> None:
    message = state_message()
    message["game_active"] = False
    message["controlled"] = None
    message["opponent"] = None
    state = validate_state_message(message)
    assert state.controlled is None and state.opponent is None


def test_missing_player_is_rejected_while_active() -> None:
    message = copy.deepcopy(state_message())
    message["controlled"] = None
    with pytest.raises(ValueError, match="requires both players"):
        validate_state_message(message)

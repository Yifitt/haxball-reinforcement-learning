import json
from pathlib import Path

import pytest

from integration.controller.actions import get_action
from integration.controller.policies import ChasePolicy, PersistentRandomPolicy, create_policy
from integration.controller.protocol import validate_state_message
from integration.scripts.random_agent import action_displacement_verified


def state(
    *,
    controlled_x: float = -60.0,
    opponent_x: float = 60.0,
    controlled_y: float = 0.0,
    opponent_y: float = 0.0,
    ball_x: float = 0.0,
    ball_y: float = 0.0,
):
    def player(player_id: int, team: int, x: float, y: float) -> dict[str, float | int]:
        return {"id": player_id, "team": team, "x": x, "y": y, "vx": 0.0, "vy": 0.0}

    return validate_state_message({
        "protocol_version": 1,
        "type": "state",
        "tick_id": 1,
        "timestamp": 1.0,
        "game_active": True,
        "controlled": player(9, 1, controlled_x, controlled_y),
        "opponent": player(3, 2, opponent_x, opponent_y),
        "ball": {"x": ball_x, "y": ball_y, "vx": 0.0, "vy": 0.0},
        "match": {
            "controlled_side": "red",
            "controlled_score": 0,
            "opponent_score": 0,
            "elapsed_time": 0.0,
            "last_goal_event": None,
        },
    })


def test_chase_policy_moves_red_and_blue_toward_the_ball() -> None:
    current = state()
    assert ChasePolicy("controlled").select_action(current) == 4  # right
    assert ChasePolicy("opponent").select_action(current) == 3  # left


@pytest.mark.parametrize(
    ("role", "ball_x", "ball_y", "expected_keys"),
    (
        ("controlled", 100.0, 0.0, ["right"]),
        ("controlled", -100.0, 0.0, ["up"]),
        ("opponent", 100.0, 0.0, ["up"]),
        ("opponent", -100.0, 0.0, ["left"]),
        ("controlled", 0.0, -100.0, ["up"]),
        ("opponent", 0.0, 100.0, ["down"]),
        ("controlled", 100.0, -100.0, ["up", "right"]),
        ("opponent", -100.0, 100.0, ["down", "left"]),
    ),
)
def test_chase_uses_real_physical_directions_for_red_and_blue(
    role: str,
    ball_x: float,
    ball_y: float,
    expected_keys: list[str],
) -> None:
    current = state(
        controlled_x=0.0,
        opponent_x=0.0,
        ball_x=ball_x,
        ball_y=ball_y,
    )
    definition = get_action(ChasePolicy(role).select_action(current))
    assert definition["keys"] == expected_keys
    assert definition["kick"] is False


@pytest.mark.parametrize(
    ("role", "player_x", "ball_x"),
    (
        ("controlled", -300.0, -340.0),
        ("opponent", 300.0, 340.0),
    ),
)
def test_chase_near_own_goal_routes_around_without_kicking_or_pushing_goalward(
    role: str, player_x: float, ball_x: float
) -> None:
    current = state(
        controlled_x=player_x if role == "controlled" else 0.0,
        opponent_x=player_x if role == "opponent" else 0.0,
        ball_x=ball_x,
    )
    definition = get_action(ChasePolicy(role).select_action(current))
    assert definition["kick"] is False
    assert not ({"left", "right"} & set(definition["keys"]))


def test_center_kickoff_and_opponent_goal_approach_are_team_correct() -> None:
    kickoff = state(controlled_x=-277.5, opponent_x=277.5, ball_x=0.0)
    assert get_action(ChasePolicy("controlled").select_action(kickoff))["name"] == "right"
    assert get_action(ChasePolicy("opponent").select_action(kickoff))["name"] == "left"

    red_ball = state(controlled_x=320.0, opponent_x=0.0, ball_x=350.0)
    blue_ball = state(controlled_x=0.0, opponent_x=-320.0, ball_x=-350.0)
    assert get_action(ChasePolicy("controlled").select_action(red_ball))["name"] == "right"
    assert get_action(ChasePolicy("opponent").select_action(blue_ball))["name"] == "left"


def test_chase_policy_kicks_when_close_and_is_deterministic() -> None:
    current = state(controlled_x=-5.0, opponent_x=5.0)
    red = ChasePolicy("controlled")
    blue = ChasePolicy("opponent")
    assert red.select_action(current) == red.select_action(current) == 13
    assert blue.select_action(current) == blue.select_action(current) == 12


def test_close_aligned_chase_keeps_movement_while_kicking() -> None:
    current = state(controlled_x=-5.0, opponent_x=5.0)
    for role, expected in (("controlled", "right+kick"), ("opponent", "left+kick")):
        definition = get_action(ChasePolicy(role).select_action(current))
        assert definition["name"] == expected
        assert definition["keys"]
        assert definition["kick"] is True


def test_chase_role_mapping_is_independent_of_join_id_order() -> None:
    current = state(controlled_x=-80.0, opponent_x=80.0)
    assert current.controlled.id == 9 and current.opponent.id == 3
    assert get_action(ChasePolicy("controlled").select_action(current))["name"] == "right"
    assert get_action(ChasePolicy("opponent").select_action(current))["name"] == "left"


def test_sanitized_real_state_replay_does_not_stall_at_vertical_only_target() -> None:
    fixture = Path(__file__).parent / "fixtures" / "chase_vertical_oscillation.json"
    current = validate_state_message(json.loads(fixture.read_text()))
    definition = get_action(ChasePolicy("opponent").select_action(current))
    assert definition["name"] == "up-left"


def test_both_policies_generate_valid_independent_actions_from_one_state() -> None:
    current = state(ball_y=20.0)
    controlled = create_policy("chase", "controlled", seed=0)
    opponent = create_policy("persistent-random", "opponent", seed=1)
    actions = (controlled.select_action(current), opponent.select_action(current))
    assert all(0 <= action < 18 for action in actions)


def test_persistent_random_holds_movement_and_resets() -> None:
    current = state()
    policy = PersistentRandomPolicy(5, action_hold_steps=3, kick_probability=0.0)
    first = [policy.select_action(current) for _ in range(3)]
    assert len(set(first)) == 1
    policy.reset()
    assert policy.select_action(current) == first[0]


def test_action_direction_diagnostic_handles_axes_diagonals_and_kick() -> None:
    assert action_displacement_verified(4, 1.0, 0.0)
    assert action_displacement_verified(3, -1.0, 0.0)
    assert action_displacement_verified(6, 1.0, -1.0)
    assert not action_displacement_verified(6, -1.0, -1.0)
    assert action_displacement_verified(9, 0.0, 0.0)

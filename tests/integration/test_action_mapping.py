from integration.controller.actions import ACTIONS, get_action


def test_all_18_actions_have_expected_movement_and_kick() -> None:
    expected = [
        [], ["up"], ["down"], ["left"], ["right"],
        ["up", "left"], ["up", "right"], ["down", "left"], ["down", "right"],
    ]
    assert len(ACTIONS) == 18
    for action_id in range(18):
        action = get_action(action_id)
        assert action["id"] == action_id
        assert action["keys"] == expected[action_id % 9]
        assert action["kick"] is (action_id >= 9)


def test_invalid_actions_are_rejected() -> None:
    for invalid in (-1, 18, True, 1.5):
        try:
            get_action(invalid)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {invalid!r}")

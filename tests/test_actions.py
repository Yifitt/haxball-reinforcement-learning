import numpy as np
import pytest

from haxball_env.actions import MOVEMENT_DIRECTIONS, decode_action, encode_action


@pytest.mark.parametrize("action", range(18))
def test_every_action_decodes(action: int) -> None:
    direction, kick = decode_action(action)
    expected = np.asarray(MOVEMENT_DIRECTIONS[action % 9], dtype=np.float64)
    if np.linalg.norm(expected):
        expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(direction, expected)
    assert kick is (action >= 9)
    assert encode_action(action % 9, kick) == action


@pytest.mark.parametrize("action", [-1, 18, 2.5])
def test_invalid_actions_are_rejected(action: int) -> None:
    with pytest.raises(ValueError):
        decode_action(action)

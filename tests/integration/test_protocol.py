import json

import pytest

from integration.controller.protocol import parse_message, serialize_message


def test_protocol_serialization_round_trip() -> None:
    encoded = serialize_message("action", tick_id=123, client="opponent", action=7)
    assert parse_message(encoded) == {
        "protocol_version": 1,
        "type": "action",
        "tick_id": 123,
        "client": "opponent",
        "action": 7,
    }
    assert encoded == json.dumps(json.loads(encoded), separators=(",", ":"))


@pytest.mark.parametrize(
    "raw",
    ["not-json", "[]", '{"protocol_version":2,"type":"hello"}', '{"protocol_version":1,"type":"wat"}'],
)
def test_invalid_protocol_messages_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_message(raw)


def test_non_finite_serialization_is_rejected() -> None:
    with pytest.raises(ValueError):
        serialize_message("state", value=float("nan"))

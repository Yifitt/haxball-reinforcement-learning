from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
MESSAGE_TYPES = {
    "hello",
    "room_status",
    "client_status",
    "readiness",
    "state",
    "action",
    "action_applied",
    "state_request",
    "reset",
    "error",
    "shutdown",
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class BodyState:
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True, slots=True)
class PlayerState(BodyState):
    id: int
    team: int


@dataclass(frozen=True, slots=True)
class MatchState:
    controlled_side: str
    controlled_score: int
    opponent_score: int
    elapsed_time: float
    last_goal_event: dict[str, Any] | None
    last_touch_team: str | None = None


@dataclass(frozen=True, slots=True)
class StateMessage:
    tick_id: int
    timestamp: float
    game_active: bool
    controlled: PlayerState | None
    opponent: PlayerState | None
    ball: BodyState | None
    match: MatchState
    lifecycle_id: int = 0
    state_sequence: int = 0
    forced_snapshot: bool = False

    @property
    def controlled_player(self) -> PlayerState | None:
        return self.controlled

    @property
    def opponent_player(self) -> PlayerState | None:
        return self.opponent


def _body(value: Any, name: str, *, player: bool) -> BodyState | PlayerState | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object or null")
    fields = {
        key: _finite_number(value.get(key), f"{name}.{key}")
        for key in ("x", "y", "vx", "vy")
    }
    if player:
        team = _integer(value.get("team"), f"{name}.team", minimum=1)
        if team not in (1, 2):
            raise ValueError(f"{name}.team must be 1 or 2")
        return PlayerState(id=_integer(value.get("id"), f"{name}.id"), team=team, **fields)
    return BodyState(**fields)


def validate_state_message(value: Any) -> StateMessage:
    if not isinstance(value, dict):
        raise ValueError("state message must be an object")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("type") != "state":
        raise ValueError("not a protocol v1 state message")
    game_active = value.get("game_active")
    if not isinstance(game_active, bool):
        raise ValueError("game_active must be boolean")
    controlled = _body(value.get("controlled"), "controlled", player=True)
    opponent = _body(value.get("opponent"), "opponent", player=True)
    ball = _body(value.get("ball"), "ball", player=False)
    if game_active and (controlled is None or opponent is None or ball is None):
        raise ValueError("active state requires both players and the ball")

    match = value.get("match")
    if not isinstance(match, dict):
        raise ValueError("match must be an object")
    side = match.get("controlled_side")
    if side not in ("red", "blue"):
        raise ValueError("controlled_side must be red or blue")
    last_goal = match.get("last_goal_event")
    if last_goal is not None:
        if not isinstance(last_goal, dict) or last_goal.get("team") not in ("red", "blue"):
            raise ValueError("invalid last_goal_event")
        _integer(last_goal.get("tick_id"), "last_goal_event.tick_id")
    match_state = MatchState(
        controlled_side=side,
        controlled_score=_integer(match.get("controlled_score"), "controlled_score"),
        opponent_score=_integer(match.get("opponent_score"), "opponent_score"),
        elapsed_time=_finite_number(match.get("elapsed_time"), "elapsed_time"),
        last_goal_event=last_goal,
        last_touch_team=(
            match.get("last_touch_team")
            if match.get("last_touch_team") in ("red", "blue") else None
        ),
    )
    return StateMessage(
        tick_id=_integer(value.get("tick_id"), "tick_id"),
        timestamp=_finite_number(value.get("timestamp"), "timestamp"),
        game_active=game_active,
        controlled=controlled if isinstance(controlled, PlayerState) else None,
        opponent=opponent if isinstance(opponent, PlayerState) else None,
        ball=ball if isinstance(ball, BodyState) else None,
        match=match_state,
        lifecycle_id=_integer(value.get("lifecycle_id", 0), "lifecycle_id"),
        state_sequence=_integer(value.get("state_sequence", 0), "state_sequence"),
        forced_snapshot=value.get("forced_snapshot") is True,
    )


def parse_message(raw: str | bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("message must be an object")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    if value.get("type") not in MESSAGE_TYPES:
        raise ValueError("unknown message type")
    return value


def serialize_message(message_type: str, **fields: Any) -> str:
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"unknown message type: {message_type}")
    return json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "type": message_type, **fields},
        separators=(",", ":"),
        allow_nan=False,
    )

from collections.abc import Iterator
import json
from typing import Any

import pytest

from integration.controller.client import BridgeClient
from integration.controller import client as bridge_client_module
from integration.scripts import random_agent


def client_with_messages(messages: list[dict[str, Any]]) -> BridgeClient:
    client = BridgeClient()
    iterator: Iterator[dict[str, Any]] = iter(messages)
    client.connection = object()  # type: ignore[assignment]
    client.receive = lambda timeout=None: next(iterator)  # type: ignore[method-assign]
    return client


def test_controller_retries_refused_dial_without_duplicate_hello(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    attempts = 0

    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = 0

        def send(self, payload: str) -> None:
            self.sent.append(payload)

        def recv(self, timeout: float) -> str:
            assert timeout == 60.0
            return json.dumps({
                "protocol_version": 1,
                "type": "hello",
                "role": "bridge",
                "accepted": True,
            })

        def close(self) -> None:
            self.closed += 1

    connection = FakeConnection()

    def fake_connect(
        _url: str, *, proxy: None, open_timeout: float,
    ) -> FakeConnection:
        nonlocal attempts
        assert proxy is None
        attempts += 1
        if attempts == 1:
            raise ConnectionRefusedError(111, "connection refused")
        return connection

    monkeypatch.setattr(bridge_client_module, "connect", fake_connect)
    monkeypatch.setattr(bridge_client_module.time, "sleep", lambda _delay: None)
    client = BridgeClient(
        "ws://127.0.0.1:8765", human_opponent=True, public_room=True,
    )
    client.connect(timeout=1.0)

    assert attempts == 2
    sent = [json.loads(payload) for payload in connection.sent]
    assert [payload["type"] for payload in sent] == ["hello", "readiness"]
    assert sum(payload["type"] == "hello" for payload in sent) == 1
    assert client.connection is connection
    output = capsys.readouterr().out
    assert output.count("local_connection_attempt: component=python_controller") == 2
    assert "host=127.0.0.1 port=8765 direct=true attempt=1" in output
    assert "host=127.0.0.1 port=8765 direct=true attempt=2" in output
    assert "error=ConnectionRefusedError errno=111" in output


def test_private_controller_keeps_single_attempt_and_stationary_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_timeouts: list[float] = []
    sent: list[dict[str, Any]] = []

    class FakeConnection:
        def send(self, payload: str) -> None:
            sent.append(json.loads(payload))

        def recv(self, timeout: float) -> str:
            received_timeouts.append(timeout)
            return json.dumps({
                "protocol_version": 1, "type": "hello", "accepted": True,
            })

        def close(self) -> None:
            pass

    attempts = []
    monkeypatch.setattr(
        bridge_client_module,
        "connect",
        lambda _url, *, proxy, open_timeout: (
            attempts.append((proxy, open_timeout)) or FakeConnection()
        ),
    )
    client = BridgeClient("ws://127.0.0.1:8765")
    client.connect()
    assert len(attempts) == 1
    assert attempts[0][0] is None
    assert received_timeouts == [5.0]
    assert [payload["type"] for payload in sent] == ["hello", "readiness"]
    assert sent[-1]["startup_mode"] == "stationary_opponent"


def test_python_waits_for_readiness_and_ignores_early_state_and_nonfatal_error() -> None:
    client = client_with_messages([
        {"type": "state", "tick_id": 1},
        {"type": "error", "code": "not_ready", "fatal": False},
        {"type": "readiness", "barrier_ready": False, "missing": ["controlled_input"]},
        {"type": "readiness", "barrier_ready": True, "missing": []},
    ])
    snapshot = client.wait_until_ready(timeout=1)
    assert snapshot["barrier_ready"] is True
    assert client.pre_ready_states == 1
    assert client.nonfatal_errors == 1


def test_post_ready_disconnect_error_remains_fatal() -> None:
    client = client_with_messages([
        {
            "type": "error",
            "code": "component_disconnected",
            "component": "controlled_browser",
            "fatal": True,
        }
    ])
    with pytest.raises(RuntimeError, match="component_disconnected component=controlled_browser"):
        client.receive_state(timeout=1)


def test_random_agent_waits_for_barrier_before_first_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class FakeClient:
        def __init__(self, _url: str, **kwargs: Any) -> None:
            self.reset_count = 0
            self.first_state_received_count = 0
            self.first_state_received_timestamp_ms = 0.0
            self.rejected_actions = {"controlled": 0, "opponent": 0}

        def connect(self) -> None:
            events.append("connect")

        def wait_until_ready(self, timeout: float) -> None:
            assert timeout == 60.0
            events.append("barrier")

        def receive_state(self, timeout: float) -> Any:
            events.append("state")
            self.first_state_received_count += 1
            return type("State", (), {
                "tick_id": 1,
                "timestamp": 0.0,
                "game_active": True,
                "controlled": None,
                "opponent": None,
                "ball": None,
                "match": None,
                "lifecycle_id": 1,
            })()

        def send_action(
            self, *, tick_id: int, client: str, action: int, lifecycle_id: int = 0,
        ) -> None:
            assert tick_id == 1
            assert lifecycle_id == 1
            events.append(f"action:{client}")

        def drain_applied_events(self) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            events.append("close")

    times = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(random_agent, "BridgeClient", FakeClient)
    monkeypatch.setattr(random_agent.time, "monotonic", lambda: next(times))
    random_agent.run_agent(duration=1.0, seed=0, bridge_url="local")
    assert events == ["connect", "barrier", "state", "action:controlled", "close"]
    assert capsys.readouterr().out.count("controller_action_loop_started: count=1") == 1


def test_human_agent_waits_for_manual_join_without_opponent_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, _url: str, **kwargs: Any) -> None:
            observed.update(kwargs)
            self.reset_count = 0
            self.first_state_received_count = 0
            self.first_state_received_timestamp_ms = 0.0
            self.rejected_actions = {"controlled": 0, "opponent": 0}

        def connect(self) -> None: pass
        def wait_until_ready(self, timeout: float) -> None: observed["timeout"] = timeout
        def close(self) -> None: pass
        def drain_applied_events(self) -> list[dict[str, Any]]: return []

    monkeypatch.setattr(random_agent, "BridgeClient", FakeClient)
    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(random_agent.time, "monotonic", lambda: next(times))
    random_agent.run_agent(
        duration=1.0, seed=0, bridge_url="local", human_opponent=True)
    assert observed["human_opponent"] is True
    assert observed["opponent_control_required"] is False
    assert observed["timeout"] == 86_400.0


def test_public_agent_uses_infrastructure_handshake_timeout_and_queue_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, _url: str, **kwargs: Any) -> None:
            observed.update(kwargs)
            self.reset_count = 0
            self.first_state_received_count = 0
            self.first_state_received_timestamp_ms = 0.0
            self.rejected_actions = {"controlled": 0, "opponent": 0}

        def connect(self) -> None: pass
        def wait_until_ready(self, timeout: float) -> None: observed["timeout"] = timeout
        def close(self) -> None: pass
        def drain_applied_events(self) -> list[dict[str, Any]]: return []

    monkeypatch.setattr(random_agent, "BridgeClient", FakeClient)
    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(random_agent.time, "monotonic", lambda: next(times))
    random_agent.run_agent(
        duration=1.0, seed=0, bridge_url="local",
        human_opponent=True, public_room=True)
    assert observed["public_room"] is True
    assert observed["human_opponent"] is True
    assert observed["timeout"] == 60.0
    output = capsys.readouterr().out
    assert "startup_barrier: mode=public_human_queue ready=true" in output
    assert "stationary_opponent" not in output


def test_bridge_client_accepts_identical_opening_state_after_lifecycle_reset() -> None:
    def state(lifecycle_id: int) -> dict[str, Any]:
        body = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}
        return {
            "protocol_version": 1, "type": "state", "tick_id": 7,
            "timestamp": 1_700_000_000_000.0, "game_active": True,
            "controlled": {"id": 1, "team": 1, **body},
            "opponent": {"id": 2, "team": 2, **body}, "ball": body,
            "match": {
                "controlled_side": "red", "controlled_score": 0,
                "opponent_score": 0, "elapsed_time": 0.0,
                "last_goal_event": None,
            },
            "lifecycle_id": lifecycle_id, "state_sequence": 1,
            "forced_snapshot": True,
        }

    client = client_with_messages([
        {"type": "reset", "reason": "game_start", "lifecycle_id": 1, "tick_id": 6},
        state(1),
        {"type": "reset", "reason": "game_start", "lifecycle_id": 2, "tick_id": 7},
        state(2),
    ])
    first = client.receive_state(timeout=1)
    second = client.receive_state(timeout=1)
    assert first.tick_id == second.tick_id == 7
    assert (first.lifecycle_id, second.lifecycle_id) == (1, 2)
    assert client.first_state_received_count == 2
    assert client.reset_count == 2

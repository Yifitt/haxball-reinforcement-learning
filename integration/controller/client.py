from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

from websockets.sync.client import ClientConnection, connect

from .actions import get_action
from .protocol import StateMessage, parse_message, serialize_message, validate_state_message


class BridgeClient:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:8765",
        *,
        opponent_control_required: bool = False,
        human_opponent: bool = False,
        public_room: bool = False,
    ) -> None:
        self.url = url
        self.opponent_control_required = opponent_control_required
        self.human_opponent = human_opponent
        self.public_room = public_room
        self.startup_mode = (
            "public_human_queue" if public_room else
            "human_opponent" if human_opponent else
            "dual_control" if opponent_control_required else "stationary_opponent"
        )
        self.connection: ClientConnection | None = None
        self.latest_readiness: dict[str, Any] | None = None
        self.pre_ready_states = 0
        self.nonfatal_errors = 0
        self.rejected_actions = {"controlled": 0, "opponent": 0}
        self.applied_events: list[dict[str, Any]] = []
        self.reset_count = 0
        self.current_lifecycle_id: int | None = None
        self.last_received_tick: int | None = None
        self.first_state_received_count = 0
        self.first_state_received_timestamp_ms: float | None = None

    def connect(self, timeout: float = 10.0) -> None:
        if self.connection is not None:
            raise RuntimeError("bridge client is already connected")
        endpoint = urlsplit(self.url)
        host = endpoint.hostname or "unknown"
        port = endpoint.port or (443 if endpoint.scheme == "wss" else 80)
        deadline = time.monotonic() + timeout
        attempt = 0
        delay = 0.05
        while True:
            attempt += 1
            print(
                "local_connection_attempt: component=python_controller "
                f"transport=websocket host={host} port={port} direct=true attempt={attempt}",
                flush=True,
            )
            remaining = max(0.0, deadline - time.monotonic())
            try:
                # The bridge is local-only. websockets 15 enables system proxy
                # discovery by default, which can route even loopback dials away
                # from the bridge when WS_PROXY/HTTPS_PROXY bypass rules are wrong.
                self.connection = connect(
                    self.url,
                    proxy=None,
                    open_timeout=min(5.0, remaining),
                )
                break
            except (OSError, TimeoutError) as error:
                error_name = type(error).__name__
                error_number = getattr(error, "errno", None)
                print(
                    "local_connection_retry: component=python_controller "
                    f"host={host} port={port} error={error_name} "
                    f"errno={error_number if error_number is not None else 'none'}",
                    flush=True,
                )
                if time.monotonic() >= deadline:
                    raise ConnectionError(
                        "bridge WebSocket connection failed "
                        f"host={host} port={port} attempts={attempt}"
                    ) from error
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(delay * 2.0, 0.5)

        try:
            # Dial retries happen before protocol traffic. A successful socket gets
            # exactly one controller hello and one readiness declaration.
            self.connection.send(serialize_message("hello", role="agent"))
            hello_timeout = 60.0 if self.public_room else 5.0
            hello = parse_message(self.connection.recv(timeout=hello_timeout))
            if hello.get("type") != "hello" or hello.get("accepted") is not True:
                raise RuntimeError("bridge did not acknowledge the Python controller protocol")
            self.connection.send(serialize_message(
                "readiness",
                python_protocol_ready=True,
                opponent_control_required=self.opponent_control_required,
                human_opponent=self.human_opponent,
                public_room=self.public_room,
                startup_mode=self.startup_mode,
            ))
        except Exception:
            self.close()
            raise

    def _handle_control_message(self, payload: dict[str, Any]) -> bool:
        if payload["type"] == "readiness":
            self.latest_readiness = payload
            return True
        if payload["type"] == "shutdown":
            raise ConnectionAbortedError("bridge requested shutdown")
        if payload["type"] == "error":
            if payload.get("fatal") is False:
                self.nonfatal_errors += 1
                client = payload.get("client")
                if client in self.rejected_actions:
                    self.rejected_actions[client] += 1
                return True
            details = payload.get("code", "unknown")
            if payload.get("component"):
                details += f" component={payload['component']}"
            if payload.get("missing"):
                details += f" missing={','.join(payload['missing'])}"
            raise RuntimeError(f"bridge error: {details}")
        if payload["type"] == "action_applied":
            self.applied_events.append(payload)
            return True
        if payload["type"] == "reset":
            self.reset_count += 1
            self.current_lifecycle_id = (
                payload["lifecycle_id"] if isinstance(payload.get("lifecycle_id"), int) else None
            )
            self.last_received_tick = None
            return True
        return False

    def wait_until_ready(self, timeout: float = 60.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                payload = self.receive(timeout=remaining)
            except TimeoutError:
                break
            if payload["type"] == "state":
                self.pre_ready_states += 1
                continue
            if self._handle_control_message(payload):
                if self.latest_readiness is not None and self.latest_readiness.get("barrier_ready") is True:
                    return self.latest_readiness
        missing = (self.latest_readiness or {}).get("missing", ["readiness_snapshot"])
        raise TimeoutError(f"startup readiness barrier timed out; missing={','.join(missing)}")

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("bridge client is not connected")
        return parse_message(self.connection.recv(timeout=timeout))

    def receive_state(self, timeout: float | None = None) -> StateMessage:
        while True:
            payload = self.receive(timeout)
            if payload["type"] == "state":
                state = validate_state_message(payload)
                if state.lifecycle_id != self.current_lifecycle_id:
                    self.current_lifecycle_id = state.lifecycle_id
                    self.last_received_tick = None
                if self.last_received_tick is None:
                    self.first_state_received_count += 1
                    self.first_state_received_timestamp_ms = time.time() * 1000.0
                self.last_received_tick = state.tick_id
                return state
            self._handle_control_message(payload)

    def send_action(
        self, *, tick_id: int, client: str = "controlled", action: int,
        lifecycle_id: int = 0,
    ) -> None:
        get_action(action)
        if client not in ("controlled", "opponent"):
            raise ValueError(f"unknown action client: {client}")
        if self.connection is None:
            raise RuntimeError("bridge client is not connected")
        self.connection.send(serialize_message(
            "action", tick_id=tick_id, client=client, action=action,
            lifecycle_id=lifecycle_id))

    def drain_applied_events(self) -> list[dict[str, Any]]:
        events = self.applied_events
        self.applied_events = []
        return events

    def shutdown(self) -> None:
        if self.connection is not None:
            self.connection.send(serialize_message("shutdown", reason="agent_finished"))

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "BridgeClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

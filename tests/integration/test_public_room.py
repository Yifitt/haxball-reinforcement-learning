from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import threading

import pytest

from integration.scripts import smoke_real_haxball
from integration.scripts import random_agent


def test_public_room_defaults_match_release_configuration() -> None:
    options = smoke_real_haxball.resolve_room_options(
        public_room=True, room_name=None, max_players=None,
        score_limit=None, time_limit=None)
    assert options == {
        "public_room": True,
        "room_name": "RL Bot | 1v1",
        "max_players": 12,
        "score_limit": 5,
        "time_limit": 0,
        "enable_player_queue": True,
        "matches_per_turn": 1,
        "queue_afk_timeout": 0.0,
    }


def test_public_room_configuration_rejects_unsafe_counts() -> None:
    with pytest.raises(ValueError, match="max_players"):
        smoke_real_haxball.resolve_room_options(
            public_room=True, room_name=None, max_players=1,
            score_limit=None, time_limit=None)


def test_public_dry_run_uses_release_without_exposing_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    credential = "test-token-placeholder"
    monkeypatch.setenv("HAXBALL_HEADLESS_TOKEN", credential)
    monkeypatch.setattr(sys, "argv", ["smoke_real_haxball", "--public-room", "--dry-run", "--json"])
    smoke_real_haxball.main()
    output = capsys.readouterr().out
    assert credential not in output
    report = json.loads(output)
    assert report["authenticated_room_launched"] is False
    assert report["checkpoint"].endswith("checkpoints/releases/selfplay_v1/model.pt")
    assert report["public_room"] is True
    assert report["startup_mode"] == "public_human_queue"


def test_exact_public_human_command_resolves_queue_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "smoke_real_haxball", "--public-room", "--human-opponent",
        "--enable-player-queue", "--dry-run", "--json",
    ])
    smoke_real_haxball.main()
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["startup_mode"] == "public_human_queue"
    assert report["human_opponent"] is True
    assert "mode=public_human_queue" in captured.err
    assert "mode=stationary_opponent" not in captured.err
    assert "private_room" not in captured.err


def test_removed_data_collection_flag_is_rejected_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "smoke_real_haxball", "--public-room", "--record-" + "human-matches",
        "--dry-run", "--json",
    ])
    with pytest.raises(SystemExit, match="2"):
        smoke_real_haxball.main()
    assert not (tmp_path / "data").exists()


def test_controller_rejects_removed_data_collection_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "random_agent", "--record-" + "human-matches",
    ])
    with pytest.raises(SystemExit, match="2"):
        random_agent.main()


def test_queue_cli_options_are_reported_by_dry_run(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "smoke_real_haxball", "--public-room", "--enable-player-queue",
        "--max-players", "14", "--matches-per-turn", "3",
        "--queue-afk-timeout", "0", "--dry-run", "--json",
    ])
    smoke_real_haxball.main()
    report = json.loads(capsys.readouterr().out)
    assert report["enable_player_queue"] is True
    assert report["max_players"] == 14
    assert report["matches_per_turn"] == 3
    assert report["queue_afk_timeout"] == 0


def test_nonzero_afk_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="AFK detection is disabled"):
        smoke_real_haxball.resolve_room_options(
            public_room=True, room_name=None, max_players=None,
            score_limit=None, time_limit=None, queue_afk_timeout=30,
        )


def test_real_orchestrator_passes_one_public_startup_config_to_bridge_host_and_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[list[str], dict[str, str]]] = []
    supervised: dict[str, object] = {}
    events: list[str] = []

    class FakeProcess:
        def poll(self) -> int: return 0

    def fake_popen(command: list[str], *, cwd: object, env: dict[str, str]) -> FakeProcess:
        events.append("host_launch")
        launches.append((command, env))
        return FakeProcess()

    class FakeBridge:
        def __init__(self, environment: dict[str, str]) -> None:
            self.environment = environment
            self.listening = threading.Event()
            self.listening.set()
            self.stopped = False

        def poll(self) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

    bridge_holder: dict[str, FakeBridge] = {}

    def fake_bridge_launch(
        _cls: type[smoke_real_haxball.BridgeChild],
        command: list[str], *, cwd: Path, env: dict[str, str],
    ) -> FakeBridge:
        events.append("bridge_launch")
        assert command[-1] == "bridge/websocket_server.js"
        bridge = FakeBridge(env)
        bridge_holder["bridge"] = bridge
        return bridge

    def fake_wait_for_bridge(
        bridge: FakeBridge, host: str, port: int, timeout: float = 10.0,
    ) -> None:
        events.append("bridge_ready")
        assert bridge is bridge_holder["bridge"]
        assert (host, port) == ("127.0.0.1", 8765)

    class FakeSupervisor:
        def __init__(self, command: list[str], *, cwd: object, env: dict[str, str]) -> None:
            supervised.update(command=command, environment=env)

        def start(self) -> None:
            events.append("browser_launch")

        def stop(self) -> None:
            pass

    monkeypatch.setenv("HAXBALL_HEADLESS_TOKEN", "test-token-placeholder")
    monkeypatch.setattr(smoke_real_haxball.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        smoke_real_haxball.BridgeChild, "launch", classmethod(fake_bridge_launch),
    )
    monkeypatch.setattr(smoke_real_haxball, "BotClientSupervisor", FakeSupervisor)
    monkeypatch.setattr(smoke_real_haxball, "wait_for_bridge", fake_wait_for_bridge)
    monkeypatch.setattr(smoke_real_haxball, "stop_process", lambda _process: None)
    monkeypatch.setattr(
        smoke_real_haxball,
        "run_agent",
        lambda **_kwargs: events.append("controller_launch") or {},
    )
    room_options = smoke_real_haxball.resolve_room_options(
        public_room=True, room_name=None, max_players=None,
        score_limit=None, time_limit=None,
    )

    smoke_real_haxball.real_smoke(
        duration=1, seed=0, headed=False, policy="checkpoint",
        opponent_policy="stationary", action_hold_steps=6,
        kick_probability=0.1, action_diagnostic=False,
        checkpoint=str(smoke_real_haxball.RELEASE_CHECKPOINT),
        human_opponent=True, room_options=room_options,
    )

    host_command, host_environment = launches[0]
    bridge_environment = bridge_holder["bridge"].environment
    assert host_command[-1] == "headless_host/launch_host.js"
    assert bridge_environment is host_environment is supervised["environment"]
    assert bridge_environment["HAXBALL_STARTUP_MODE"] == "public_human_queue"
    assert bridge_environment["HAXBALL_PUBLIC_ROOM"] == "1"
    assert bridge_environment["HAXBALL_HUMAN_OPPONENT"] == "1"
    assert bridge_environment["HAXBALL_OPPONENT_CONTROL_REQUIRED"] == "0"
    assert bridge_environment["HAXBALL_BRIDGE_URL"] == "ws://127.0.0.1:8765"
    assert events == [
        "bridge_launch", "bridge_ready", "host_launch", "browser_launch",
        "controller_launch",
    ]
    assert bridge_holder["bridge"].stopped is True


def test_bridge_readiness_retries_until_delayed_listener_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    class FakeBridge:
        def __init__(self) -> None:
            self.listening = threading.Event()

        def poll(self) -> None:
            return None

    bridge = FakeBridge()
    connection_attempts = 0

    def fake_connection(_endpoint: tuple[str, int], *, timeout: float) -> FakeConnection:
        nonlocal connection_attempts
        connection_attempts += 1
        if connection_attempts == 1:
            raise ConnectionRefusedError(111, "connection refused")
        bridge.listening.set()
        return FakeConnection()

    monkeypatch.setattr(smoke_real_haxball.socket, "create_connection", fake_connection)
    monkeypatch.setattr(smoke_real_haxball.time, "sleep", lambda _delay: None)
    smoke_real_haxball.wait_for_bridge(bridge, "127.0.0.1", 8765, timeout=1.0)  # type: ignore[arg-type]
    assert connection_attempts == 3


def test_bridge_listening_then_early_exit_surfaces_exit_code_and_stderr() -> None:
    environment = os.environ.copy()
    child = smoke_real_haxball.BridgeChild.launch(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('bridge_listening: ws://127.0.0.1:1', flush=True); "
                "print('deterministic bridge failure', file=sys.stderr, flush=True); "
                "raise SystemExit(23)"
            ),
        ],
        cwd=Path.cwd(),
        env=environment,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            smoke_real_haxball.wait_for_bridge(child, "127.0.0.1", 1, timeout=1.0)
        detail = str(raised.value)
        assert "exit_code=23" in detail
        assert "deterministic bridge failure" in detail
    finally:
        child.stop()


def test_bridge_diagnostics_redact_credentials_and_room_urls() -> None:
    safe = smoke_real_haxball._redact_bridge_output(
        "token-value password-value https://www.haxball.com/play?c=private-code",
        ("token-value", "password-value"),
    )
    assert "token-value" not in safe
    assert "password-value" not in safe
    assert "private-code" not in safe
    assert safe.count("[REDACTED_SECRET]") == 2
    assert "[REDACTED_ROOM_URL]" in safe


def test_unrelated_port_occupant_does_not_satisfy_bridge_readiness() -> None:
    class FakeBridge:
        def __init__(self) -> None:
            self.listening = threading.Event()

        def poll(self) -> None:
            return None

    with socket.socket() as unrelated:
        unrelated.bind(("127.0.0.1", 0))
        unrelated.listen()
        port = int(unrelated.getsockname()[1])
        with pytest.raises(RuntimeError) as raised:
            smoke_real_haxball.wait_for_bridge(
                FakeBridge(), "127.0.0.1", port, timeout=0.05,  # type: ignore[arg-type]
            )
    detail = str(raised.value)
    assert "tcp_accepting=true" in detail
    assert "port_may_be_occupied_by_unrelated_process=true" in detail


def test_orchestrator_stops_bridge_and_skips_host_when_readiness_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBridge:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    bridge = FakeBridge()

    def fake_launch(
        _cls: type[smoke_real_haxball.BridgeChild],
        _command: list[str], *, cwd: Path, env: dict[str, str],
    ) -> FakeBridge:
        return bridge

    monkeypatch.setenv("HAXBALL_HEADLESS_TOKEN", "test-token-placeholder")
    monkeypatch.setattr(
        smoke_real_haxball.BridgeChild, "launch", classmethod(fake_launch),
    )
    monkeypatch.setattr(
        smoke_real_haxball,
        "wait_for_bridge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(
        smoke_real_haxball.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("host must not launch before bridge readiness"),
    )
    options = smoke_real_haxball.resolve_room_options(
        public_room=False, room_name=None, max_players=None,
        score_limit=None, time_limit=None,
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        smoke_real_haxball.real_smoke(
            duration=1, seed=0, headed=False, policy="random",
            opponent_policy="stationary", action_hold_steps=6,
            kick_probability=0.1, action_diagnostic=False,
            checkpoint=None, room_options=options,
        )
    assert bridge.stopped is True

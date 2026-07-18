import pytest
from types import SimpleNamespace

from integration.controller.metrics import AgentMetrics, percentile


def test_latency_percentiles() -> None:
    values = [10.0, 40.0, 20.0, 30.0]
    assert percentile(values, 0.5) == 20.0
    assert percentile(values, 0.95) == 40.0
    assert percentile([], 0.5) is None


def test_tick_gaps_count_dropped_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("integration.controller.metrics.time.time", lambda: 100.0)
    metrics = AgentMetrics()
    metrics.record_state(4, 99_990.0)
    metrics.record_state(7, 99_995.0)
    assert metrics.states == 2
    assert metrics.dropped_states == 2
    assert metrics.host_to_agent_ms == [10.0, 5.0]


def test_dual_client_metrics_remain_independent() -> None:
    metrics = AgentMetrics()
    state = SimpleNamespace(
        controlled=SimpleNamespace(x=-10.0, y=0.0, team=1),
        opponent=SimpleNamespace(x=20.0, y=0.0, team=2),
        ball=SimpleNamespace(x=0.0, y=0.0),
        match=SimpleNamespace(last_goal_event=None),
    )
    metrics.record_dual_state(state)
    state.controlled.x = -7.0
    state.opponent.x = 16.0
    metrics.record_dual_state(state)
    metrics.record_action("controlled", 13)
    metrics.record_action("opponent", 3)
    metrics.record_applied("controlled")
    report = metrics.report()
    assert report["controlled_total_displacement"] == 3.0
    assert report["opponent_total_displacement"] == 4.0
    assert report["controlled_kicks"] == 1
    assert report["opponent_kicks"] == 0
    assert report["controlled_applied_actions_per_second"] > 0
    assert report["opponent_applied_actions_per_second"] == 0

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass(slots=True)
class AgentMetrics:
    started_at: float = field(default_factory=time.monotonic)
    states: int = 0
    actions: int = 0
    dropped_states: int = 0
    invalid_states: int = 0
    last_tick: int | None = None
    host_to_agent_ms: list[float] = field(default_factory=list)
    actions_by_client: dict[str, int] = field(default_factory=lambda: {"controlled": 0, "opponent": 0})
    applied_by_client: dict[str, int] = field(default_factory=lambda: {"controlled": 0, "opponent": 0})
    kicks_by_client: dict[str, int] = field(default_factory=lambda: {"controlled": 0, "opponent": 0})
    displacement_by_client: dict[str, float] = field(default_factory=lambda: {"controlled": 0.0, "opponent": 0.0})
    ball_distance_sum: dict[str, float] = field(default_factory=lambda: {"controlled": 0.0, "opponent": 0.0})
    ball_distance_samples: dict[str, int] = field(default_factory=lambda: {"controlled": 0, "opponent": 0})
    last_position: dict[str, tuple[float, float] | None] = field(
        default_factory=lambda: {"controlled": None, "opponent": None})
    goals_by_client: dict[str, int] = field(default_factory=lambda: {"controlled": 0, "opponent": 0})
    last_goal_tick: int | None = None

    def record_state(self, tick_id: int, timestamp_ms: float) -> None:
        self.states += 1
        if self.last_tick is not None and tick_id > self.last_tick + 1:
            self.dropped_states += tick_id - self.last_tick - 1
        self.last_tick = tick_id
        latency = time.time() * 1000.0 - timestamp_ms
        if math.isfinite(latency) and latency >= 0:
            self.host_to_agent_ms.append(latency)
            if len(self.host_to_agent_ms) > 10_000:
                del self.host_to_agent_ms[: len(self.host_to_agent_ms) - 10_000]

    def record_dual_state(self, state: object) -> None:
        ball = getattr(state, "ball", None)
        for client in ("controlled", "opponent"):
            player = getattr(state, client, None)
            if player is None or ball is None:
                self.last_position[client] = None
                continue
            position = (player.x, player.y)
            previous = self.last_position[client]
            if previous is not None:
                self.displacement_by_client[client] += math.hypot(
                    position[0] - previous[0], position[1] - previous[1])
            self.last_position[client] = position
            self.ball_distance_sum[client] += math.hypot(player.x - ball.x, player.y - ball.y)
            self.ball_distance_samples[client] += 1
        goal = getattr(getattr(state, "match", None), "last_goal_event", None)
        if goal and goal.get("tick_id") != self.last_goal_tick:
            self.last_goal_tick = goal["tick_id"]
            scoring_team = 1 if goal["team"] == "red" else 2
            controlled = getattr(state, "controlled", None)
            scorer = "controlled" if controlled and controlled.team == scoring_team else "opponent"
            self.goals_by_client[scorer] += 1

    def record_action(self, client: str, action: int) -> None:
        self.actions += 1
        self.actions_by_client[client] += 1
        if action >= 9:
            self.kicks_by_client[client] += 1

    def record_applied(self, client: str) -> None:
        if client in self.applied_by_client:
            self.applied_by_client[client] += 1

    def reset_episode(self) -> None:
        self.last_position = {"controlled": None, "opponent": None}

    def report(self) -> dict[str, float | int | None]:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        report = {
            "elapsed_seconds": elapsed,
            "states": self.states,
            "actions": self.actions,
            "states_per_second": self.states / elapsed,
            "actions_per_second": self.actions / elapsed,
            "dropped_states": self.dropped_states,
            "invalid_states": self.invalid_states,
            "median_host_to_agent_ms": percentile(self.host_to_agent_ms, 0.5),
            "p95_host_to_agent_ms": percentile(self.host_to_agent_ms, 0.95),
        }
        for client in ("controlled", "opponent"):
            samples = self.ball_distance_samples[client]
            report[f"{client}_actions_per_second"] = self.actions_by_client[client] / elapsed
            report[f"{client}_applied_actions_per_second"] = self.applied_by_client[client] / elapsed
            report[f"{client}_total_displacement"] = self.displacement_by_client[client]
            report[f"{client}_mean_ball_distance"] = (
                self.ball_distance_sum[client] / samples if samples else None)
            report[f"{client}_kicks"] = self.kicks_by_client[client]
            report[f"goals_by_{client}"] = self.goals_by_client[client]
        return report

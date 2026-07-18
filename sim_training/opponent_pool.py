from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy_contract.chase_contract import ACTUAL_KICK_RANGE, goal_directed_chase
from sim_training.policy import chase_bins

BASE_OPPONENTS = (
    "standard_chase",
    "delayed_chase",
    "noisy_chase",
    "defensive_chase",
    "aggressive_chase",
    "mixed_persistent_random_chase",
)
MODEL_OPPONENTS = ("previous_policy", "self_play")


@dataclass(frozen=True)
class OpponentPoolConfiguration:
    names: tuple[str, ...]
    weights: tuple[float, ...]


def _physical_to_policy_bins(dx: np.ndarray, dy: np.ndarray, kick: np.ndarray, team: np.ndarray):
    mirror_x = np.where(team == 2, 1, -1)  # upstream Red=2, Blue=4
    return np.stack((dx * mirror_x + 1, dy + 1, kick), axis=-1).astype(np.int64)


class OpponentPool:
    """Episode-sampled, vectorized scripted/model opponent pool."""

    def __init__(
        self,
        n_envs: int,
        *,
        seed: int,
        configuration: OpponentPoolConfiguration,
        previous_models: tuple[object, ...] = (),
        self_play_models: tuple[object, ...] = (),
        self_play_labels: tuple[str, ...] = (),
        self_play_weights: tuple[float, ...] | None = None,
    ) -> None:
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)
        self.configuration = configuration
        weights = np.asarray(configuration.weights, dtype=np.float64)
        if len(configuration.names) != len(weights) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("opponent pool names and normalized weights must match")
        if any(name not in BASE_OPPONENTS + MODEL_OPPONENTS for name in configuration.names):
            raise ValueError("unknown opponent pool member")
        self.previous_models = previous_models
        self.self_play_models = self_play_models
        if self_play_models and len(self_play_models) != len(self_play_labels):
            raise ValueError("self-play models and labels must match")
        self.self_play_labels = self_play_labels
        self._self_play_by_label = dict(zip(self_play_labels, self_play_models, strict=True))
        supplied_weights = np.asarray(
            self_play_weights if self_play_weights is not None else np.ones(len(self_play_labels)),
            dtype=np.float64,
        )
        if len(supplied_weights) != len(self_play_labels) or (
            len(self_play_labels) and supplied_weights.sum() <= 0
        ):
            raise ValueError("self-play weights must be positive and match labels")
        self.self_play_weights = (
            supplied_weights / supplied_weights.sum() if len(self_play_labels)
            else supplied_weights
        )
        self.assignments = np.empty(n_envs, dtype=object)
        self.delay_remaining = np.zeros(n_envs, dtype=np.int16)
        self.mixed_remaining = np.zeros(n_envs, dtype=np.int16)
        self.mixed_action = np.ones((n_envs, 3), dtype=np.int64)
        self.model_index = np.zeros(n_envs, dtype=np.int16)
        self.model_assignment = np.full(n_envs, "", dtype=object)
        self.reset(np.ones(n_envs, dtype=bool))

    def force_assignments(self, labels: np.ndarray) -> None:
        """Force deterministic evaluation assignments without changing training sampling."""
        labels = np.asarray(labels, dtype=object)
        if labels.shape != (self.n_envs,):
            raise ValueError("forced opponent labels must match n_envs")
        model_labels = np.isin(labels, np.asarray(self.self_play_labels, dtype=object))
        unknown = ~model_labels & ~np.isin(labels, np.asarray(BASE_OPPONENTS, dtype=object))
        if unknown.any():
            raise ValueError(f"unknown forced opponent label: {labels[np.flatnonzero(unknown)[0]]}")
        self.assignments[:] = labels
        self.assignments[model_labels] = "self_play"
        self.model_assignment[model_labels] = labels[model_labels]
        self.delay_remaining[:] = 0
        self.mixed_remaining[:] = 0

    def reset(self, mask: np.ndarray, kickoff_delay: np.ndarray | None = None) -> None:
        indices = np.flatnonzero(mask)
        if not indices.size:
            return
        choices = self.rng.choice(
            len(self.configuration.names), size=indices.size, p=self.configuration.weights)
        self.assignments[indices] = np.asarray(self.configuration.names, dtype=object)[choices]
        base_delay = self.rng.integers(2, 6, size=indices.size)
        if kickoff_delay is not None:
            base_delay += np.asarray(kickoff_delay)[indices]
        delayed = self.assignments[indices] == "delayed_chase"
        self.delay_remaining[indices] = np.where(delayed, base_delay, 0)
        self.mixed_remaining[indices] = 0
        for name, models in (("previous_policy", self.previous_models),):
            selected = indices[self.assignments[indices] == name]
            if selected.size:
                if not models:
                    raise ValueError(f"opponent pool selected {name} without checkpoint models")
                self.model_index[selected] = self.rng.integers(0, len(models), size=selected.size)

        selected = indices[self.assignments[indices] == "self_play"]
        if selected.size:
            if not self.self_play_labels:
                raise ValueError("opponent pool selected self_play without checkpoint models")
            choices = self.rng.choice(
                len(self.self_play_labels), size=selected.size, p=self.self_play_weights)
            self.model_assignment[selected] = np.asarray(self.self_play_labels, dtype=object)[choices]

    def set_self_play_models(
        self, models: tuple[object, ...], labels: tuple[str, ...],
        weights: tuple[float, ...] | None = None,
    ) -> None:
        """Change the active generation set without changing ongoing episodes."""
        if len(models) != len(labels) or len(labels) != len(set(labels)):
            raise ValueError("self-play models need unique matching labels")
        active = dict(zip(labels, models, strict=True))
        in_use = set(self.model_assignment[self.assignments == "self_play"].tolist())
        retained = {
            label: model for label, model in self._self_play_by_label.items()
            if label in in_use
        }
        self._self_play_by_label = {**retained, **active}
        self.self_play_models = models
        self.self_play_labels = labels
        supplied = np.asarray(
            weights if weights is not None else np.ones(len(labels)), dtype=np.float64)
        if len(supplied) != len(labels) or (len(labels) and supplied.sum() <= 0):
            raise ValueError("self-play weights must be positive and match labels")
        self.self_play_weights = supplied / supplied.sum() if len(labels) else supplied

    def _goal_chase(self, state, player_index: int | np.ndarray, **kwargs) -> np.ndarray:
        rows = np.arange(self.n_envs)
        indices = np.broadcast_to(np.asarray(player_index, dtype=np.int64), rows.shape)
        team = state.team[rows, indices]
        browser_team = np.where(team == 2, 1, 2)
        decision = goal_directed_chase(
            state.player_pos[rows, indices], state.ball_pos, browser_team, **kwargs)
        return _physical_to_policy_bins(decision.dx, decision.dy, decision.kick, team)

    def actions(
        self, state, player_index: int | np.ndarray,
        opponent_observations: np.ndarray | None = None,
    ) -> np.ndarray:
        standard = chase_bins(state, player_index)
        output = standard.copy()
        rows = np.arange(self.n_envs)
        indices = np.broadcast_to(np.asarray(player_index, dtype=np.int64), rows.shape)
        team = state.team[rows, indices]

        delayed = self.assignments == "delayed_chase"
        waiting = delayed & (self.delay_remaining > 0)
        output[waiting] = np.asarray((1, 1, 0))
        self.delay_remaining[waiting] -= 1

        noisy = self.assignments == "noisy_chase"
        noisy_pause = noisy & (self.rng.random(self.n_envs) < 0.12)
        output[noisy_pause] = np.asarray((1, 1, 0))
        noisy_vertical = noisy & ~noisy_pause & (self.rng.random(self.n_envs) < 0.18)
        output[noisy_vertical, 1] = self.rng.integers(0, 3, noisy_vertical.sum())
        output[noisy_vertical, 2] = 0

        defensive = self.assignments == "defensive_chase"
        attack = np.where(team == 2, 1.0, -1.0)
        ball_on_own_half = state.ball_pos[:, 0] * attack < 0.0
        hold = defensive & ~ball_on_own_half
        defensive_anchor_x = -attack * 220.0
        player_x = state.player_pos[rows, indices, 0]
        player_y = state.player_pos[rows, indices, 1]
        anchor_delta_x = defensive_anchor_x - player_x
        ball_delta_x = state.ball_pos[:, 0] - player_x
        crosses_ball = (
            (ball_delta_x * anchor_delta_x > 0.0)
            & (np.abs(ball_delta_x) < np.abs(anchor_delta_x))
            & (np.abs(state.ball_pos[:, 1] - player_y) < ACTUAL_KICK_RANGE + 7.0)
        )
        dx = np.sign(anchor_delta_x).astype(np.int64)
        dx[crosses_ball] = 0
        route_dy = np.where(state.ball_pos[:, 1] >= player_y, -1, 1)
        dy = np.sign(-player_y).astype(np.int64)
        dy[crosses_ball] = route_dy[crosses_ball]
        output[hold] = _physical_to_policy_bins(
            dx, dy, np.zeros(self.n_envs, dtype=np.int64), team)[hold]

        aggressive = self.assignments == "aggressive_chase"
        if aggressive.any():
            aggressive_actions = self._goal_chase(
                state, player_index, behind_ball_distance=16.0,
                alignment_tolerance=20.0, kick_range=ACTUAL_KICK_RANGE)
            output[aggressive] = aggressive_actions[aggressive]

        mixed = self.assignments == "mixed_persistent_random_chase"
        refresh = mixed & (self.mixed_remaining <= 0) & (self.rng.random(self.n_envs) < 0.22)
        self.mixed_action[refresh, 0] = self.rng.integers(0, 3, refresh.sum())
        self.mixed_action[refresh, 1] = self.rng.integers(0, 3, refresh.sum())
        self.mixed_action[refresh, 2] = 0
        self.mixed_remaining[refresh] = self.rng.integers(2, 6, refresh.sum())
        distance = np.linalg.norm(
            state.ball_pos - state.player_pos[rows, indices], axis=-1)
        use_random = mixed & (self.mixed_remaining > 0) & (distance > 60.0)
        output[use_random] = self.mixed_action[use_random]
        self.mixed_remaining[mixed & (self.mixed_remaining > 0)] -= 1

        if opponent_observations is not None:
            for name, models in (("previous_policy", self.previous_models),):
                assigned = self.assignments == name
                for index, model in enumerate(models):
                    selected = assigned & (self.model_index == index)
                    if selected.any():
                        output[selected] = model.predict_bins(opponent_observations[selected])
            assigned = self.assignments == "self_play"
            for label, model in self._self_play_by_label.items():
                selected = assigned & (self.model_assignment == label)
                if selected.any():
                    output[selected] = model.predict_bins(opponent_observations[selected])
        return output

    def names(self) -> np.ndarray:
        return self.assignments.copy()

    def labels(self) -> np.ndarray:
        labels = self.assignments.copy()
        selected = self.assignments == "self_play"
        labels[selected] = self.model_assignment[selected]
        return labels

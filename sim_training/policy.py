from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from policy_contract.checkpoint_contract import PortablePolicy
from policy_contract.chase_contract import goal_directed_chase


def sample_actions(model: PortablePolicy, observations: np.ndarray):
    tensor = torch.as_tensor(observations, dtype=torch.float32)
    logits_x, logits_y, logits_k, values = model(tensor)
    distributions = tuple(Categorical(logits=logits) for logits in (logits_x, logits_y, logits_k))
    bins = torch.stack(tuple(distribution.sample() for distribution in distributions), dim=-1)
    log_probability = sum(
        distribution.log_prob(bins[:, index])
        for index, distribution in enumerate(distributions)
    )
    entropy = sum(distribution.entropy() for distribution in distributions)
    return bins, log_probability, entropy, values


def evaluate_actions(model: PortablePolicy, observations: torch.Tensor, bins: torch.Tensor):
    logits_x, logits_y, logits_k, values = model(observations)
    distributions = tuple(Categorical(logits=logits) for logits in (logits_x, logits_y, logits_k))
    log_probability = sum(
        distribution.log_prob(bins[:, index])
        for index, distribution in enumerate(distributions)
    )
    entropy = sum(distribution.entropy() for distribution in distributions)
    return log_probability, entropy, values


def chase_bins(state, player_index: int | np.ndarray) -> np.ndarray:
    """Goal-directed chase converted once into upstream's normalized frame."""
    rows = np.arange(len(state.ball_pos))
    indices = np.broadcast_to(np.asarray(player_index, dtype=np.int64), rows.shape)
    team = state.team[rows, indices]
    browser_team = np.where(team == 2, 1, np.where(team == 4, 2, 0))
    decision = goal_directed_chase(
        state.player_pos[rows, indices], state.ball_pos, browser_team)
    mirror_x = np.where(team == 2, 1, -1)  # upstream RED=2, BLUE=4
    policy_x = decision.dx * mirror_x
    return np.stack((policy_x + 1, decision.dy + 1, decision.kick), axis=-1)


@torch.no_grad()
def deterministic_bins(model: PortablePolicy, observations: np.ndarray) -> np.ndarray:
    return model.predict_bins(observations)


@torch.no_grad()
def deterministic_bins_with_entropy(
    model: PortablePolicy, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One batched forward pass for deterministic actions and policy entropy."""
    tensor = torch.as_tensor(observations, dtype=torch.float32)
    logits_x, logits_y, logits_k, _ = model(tensor)
    logits = (logits_x, logits_y, logits_k)
    bins = torch.stack(tuple(value.argmax(-1) for value in logits), dim=-1)
    entropy = sum(Categorical(logits=value).entropy() for value in logits)
    return bins.numpy(), entropy.numpy()

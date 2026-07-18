from __future__ import annotations

from sim_training.observation_builder import contract_obs_builder_class
from sim_training.reward_config import build_reward
from sim_training.randomized_reset import RandomizedKickoffMutator
from sim_training.tracking_action import TrackingDiscreteAction
from sim_training.pulse_engine import PulseTransitionEngine
from sim_training.reward_config import (
    DEFAULT_INVALID_KICK_PENALTY,
    DEFAULT_REPEATED_INVALID_KICK_PENALTY,
)


def make_training_env(
    n_envs: int,
    *,
    action_repeat: int = 8,
    step_limit: int = 2000,
    seed: int = 0,
    randomized_starts: bool = True,
    invalid_kick_penalty: float = DEFAULT_INVALID_KICK_PENALTY,
    repeated_invalid_kick_penalty: float = DEFAULT_REPEATED_INVALID_KICK_PENALTY,
):
    from haxballgym import TransitionEngine
    from haxballgym.done import GoalCondition, TimeoutCondition
    from haxballgym.env import Env
    from haxballgym.mutator import KickoffMutator

    if n_envs < 1 or action_repeat < 1:
        raise ValueError("n_envs and action_repeat must be positive")
    engine = PulseTransitionEngine(
        TransitionEngine(n_envs, 1, 1, step_limit=step_limit, tick_skip=1),
        action_repeat=action_repeat,
    )
    builder = contract_obs_builder_class()()
    action_parser = TrackingDiscreteAction(kick_values=2)
    mutator = (
        RandomizedKickoffMutator(seed, action_repeat=action_repeat)
        if randomized_starts else KickoffMutator()
    )
    return Env(
        engine=engine,
        obs_builder=builder,
        action_parser=action_parser,
        reward_fn=build_reward(
            action_parser, invalid_kick_penalty=invalid_kick_penalty,
            repeated_invalid_kick_penalty=repeated_invalid_kick_penalty),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(step_limit),
        state_mutator=mutator,
    )

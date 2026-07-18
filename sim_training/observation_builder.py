from __future__ import annotations

from policy_contract.observation_contract import OBSERVATION_SIZE, build_sim_observation


def contract_obs_builder_class():
    """Create the upstream component lazily so browser inference never imports Rust."""
    from haxballgym.obs import ObsBuilder

    class ContractObsBuilder(ObsBuilder):
        def obs_dim(self, n_players: int) -> int:
            if n_players != 2:
                raise ValueError("policy contract v1 supports exactly 1v1")
            return OBSERVATION_SIZE

        def build_obs(self, state):
            return build_sim_observation(state)

    return ContractObsBuilder

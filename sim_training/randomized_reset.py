from __future__ import annotations

import numpy as np

RESET_DISTRIBUTION_VERSION = "realistic-randomized-kickoff-v4"
RESET_DISTRIBUTION = {
    "player_y": [-60.0, 60.0],
    "player_x_offset": [-12.0, 12.0],
    "ball_y": [-55.0, 55.0],
    "ball_x": [-10.0, 10.0],
    "player_velocity": [-0.08, 0.08],
    "ball_velocity": [-0.35, 0.35],
    "kickoff_delay_decisions": [0, 2],
    "closest_team": ["red", "blue"],
}


class RandomizedKickoffMutator:
    """Seeded, batched, physically separated states near classic HaxBall kickoffs."""

    def __init__(self, seed: int, *, action_repeat: int = 8) -> None:
        self.seed = int(seed)
        self.action_repeat = int(action_repeat)
        self.rng = np.random.default_rng(seed)
        self.closest_team = np.empty(0, dtype=np.int8)  # browser IDs: Red=1, Blue=2
        self.kickoff_delay = np.empty(0, dtype=np.int8)
        self.reset_generation = np.empty(0, dtype=np.int64)

    def _sample(self, n: int, action_repeat: int = 8):
        rng = self.rng
        delay = rng.integers(0, 3, size=n, dtype=np.int8)
        ball_pos = np.stack(
            (rng.uniform(-10.0, 10.0, n), rng.uniform(-55.0, 55.0, n)), axis=-1)
        player_y = rng.uniform(-60.0, 60.0, (n, 2))
        spawn_offset = rng.uniform(-12.0, 12.0, (n, 2))
        player_pos = np.empty((n, 2, 2), dtype=np.float64)
        player_pos[:, 0, 0] = -277.5 + spawn_offset[:, 0]
        player_pos[:, 1, 0] = 277.5 + spawn_offset[:, 1]
        player_pos[..., 1] = player_y
        player_vel = rng.uniform(-0.08, 0.08, (n, 2, 2))
        ball_vel = rng.uniform(-0.35, 0.35, (n, 2))

        # Represent the optional short no-input delay by advancing the sampled
        # low-speed state. Players and ball are far enough apart that no
        # collision is skipped, and clipping keeps the state stadium-valid.
        elapsed = delay.astype(np.float64)[:, None] * action_repeat
        ball_pos += ball_vel * elapsed
        player_pos += player_vel * elapsed[:, None, :]
        np.clip(ball_pos[:, 0], -18.0, 18.0, out=ball_pos[:, 0])
        np.clip(ball_pos[:, 1], -65.0, 65.0, out=ball_pos[:, 1])
        np.clip(player_pos[..., 0], -305.0, 305.0, out=player_pos[..., 0])
        np.clip(player_pos[..., 1], -68.0, 68.0, out=player_pos[..., 1])

        distance = np.linalg.norm(player_pos - ball_pos[:, None, :], axis=-1)
        # Symmetric independent spawn sampling makes the identity of the
        # closest team a reproducible random kickoff attribute.
        closest = (np.argmin(distance, axis=1) + 1).astype(np.int8)
        return ball_pos, ball_vel, player_pos, player_vel, closest, delay

    def reset_all(self, engine):
        sampled = self._sample(engine.n_envs, self.action_repeat)
        bp, bv, pp, pv, self.closest_team, self.kickoff_delay = sampled
        self.reset_generation = np.ones(engine.n_envs, dtype=np.int64)
        return engine.set_state(bp, bv, pp, pv)

    def reset_mask(self, engine, mask: np.ndarray) -> None:
        state = engine.snapshot()
        bp, bv = state.ball_pos.copy(), state.ball_vel.copy()
        pp, pv = state.player_pos.copy(), state.player_vel.copy()
        steps = state.steps.copy()
        indices = np.flatnonzero(mask)
        if indices.size:
            nbp, nbv, npp, npv, closest, delay = self._sample(
                indices.size, self.action_repeat)
            bp[indices], bv[indices], pp[indices], pv[indices] = nbp, nbv, npp, npv
            steps[indices] = 0
            self.closest_team[indices] = closest
            self.kickoff_delay[indices] = delay
            self.reset_generation[indices] += 1
        engine.set_state(bp, bv, pp, pv, steps)

    def kickoff_labels(self) -> np.ndarray:
        teams = np.where(self.closest_team == 1, "red_closest", "blue_closest")
        return np.char.add(np.char.add(teams, "_delay_"), self.kickoff_delay.astype(str))

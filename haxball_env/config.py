"""Tunable parameters for the small 1v1 simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RewardConfig:
    goal: float = 1.0
    concede: float = -1.0
    ball_progress: float = 0.10
    ball_regress: float = 0.10
    touch: float = 0.002
    inactivity: float = -0.0001


@dataclass(frozen=True, slots=True)
class EnvConfig:
    # These values are deliberately approximate and are not calibrated to HaxBall.
    field_width: float = 20.0
    field_height: float = 12.0
    goal_size: float = 4.0
    player_radius: float = 0.60
    ball_radius: float = 0.30
    player_acceleration: float = 18.0
    player_max_speed: float = 6.0
    player_damping: float = 0.90
    ball_damping: float = 0.995
    ball_max_speed: float = 16.0
    # Kept under its original name for compatibility; this is ball-wall restitution.
    wall_restitution: float = 0.80
    player_wall_restitution: float = 0.0
    player_collision_restitution: float = 0.0
    ball_restitution: float = 0.85
    kick_range: float = 1.45
    kick_impulse: float = 9.0
    kick_cooldown: float = 0.50
    physics_timestep: float = 1.0 / 60.0
    action_repeat: int = 4
    score_limit: int = 3
    episode_time_limit: float = 120.0
    spawn_jitter: float = 0.10
    render_width: int = 960
    render_height: int = 640
    render_fps: int = 60
    render_margin: int = 64
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.goal_size <= 0 or self.goal_size >= self.field_height:
            raise ValueError("goal_size must be between zero and field_height")
        if self.action_repeat < 1 or self.score_limit < 1:
            raise ValueError("action_repeat and score_limit must be positive")
        if self.physics_timestep <= 0 or self.episode_time_limit <= 0:
            raise ValueError("time values must be positive")
        if self.render_width < 320 or self.render_height < 240:
            raise ValueError("render dimensions are too small")
        if self.render_fps < 1 or self.render_margin < 0:
            raise ValueError("render_fps must be positive and render_margin non-negative")

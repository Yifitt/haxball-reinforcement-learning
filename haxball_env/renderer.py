"""Small optional Pygame renderer; Pygame is imported only when instantiated."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .config import EnvConfig
from .physics import GameState


class PygameRenderer:
    def __init__(self, config: EnvConfig) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise gym.error.DependencyNotInstalled(
                'Human rendering requires pygame; install with: pip install -e ".[render]"'
            ) from exc

        self.pygame: Any = pygame
        self.config = config
        pygame.init()
        pygame.display.set_caption("haxball_rl validation")
        self.screen = pygame.display.set_mode((config.render_width, config.render_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.closed = False

        available_width = config.render_width - 2 * config.render_margin - 80
        available_height = config.render_height - 2 * config.render_margin - 70
        self.scale = min(
            available_width / config.field_width,
            available_height / config.field_height,
        )
        self.field_pixel_width = int(round(config.field_width * self.scale))
        self.field_pixel_height = int(round(config.field_height * self.scale))
        self.field_left = (config.render_width - self.field_pixel_width) // 2
        self.field_top = (config.render_height - self.field_pixel_height) // 2 + 24

    def _point(self, position: np.ndarray) -> tuple[int, int]:
        x = self.field_left + (float(position[0]) + self.config.field_width / 2) * self.scale
        y = self.field_top + (self.config.field_height / 2 - float(position[1])) * self.scale
        return int(round(x)), int(round(y))

    def _draw_field(self) -> None:
        pygame = self.pygame
        field = pygame.Rect(
            self.field_left,
            self.field_top,
            self.field_pixel_width,
            self.field_pixel_height,
        )
        pygame.draw.rect(self.screen, (37, 126, 68), field)
        line_color = (225, 235, 225)
        center_y = self.field_top + self.field_pixel_height // 2
        goal_half = int(round(self.config.goal_size * self.scale / 2))
        goal_depth = max(24, int(round(self.scale * 0.7)))

        pygame.draw.line(self.screen, line_color, field.topleft, field.topright, 3)
        pygame.draw.line(self.screen, line_color, field.bottomleft, field.bottomright, 3)
        for x in (field.left, field.right):
            pygame.draw.line(self.screen, line_color, (x, field.top), (x, center_y - goal_half), 3)
            pygame.draw.line(self.screen, line_color, (x, center_y + goal_half), (x, field.bottom), 3)
        pygame.draw.line(self.screen, (160, 205, 170), (field.centerx, field.top), (field.centerx, field.bottom), 2)
        pygame.draw.circle(self.screen, (160, 205, 170), field.center, int(self.scale * 1.5), 2)

        left_goal = pygame.Rect(field.left - goal_depth, center_y - goal_half, goal_depth, 2 * goal_half)
        right_goal = pygame.Rect(field.right, center_y - goal_half, goal_depth, 2 * goal_half)
        pygame.draw.rect(self.screen, line_color, left_goal, 3)
        pygame.draw.rect(self.screen, line_color, right_goal, 3)

    def _draw_text(
        self,
        scores: np.ndarray,
        remaining_time: float,
        controlled_side: int,
        state: GameState,
        episode_done: bool,
    ) -> None:
        status = "FINISHED - reset required" if episode_done else "ACTIVE"
        ready = state.kick_cooldowns[controlled_side] <= 0.0
        cooldown = "ready" if ready else f"{state.kick_cooldowns[controlled_side]:.2f}s"
        title = self.font.render(
            f"Left {scores[0]}  -  {scores[1]} Right     {max(0.0, remaining_time):05.1f}s",
            True,
            (240, 240, 240),
        )
        details = self.small_font.render(
            f"Controlled: {'left' if controlled_side == 0 else 'right'}   Kick: {cooldown}   {status}",
            True,
            (230, 230, 230),
        )
        self.screen.blit(title, title.get_rect(center=(self.config.render_width // 2, 22)))
        self.screen.blit(
            details,
            details.get_rect(center=(self.config.render_width // 2, self.config.render_height - 18)),
        )

    def render(
        self,
        *,
        state: GameState,
        scores: np.ndarray,
        remaining_time: float,
        controlled_side: int,
        episode_done: bool,
    ) -> bool:
        if self.closed:
            return False
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                return False

        self.screen.fill((23, 31, 38))
        self._draw_field()
        colors = [(55, 160, 255), (242, 83, 83)]
        for side in (0, 1):
            center = self._point(state.player_positions[side])
            radius = max(2, int(round(self.config.player_radius * self.scale)))
            pygame.draw.circle(self.screen, colors[side], center, radius)
            if side == controlled_side:
                pygame.draw.circle(self.screen, (255, 225, 80), center, radius + 4, 2)
        pygame.draw.circle(
            self.screen,
            (245, 245, 245),
            self._point(state.ball_position),
            max(2, int(round(self.config.ball_radius * self.scale))),
        )
        self._draw_text(scores, remaining_time, controlled_side, state, episode_done)
        pygame.display.flip()
        self.clock.tick(self.config.render_fps)
        return True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
        self.pygame.display.quit()
        self.pygame.quit()

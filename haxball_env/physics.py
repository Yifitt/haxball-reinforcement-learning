"""Deterministic NumPy physics for the first environment milestone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import EnvConfig

Vector = NDArray[np.float64]


@dataclass(slots=True)
class GameState:
    player_positions: NDArray[np.float64]
    player_velocities: NDArray[np.float64]
    ball_position: Vector
    ball_velocity: Vector
    kick_cooldowns: Vector

    def copy(self) -> "GameState":
        return GameState(
            self.player_positions.copy(),
            self.player_velocities.copy(),
            self.ball_position.copy(),
            self.ball_velocity.copy(),
            self.kick_cooldowns.copy(),
        )


@dataclass(frozen=True, slots=True)
class PhysicsResult:
    goal: int | None
    touches: tuple[bool, bool]
    kicks: tuple[bool, bool]


def limit_speed(velocity: Vector, maximum: float) -> None:
    speed = float(np.linalg.norm(velocity))
    if speed > maximum:
        velocity *= maximum / speed


def accelerate_player(velocity: Vector, direction: Vector, config: EnvConfig) -> None:
    velocity *= config.player_damping
    velocity += direction * config.player_acceleration * config.physics_timestep
    limit_speed(velocity, config.player_max_speed)


def apply_ball_friction(velocity: Vector, config: EnvConfig) -> None:
    velocity *= config.ball_damping
    if float(np.dot(velocity, velocity)) < 1e-12:
        velocity.fill(0.0)


def collide_with_walls(
    position: Vector,
    velocity: Vector,
    radius: float,
    config: EnvConfig,
    *,
    can_score: bool = False,
    restitution: float | None = None,
) -> int | None:
    """Keep a body in bounds, returning the scoring side for a ball goal."""
    response = config.wall_restitution if restitution is None else restitution
    half_w = config.field_width / 2.0
    half_h = config.field_height / 2.0
    y_limit = half_h - radius

    if position[1] > y_limit:
        position[1] = y_limit
        if velocity[1] > 0.0:
            velocity[1] = -velocity[1] * response
    elif position[1] < -y_limit:
        position[1] = -y_limit
        if velocity[1] < 0.0:
            velocity[1] = -velocity[1] * response

    in_goal_mouth = abs(float(position[1])) <= config.goal_size / 2.0
    if can_score and in_goal_mouth:
        if position[0] > half_w:
            return 0
        if position[0] < -half_w:
            return 1
        # The x wall is open here; allow the ball to travel across the goal line.
        return None

    x_limit = half_w - radius
    if position[0] > x_limit:
        position[0] = x_limit
        if velocity[0] > 0.0:
            velocity[0] = -velocity[0] * response
    elif position[0] < -x_limit:
        position[0] = -x_limit
        if velocity[0] < 0.0:
            velocity[0] = -velocity[0] * response
    return None


def _translation_limit(position: Vector, direction: Vector, lower: Vector, upper: Vector) -> float:
    """Maximum non-negative distance position can move along a unit direction."""
    limit = np.inf
    for axis in range(2):
        if direction[axis] > 1e-12:
            limit = min(limit, (upper[axis] - position[axis]) / direction[axis])
        elif direction[axis] < -1e-12:
            limit = min(limit, (lower[axis] - position[axis]) / direction[axis])
    return max(0.0, float(limit))


def _nearest_separated_position(
    position: Vector,
    fixed: Vector,
    minimum: float,
    lower: Vector,
    upper: Vector,
) -> Vector | None:
    """Find the nearest in-bounds point at least ``minimum`` from ``fixed``."""
    offset = position - fixed
    distance = float(np.linalg.norm(offset))
    preferred = offset / distance if distance > 1e-12 else np.array([1.0, 0.0])
    candidates: list[Vector] = []

    radial = fixed + preferred * minimum
    if np.all(radial >= lower - 1e-12) and np.all(radial <= upper + 1e-12):
        candidates.append(np.clip(radial, lower, upper))

    # If the radial correction crosses a wall, the closest feasible point lies
    # where the contact circle intersects one of the rectangular boundaries.
    for axis in range(2):
        other = 1 - axis
        for boundary in (lower[axis], upper[axis]):
            axis_offset = float(boundary - fixed[axis])
            remainder = minimum * minimum - axis_offset * axis_offset
            if remainder < -1e-12:
                continue
            other_offset = float(np.sqrt(max(0.0, remainder)))
            for sign in (-1.0, 1.0):
                candidate = fixed.copy()
                candidate[axis] = boundary
                candidate[other] += sign * other_offset
                if np.all(candidate >= lower - 1e-12) and np.all(candidate <= upper + 1e-12):
                    candidates.append(np.clip(candidate, lower, upper))

    if not candidates:
        # This is only reachable for unusual custom geometries where the contact
        # circle misses every wall segment. A feasible corner is still useful.
        for x in (lower[0], upper[0]):
            for y in (lower[1], upper[1]):
                candidate = np.array([x, y])
                if np.linalg.norm(candidate - fixed) >= minimum:
                    candidates.append(candidate)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: float(np.dot(candidate - position, candidate - position)),
    )


def resolve_player_collision(state: GameState, config: EnvConfig) -> None:
    lower = np.array(
        [-config.field_width / 2 + config.player_radius, -config.field_height / 2 + config.player_radius]
    )
    upper = -lower
    np.clip(state.player_positions, lower, upper, out=state.player_positions)

    delta = state.player_positions[1] - state.player_positions[0]
    distance = float(np.linalg.norm(delta))
    minimum = 2.0 * config.player_radius
    if distance >= minimum:
        return
    if distance > 1e-12:
        normal = delta / distance
    else:
        midpoint = (state.player_positions[0] + state.player_positions[1]) / 2.0
        normal = -np.sign(midpoint)
        if not np.any(normal):
            normal[0] = 1.0
        normal /= np.linalg.norm(normal)

    penetration = minimum - distance
    directions = (-normal, normal)
    capacities = [
        _translation_limit(state.player_positions[player], directions[player], lower, upper)
        for player in range(2)
    ]
    movement = [min(penetration / 2.0, capacity) for capacity in capacities]
    remaining = penetration - sum(movement)
    for player in range(2):
        extra = min(remaining, capacities[player] - movement[player])
        movement[player] += extra
        remaining -= extra
    for player in range(2):
        state.player_positions[player] += directions[player] * movement[player]

    # Players wedged against different walls of the same corner may exhaust both
    # normal-direction capacities. In that case, move one tangentially to the
    # nearest feasible contact point instead of leaving a persistent overlap.
    corrected_delta = state.player_positions[1] - state.player_positions[0]
    if float(np.linalg.norm(corrected_delta)) < minimum - 1e-12:
        fallback_candidates: list[tuple[float, int, Vector]] = []
        for player in range(2):
            fixed_player = 1 - player
            candidate = _nearest_separated_position(
                state.player_positions[player],
                state.player_positions[fixed_player],
                minimum,
                lower,
                upper,
            )
            if candidate is not None:
                displacement = candidate - state.player_positions[player]
                fallback_candidates.append(
                    (float(np.dot(displacement, displacement)), player, candidate)
                )
        if fallback_candidates:
            _, player, candidate = min(
                fallback_candidates, key=lambda item: (item[0], item[1])
            )
            state.player_positions[player] = candidate
            corrected_delta = state.player_positions[1] - state.player_positions[0]

    corrected_distance = float(np.linalg.norm(corrected_delta))
    collision_normal = (
        corrected_delta / corrected_distance if corrected_distance > 1e-12 else normal
    )
    relative = float(
        np.dot(state.player_velocities[1] - state.player_velocities[0], collision_normal)
    )
    if relative < 0.0:
        impulse = -(1.0 + config.player_collision_restitution) * relative / 2.0
        state.player_velocities[0] -= impulse * collision_normal
        state.player_velocities[1] += impulse * collision_normal
        limit_speed(state.player_velocities[0], config.player_max_speed)
        limit_speed(state.player_velocities[1], config.player_max_speed)


def resolve_player_ball_collision(state: GameState, player: int, config: EnvConfig) -> bool:
    delta = state.ball_position - state.player_positions[player]
    distance = float(np.linalg.norm(delta))
    minimum = config.player_radius + config.ball_radius
    if distance >= minimum:
        return False

    if distance > 1e-12:
        normal = delta / distance
    else:
        fallback = 1.0 if player == 0 else -1.0
        normal = np.array([fallback, 0.0], dtype=np.float64)
    state.ball_position += normal * (minimum - distance + 1e-9)

    relative_speed = float(np.dot(state.ball_velocity - state.player_velocities[player], normal))
    if relative_speed < 0.0:
        player_mass = 2.0
        impulse = -(1.0 + config.ball_restitution) * relative_speed / (1.0 + 1.0 / player_mass)
        state.ball_velocity += impulse * normal
        state.player_velocities[player] -= (impulse / player_mass) * normal
        limit_speed(state.ball_velocity, config.ball_max_speed)
    return True


def try_kick(state: GameState, player: int, config: EnvConfig) -> bool:
    if state.kick_cooldowns[player] > 0.0:
        return False
    delta = state.ball_position - state.player_positions[player]
    distance = float(np.linalg.norm(delta))
    if distance > config.kick_range:
        return False
    if distance > 1e-12:
        direction = delta / distance
    else:
        direction = np.array([1.0 if player == 0 else -1.0, 0.0])
    state.ball_velocity += direction * config.kick_impulse
    limit_speed(state.ball_velocity, config.ball_max_speed)
    state.kick_cooldowns[player] = config.kick_cooldown
    return True


def simulate_substep(
    state: GameState,
    directions: NDArray[np.float64],
    kick_requested: tuple[bool, bool],
    config: EnvConfig,
) -> PhysicsResult:
    dt = config.physics_timestep
    state.kick_cooldowns[:] = np.maximum(0.0, state.kick_cooldowns - dt)

    for player in range(2):
        accelerate_player(state.player_velocities[player], directions[player], config)
        state.player_positions[player] += state.player_velocities[player] * dt
        collide_with_walls(
            state.player_positions[player],
            state.player_velocities[player],
            config.player_radius,
            config,
            restitution=config.player_wall_restitution,
        )
    resolve_player_collision(state, config)
    for player in range(2):
        collide_with_walls(
            state.player_positions[player],
            state.player_velocities[player],
            config.player_radius,
            config,
            restitution=config.player_wall_restitution,
        )

    kicks = tuple(
        bool(kick_requested[player] and try_kick(state, player, config))
        for player in range(2)
    )
    apply_ball_friction(state.ball_velocity, config)
    state.ball_position += state.ball_velocity * dt
    goal = collide_with_walls(
        state.ball_position,
        state.ball_velocity,
        config.ball_radius,
        config,
        can_score=True,
    )

    touches = [False, False]
    if goal is None:
        for player in range(2):
            touches[player] = resolve_player_ball_collision(state, player, config)
        goal = collide_with_walls(
            state.ball_position,
            state.ball_velocity,
            config.ball_radius,
            config,
            can_score=True,
        )
    return PhysicsResult(goal, tuple(touches), kicks)

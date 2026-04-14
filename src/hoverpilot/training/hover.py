from dataclasses import dataclass

from hoverpilot.rflink.models import FlightAxisState


@dataclass(slots=True)
class RewardConfig:
    target_x_m: float = 0.0
    target_y_m: float = 0.0
    target_altitude_agl_m: float = 1.5
    max_abs_x_m: float = 8.0
    max_abs_y_m: float = 8.0
    min_altitude_agl_m: float = 0.2
    max_altitude_agl_m: float = 6.0
    position_error_weight: float = 0.15
    altitude_error_weight: float = 0.2
    attitude_error_weight: float = 0.01
    boundary_proximity_weight: float = 0.75
    terminal_failure_reward: float = -25.0
    proximity_penalty_margin_ratio: float = 0.25
    controller_active_threshold: float = 0.5


@dataclass(slots=True)
class TerminationResult:
    terminated: bool
    termination_reason: str | None = None


@dataclass(slots=True)
class RewardBreakdown:
    reward: float
    position_penalty: float
    altitude_penalty: float
    attitude_penalty: float
    boundary_proximity_penalty: float
    terminal_penalty: float
    terminated: bool
    termination_reason: str | None



def compute_termination(
    state: FlightAxisState,
    config: RewardConfig,
) -> TerminationResult:
    x_error = state.m_aircraftPositionX_MTR - config.target_x_m
    if abs(x_error) > config.max_abs_x_m:
        return TerminationResult(True, "out_of_bounds_x")

    y_error = state.m_aircraftPositionY_MTR - config.target_y_m
    if abs(y_error) > config.max_abs_y_m:
        return TerminationResult(True, "out_of_bounds_y")

    altitude_agl_m = state.m_altitudeAGL_MTR
    if altitude_agl_m < config.min_altitude_agl_m:
        return TerminationResult(True, "altitude_too_low")
    if altitude_agl_m > config.max_altitude_agl_m:
        return TerminationResult(True, "altitude_too_high")

    if state.m_hasLostComponents > 0.0:
        return TerminationResult(True, "lost_components")

    if state.m_flightAxisControllerIsActive < config.controller_active_threshold:
        return TerminationResult(True, "controller_inactive")

    return TerminationResult(False, None)



def compute_reward(
    state: FlightAxisState,
    config: RewardConfig,
) -> RewardBreakdown:
    termination = compute_termination(state, config)

    x_error = state.m_aircraftPositionX_MTR - config.target_x_m
    y_error = state.m_aircraftPositionY_MTR - config.target_y_m
    altitude_error = state.m_altitudeAGL_MTR - config.target_altitude_agl_m

    position_penalty = config.position_error_weight * ((x_error * x_error) + (y_error * y_error))
    altitude_penalty = config.altitude_error_weight * abs(altitude_error)
    attitude_penalty = config.attitude_error_weight * (
        abs(state.m_roll_DEG) + abs(state.m_inclination_DEG)
    )
    boundary_proximity_penalty = _compute_boundary_proximity_penalty(state, config)
    terminal_penalty = config.terminal_failure_reward if termination.terminated else 0.0

    reward = -(
        position_penalty
        + altitude_penalty
        + attitude_penalty
        + boundary_proximity_penalty
    ) + terminal_penalty

    return RewardBreakdown(
        reward=reward,
        position_penalty=position_penalty,
        altitude_penalty=altitude_penalty,
        attitude_penalty=attitude_penalty,
        boundary_proximity_penalty=boundary_proximity_penalty,
        terminal_penalty=terminal_penalty,
        terminated=termination.terminated,
        termination_reason=termination.termination_reason,
    )



def _compute_boundary_proximity_penalty(
    state: FlightAxisState,
    config: RewardConfig,
) -> float:
    x_penalty = _boundary_axis_penalty(
        distance_from_center=abs(state.m_aircraftPositionX_MTR - config.target_x_m),
        limit=config.max_abs_x_m,
        margin_ratio=config.proximity_penalty_margin_ratio,
    )
    y_penalty = _boundary_axis_penalty(
        distance_from_center=abs(state.m_aircraftPositionY_MTR - config.target_y_m),
        limit=config.max_abs_y_m,
        margin_ratio=config.proximity_penalty_margin_ratio,
    )

    low_altitude_gap = state.m_altitudeAGL_MTR - config.min_altitude_agl_m
    low_altitude_penalty = _boundary_edge_penalty(
        distance_to_edge=low_altitude_gap,
        limit=max(config.max_altitude_agl_m - config.min_altitude_agl_m, 1.0e-6),
        margin_ratio=config.proximity_penalty_margin_ratio,
    )
    high_altitude_gap = config.max_altitude_agl_m - state.m_altitudeAGL_MTR
    high_altitude_penalty = _boundary_edge_penalty(
        distance_to_edge=high_altitude_gap,
        limit=max(config.max_altitude_agl_m - config.min_altitude_agl_m, 1.0e-6),
        margin_ratio=config.proximity_penalty_margin_ratio,
    )

    return config.boundary_proximity_weight * (
        x_penalty + y_penalty + low_altitude_penalty + high_altitude_penalty
    )



def _boundary_axis_penalty(
    distance_from_center: float,
    limit: float,
    margin_ratio: float,
) -> float:
    return _boundary_edge_penalty(
        distance_to_edge=limit - distance_from_center,
        limit=limit,
        margin_ratio=margin_ratio,
    )



def _boundary_edge_penalty(
    distance_to_edge: float,
    limit: float,
    margin_ratio: float,
) -> float:
    if limit <= 0.0:
        return 0.0

    margin = max(limit * margin_ratio, 1.0e-6)
    if distance_to_edge >= margin:
        return 0.0
    if distance_to_edge <= 0.0:
        return 1.0

    normalized = 1.0 - (distance_to_edge / margin)
    return normalized * normalized

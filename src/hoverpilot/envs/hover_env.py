import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hoverpilot.rflink.client import RFLinkClient
from hoverpilot.rflink.models import FlightAxisState, RFControlAction
from hoverpilot.rflink.protocol import state_looks_uninitialized
from hoverpilot.training.hover import RewardConfig, TerminationResult, compute_reward, compute_termination

OBSERVATION_SIZE = 12
TRAINER_RESET_REASONS = {
    "trainer_reset",
    "trainer_reset_button",
    "crash_reset",
    "controller_reactivated",
    "engine_restarted",
    "trainer_repositioned",
}
BOOL_FIELD_THRESHOLD = 0.5
DEFAULT_RESET_WAIT_SECONDS = 8.0
DEFAULT_RESET_POLL_INTERVAL_SECONDS = 0.05


@dataclass(slots=True)
class EpisodeLifecycleResult:
    ready: bool
    started: bool
    terminated: bool
    truncated: bool
    reason: str | None = None


def state_to_observation(state: FlightAxisState) -> np.ndarray:
    """Build a compact hover-oriented observation vector.

    These fields capture the variables most directly tied to stationary hover:
    planar position, altitude, attitude, world-frame velocity, and body rates.
    We intentionally omit less essential telemetry to keep the first RL interface
    compact and easier to tune.
    """

    return np.asarray(
        [
            state.m_aircraftPositionX_MTR,
            state.m_aircraftPositionY_MTR,
            state.m_altitudeAGL_MTR,
            state.m_roll_DEG,
            state.m_inclination_DEG,
            state.m_azimuth_DEG,
            state.m_velocityWorldU_MPS,
            state.m_velocityWorldV_MPS,
            state.m_velocityWorldW_MPS,
            state.m_pitchRate_DEGpSEC,
            state.m_rollRate_DEGpSEC,
            state.m_yawRate_DEGpSEC,
        ],
        dtype=np.float32,
    )



def gym_action_to_rf_action(action: np.ndarray | list[float] | tuple[float, ...]) -> RFControlAction:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape != (4,):
        raise ValueError("action must have shape (4,) for [aileron, elevator, throttle, rudder]")

    clipped = np.clip(
        action_array,
        np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    return RFControlAction(
        aileron=float(clipped[0]),
        elevator=float(clipped[1]),
        throttle=float(clipped[2]),
        rudder=float(clipped[3]),
    )


class HoverPilotHoverEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str,
        port: int,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int | None = None,
        sleep_interval_s: float = 0.0,
        anchor_target_to_reset_state: bool = True,
        reset_button_threshold: float = 0.5,
        lost_components_threshold: float = 0.5,
        physics_time_reset_tolerance_s: float = 1.0e-3,
        max_reset_wait_seconds: float = DEFAULT_RESET_WAIT_SECONDS,
        reset_poll_interval_seconds: float = DEFAULT_RESET_POLL_INTERVAL_SECONDS,
        ready_controller_active_threshold: float | None = None,
        ready_running_threshold: float | None = None,
        ready_locked_threshold: float = BOOL_FIELD_THRESHOLD,
        require_nonzero_physics_time_for_ready: bool = True,
        allow_ground_contact_at_ready: bool = True,
        allow_reset_like_stationary_start: bool = False,
        minimum_start_altitude_agl_m: float = 0.25,
        start_groundspeed_threshold_mps: float = 1.0,
        start_airspeed_threshold_mps: float = 1.5,
        start_body_rate_threshold_deg_s: float = 60.0,
        reposition_position_margin_ratio: float = 0.35,
        reposition_altitude_margin_ratio: float = 0.5,
        reposition_speed_threshold_mps: float = 0.5,
        reset_teleport_distance_m: float = 2.0,
        client_factory: Callable[[], RFLinkClient] | None = None,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.reward_config = RewardConfig() if reward_config is None else reward_config
        self.max_episode_steps = max_episode_steps
        self.sleep_interval_s = sleep_interval_s
        self.anchor_target_to_reset_state = anchor_target_to_reset_state
        self.reset_button_threshold = reset_button_threshold
        self.lost_components_threshold = lost_components_threshold
        self.physics_time_reset_tolerance_s = physics_time_reset_tolerance_s
        self.max_reset_wait_seconds = max_reset_wait_seconds
        self.reset_poll_interval_seconds = reset_poll_interval_seconds
        self.ready_controller_active_threshold = ready_controller_active_threshold
        self.ready_running_threshold = ready_running_threshold
        self.ready_locked_threshold = ready_locked_threshold
        self.require_nonzero_physics_time_for_ready = require_nonzero_physics_time_for_ready
        self.allow_ground_contact_at_ready = allow_ground_contact_at_ready
        self.allow_reset_like_stationary_start = allow_reset_like_stationary_start
        self.minimum_start_altitude_agl_m = minimum_start_altitude_agl_m
        self.start_groundspeed_threshold_mps = start_groundspeed_threshold_mps
        self.start_airspeed_threshold_mps = start_airspeed_threshold_mps
        self.start_body_rate_threshold_deg_s = start_body_rate_threshold_deg_s
        self.reposition_position_margin_ratio = reposition_position_margin_ratio
        self.reposition_altitude_margin_ratio = reposition_altitude_margin_ratio
        self.reposition_speed_threshold_mps = reposition_speed_threshold_mps
        self.reset_teleport_distance_m = reset_teleport_distance_m
        self._client_factory = (
            client_factory if client_factory is not None else lambda: RFLinkClient(self.host, self.port)
        )
        self._client: RFLinkClient | None = None
        self._episode_steps = 0
        self._last_state: FlightAxisState | None = None
        self._pending_episode_start: tuple[FlightAxisState, str] | None = None
        self._waiting_for_reset = False
        self._reset_wait_saw_lost_components = False
        self._reset_wait_saw_crash_signature = False
        self._episode_started = False
        self._ground_contact_started_at_s: float | None = None

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray(
                [
                    -500.0,
                    -500.0,
                    -10.0,
                    -180.0,
                    -180.0,
                    -360.0,
                    -200.0,
                    -200.0,
                    -200.0,
                    -720.0,
                    -720.0,
                    -720.0,
                ],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    500.0,
                    500.0,
                    200.0,
                    180.0,
                    180.0,
                    360.0,
                    200.0,
                    200.0,
                    200.0,
                    720.0,
                    720.0,
                    720.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        self._last_state = None
        self._pending_episode_start = None
        self._waiting_for_reset = False
        self._reset_wait_saw_lost_components = False
        self._reset_wait_saw_crash_signature = False
        self._episode_started = False
        self._ground_contact_started_at_s = None
        self.close()
        self._client = self._client_factory()
        self._client.connect()

        ready_action = self._safe_start_action()
        if options and "initial_action" in options:
            ready_action = gym_action_to_rf_action(options["initial_action"])

        state, episode_start_reason = self._wait_for_ready_state(ready_action)
        return self._start_episode_from_state(state, episode_start_reason=episode_start_reason)

    def step(self, action: np.ndarray):
        if self._client is None:
            raise RuntimeError("environment must be reset() before step()")

        if self.sleep_interval_s > 0.0:
            time.sleep(self.sleep_interval_s)

        rf_action = gym_action_to_rf_action(action)
        state = self._client.step(rf_action)
        ground_contact_duration_s = self._update_ground_contact_duration(state)
        reward_breakdown = compute_reward(
            state,
            self.reward_config,
            episode_started=self._episode_started,
            ground_contact_duration_s=ground_contact_duration_s,
        )
        termination = compute_termination(
            state,
            self.reward_config,
            episode_started=self._episode_started,
            ground_contact_duration_s=ground_contact_duration_s,
        )
        readiness = self.compute_episode_start_status(state)
        trainer_reset_reason = self._detect_trainer_reset(state)
        lifecycle = EpisodeLifecycleResult(
            ready=readiness.ready,
            started=self._episode_started,
            terminated=termination.terminated,
            truncated=False,
            reason=termination.termination_reason,
        )

        if trainer_reset_reason is not None:
            self._waiting_for_reset = True
            self._reset_wait_saw_lost_components = False
            self._reset_wait_saw_crash_signature = False
            self._episode_started = False
            self._pending_episode_start = (state, trainer_reset_reason)
            if self._can_start_episode_from_state(state, reset_reason=trainer_reset_reason):
                self._waiting_for_reset = False
            reward_breakdown = replace(
                reward_breakdown,
                reward=reward_breakdown.reward + self.reward_config.terminal_failure_reward,
                terminal_penalty=reward_breakdown.terminal_penalty + self.reward_config.terminal_failure_reward,
                terminated=True,
                termination_reason=trainer_reset_reason,
            )
            termination = TerminationResult(True, trainer_reset_reason)
            lifecycle = replace(lifecycle, terminated=True, started=False, reason=trainer_reset_reason)

        self._episode_steps += 1
        truncated = self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps
        lifecycle = replace(lifecycle, truncated=truncated)

        if termination.terminated and trainer_reset_reason is None:
            self._waiting_for_reset = True
            self._reset_wait_saw_lost_components = self._is_lost_components_active(state)
            self._reset_wait_saw_crash_signature = self._is_crash_signature(state)
            self._episode_started = False
            lifecycle = replace(lifecycle, started=False)

        observation = state_to_observation(state)
        info = self._build_info(
            state=state,
            reward_breakdown=reward_breakdown,
            truncated=truncated,
            reset=False,
            episode_start_reason=None,
            waiting_for_reset=self._waiting_for_reset,
            lifecycle=lifecycle,
            readiness=readiness,
            ground_contact_duration_s=ground_contact_duration_s,
        )
        self._last_state = state
        return (
            observation,
            float(reward_breakdown.reward),
            bool(termination.terminated),
            bool(truncated),
            info,
        )

    def poll_wait_for_next_episode(
        self,
        action: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> tuple[bool, np.ndarray, dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("environment must be reset() before waiting for the next episode")

        if self._pending_episode_start is not None:
            pending_state, reason = self._pending_episode_start
            if self._can_start_episode_from_state(pending_state, reset_reason=reason):
                self._pending_episode_start = None
                observation, info = self._start_episode_from_state(pending_state, episode_start_reason=reason)
                return True, observation, info

        wait_action = self._safe_start_action() if action is None else gym_action_to_rf_action(action)
        state = self._poll_state(wait_action, interval_s=self.reset_poll_interval_seconds)
        trainer_reset_reason = self._detect_trainer_reset(state)
        readiness = self.compute_episode_start_status(state)
        if trainer_reset_reason is not None:
            self._pending_episode_start = (state, trainer_reset_reason)

        if self._pending_episode_start is not None:
            _, reason = self._pending_episode_start
        else:
            reason = None

        if self._pending_episode_start is not None and self._can_start_episode_from_state(state, reset_reason=reason):
            self._pending_episode_start = None
            observation, info = self._start_episode_from_state(state, episode_start_reason=reason)
            return True, observation, info
        if self._is_lost_components_active(state):
            self._reset_wait_saw_lost_components = True
        if self._is_crash_signature(state):
            self._reset_wait_saw_crash_signature = True
        lifecycle = EpisodeLifecycleResult(
            ready=readiness.ready,
            started=False,
            terminated=False,
            truncated=False,
            reason=readiness.reason,
        )
        info = self._build_info(
            state=state,
            reward_breakdown=None,
            truncated=False,
            reset=False,
            episode_start_reason=None,
            waiting_for_reset=True,
            lifecycle=lifecycle,
            readiness=readiness,
            ground_contact_duration_s=0.0,
        )
        self._last_state = state
        return False, state_to_observation(state), info

    def wait_for_next_episode(
        self,
        action: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ):
        while True:
            started, observation, info = self.poll_wait_for_next_episode(action=action)
            if started:
                return observation, info

    def render(self):
        return None

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def compute_episode_start_status(self, state: FlightAxisState) -> EpisodeLifecycleResult:
        # These flags are used conservatively as operational readiness hints.
        # RealFlight Link does not guarantee perfect semantics for every trainer mode,
        # so controller/engine checks are configurable instead of hard-required by default.
        if state_looks_uninitialized(state):
            return EpisodeLifecycleResult(False, False, False, False, "uninitialized_state")
        if self.require_nonzero_physics_time_for_ready and state.m_currentPhysicsTime_SEC <= 0.0:
            return EpisodeLifecycleResult(False, False, False, False, "physics_time_not_started")
        if state.m_isLocked > self.ready_locked_threshold:
            return EpisodeLifecycleResult(False, False, False, False, "vehicle_locked")
        if self._is_lost_components_active(state):
            return EpisodeLifecycleResult(False, False, False, False, "lost_components")
        if state.m_currentAircraftStatus in self.reward_config.known_terminal_aircraft_status_codes:
            return EpisodeLifecycleResult(False, False, False, False, "aircraft_status_terminal")
        if (
            self.ready_controller_active_threshold is not None
            and state.m_flightAxisControllerIsActive < self.ready_controller_active_threshold
        ):
            return EpisodeLifecycleResult(False, False, False, False, "controller_inactive")
        if (
            self.ready_running_threshold is not None
            and state.m_anEngineIsRunning < self.ready_running_threshold
        ):
            return EpisodeLifecycleResult(False, False, False, False, "engine_stopped")
        if not self.allow_ground_contact_at_ready and self._is_touching_ground(state):
            return EpisodeLifecycleResult(False, False, False, False, "touching_ground")
        return EpisodeLifecycleResult(True, True, False, False, None)

    def _wait_for_ready_state(self, action: RFControlAction) -> tuple[FlightAxisState, str]:
        deadline = time.monotonic() + self.max_reset_wait_seconds
        last_state: FlightAxisState | None = None
        last_reason = "reset_timeout"
        startup_sync_required = False
        pending_start_reason = "reset_ready"
        while time.monotonic() <= deadline:
            state = self._poll_state(action, interval_s=self.reset_poll_interval_seconds)
            readiness = self.compute_episode_start_status(state)
            trainer_reset_reason = self._detect_trainer_reset(state)
            self._last_state = state
            last_state = state
            last_reason = readiness.reason or last_reason
            if not startup_sync_required and self._requires_reset_boundary_sync(state):
                startup_sync_required = True
                self._waiting_for_reset = True

            if trainer_reset_reason is not None:
                pending_start_reason = trainer_reset_reason
                startup_sync_required = False

            if self._can_start_episode_from_state(state, reset_reason=pending_start_reason) and not startup_sync_required:
                self._waiting_for_reset = False
                return state, pending_start_reason
        diagnostics = "none"
        if last_state is not None:
            diagnostics = self._format_readiness_diagnostics(last_state)
        raise TimeoutError(f"timed out waiting for ready episode state: {last_reason}; {diagnostics}")

    def _poll_state(self, action: RFControlAction, *, interval_s: float) -> FlightAxisState:
        if interval_s > 0.0:
            time.sleep(interval_s)
        return self._client.request_state(action)

    def _safe_start_action(self) -> RFControlAction:
        return RFControlAction.safe_idle()

    def _start_episode_from_state(self, state: FlightAxisState, episode_start_reason: str):
        self._episode_steps = 0
        self._pending_episode_start = None
        self._waiting_for_reset = False
        self._reset_wait_saw_lost_components = False
        self._reset_wait_saw_crash_signature = False
        self._episode_started = True
        self._ground_contact_started_at_s = None
        if self.anchor_target_to_reset_state:
            self.reward_config = replace(
                self.reward_config,
                target_x_m=state.m_aircraftPositionX_MTR,
                target_y_m=state.m_aircraftPositionY_MTR,
                target_altitude_agl_m=state.m_altitudeAGL_MTR,
            )
        self._last_state = state
        readiness = self.compute_episode_start_status(state)
        lifecycle = EpisodeLifecycleResult(
            ready=readiness.ready,
            started=True,
            terminated=False,
            truncated=False,
            reason=episode_start_reason,
        )
        observation = state_to_observation(state)
        info = self._build_info(
            state=state,
            reward_breakdown=None,
            truncated=False,
            reset=True,
            episode_start_reason=episode_start_reason,
            waiting_for_reset=False,
            lifecycle=lifecycle,
            readiness=readiness,
            ground_contact_duration_s=0.0,
        )
        return observation, info

    def _build_info(
        self,
        *,
        state: FlightAxisState,
        reward_breakdown,
        truncated: bool,
        reset: bool,
        episode_start_reason: str | None,
        waiting_for_reset: bool,
        lifecycle: EpisodeLifecycleResult,
        readiness: EpisodeLifecycleResult,
        ground_contact_duration_s: float,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "state_summary": state.summary(),
            "debug_state": {
                "x_m": state.m_aircraftPositionX_MTR,
                "y_m": state.m_aircraftPositionY_MTR,
                "altitude_agl_m": state.m_altitudeAGL_MTR,
                "controller_active": state.m_flightAxisControllerIsActive,
                "physics_time_s": state.m_currentPhysicsTime_SEC,
                "reset_button_pressed": state.m_resetButtonHasBeenPressed,
                "lost_components": state.m_hasLostComponents,
                "aircraft_status": state.m_currentAircraftStatus,
                "touching_ground": state.m_isTouchingGround,
                "engine_running": state.m_anEngineIsRunning,
                "vehicle_locked": state.m_isLocked,
                "ground_contact_duration_s": ground_contact_duration_s,
                "ready_controller_active_threshold": self.ready_controller_active_threshold,
                "ready_running_threshold": self.ready_running_threshold,
                "ready_locked_threshold": self.ready_locked_threshold,
                "allow_reset_like_stationary_start": self.allow_reset_like_stationary_start,
                "minimum_start_altitude_agl_m": self.minimum_start_altitude_agl_m,
                "start_groundspeed_threshold_mps": self.start_groundspeed_threshold_mps,
                "start_airspeed_threshold_mps": self.start_airspeed_threshold_mps,
                "start_body_rate_threshold_deg_s": self.start_body_rate_threshold_deg_s,
                "reposition_position_margin_ratio": self.reposition_position_margin_ratio,
                "reposition_altitude_margin_ratio": self.reposition_altitude_margin_ratio,
                "reposition_speed_threshold_mps": self.reposition_speed_threshold_mps,
                "reset_teleport_distance_m": self.reset_teleport_distance_m,
                "saw_crash_signature": self._reset_wait_saw_crash_signature,
            },
            "target_hover": {
                "x_m": self.reward_config.target_x_m,
                "y_m": self.reward_config.target_y_m,
                "altitude_agl_m": self.reward_config.target_altitude_agl_m,
            },
            "episode_step": self._episode_steps,
            "reset": reset,
            "truncated": truncated,
            "episode_start_reason": episode_start_reason,
            "waiting_for_reset": waiting_for_reset,
            "episode_lifecycle": asdict(lifecycle),
            "episode_readiness": asdict(readiness),
        }
        if reward_breakdown is not None:
            info["reward_breakdown"] = asdict(reward_breakdown)
            info["termination_reason"] = reward_breakdown.termination_reason
        return info

    def _detect_trainer_reset(self, state: FlightAxisState) -> str | None:
        if state.m_resetButtonHasBeenPressed >= self.reset_button_threshold:
            return "trainer_reset_button"

        previous_state = self._last_state
        if previous_state is None:
            return None

        if (
            state.m_currentPhysicsTime_SEC + self.physics_time_reset_tolerance_s
            < previous_state.m_currentPhysicsTime_SEC
        ):
            if self._reset_wait_saw_lost_components or self._is_lost_components_active(previous_state):
                return "crash_reset"
            return "trainer_reset"

        if (
            self._waiting_for_reset
            and self._reset_wait_saw_lost_components
            and self._is_lost_components_active(previous_state)
            and not self._is_lost_components_active(state)
        ):
            return "crash_reset"

        if (
            self._waiting_for_reset
            and self._reset_wait_saw_crash_signature
            and self._is_reset_recovery_signature(state)
        ):
            return "crash_reset"

        if self._waiting_for_reset and self._reactivated(previous_state, state, selector=self._is_controller_active):
            if self._reset_wait_saw_lost_components:
                return "crash_reset"
            return "controller_reactivated"

        if self._waiting_for_reset and self._reactivated(previous_state, state, selector=self._is_engine_running):
            if self._reset_wait_saw_lost_components:
                return "crash_reset"
            return "engine_restarted"

        if self._looks_like_reset_teleport(previous_state, state):
            return "trainer_repositioned"

        if self._waiting_for_reset and self._looks_like_repositioned_ready_state(state):
            return "trainer_repositioned"

        return None

    def _update_ground_contact_duration(self, state: FlightAxisState) -> float:
        if not self._episode_started or not self._is_touching_ground(state):
            self._ground_contact_started_at_s = None
            return 0.0
        if self._ground_contact_started_at_s is None:
            self._ground_contact_started_at_s = time.monotonic()
            return 0.0
        return time.monotonic() - self._ground_contact_started_at_s

    def _is_lost_components_active(self, state: FlightAxisState) -> bool:
        return state.m_hasLostComponents > self.lost_components_threshold

    def _is_touching_ground(self, state: FlightAxisState) -> bool:
        return state.m_isTouchingGround > self.reward_config.touching_ground_threshold


    def _is_crash_signature(self, state: FlightAxisState) -> bool:
        # Conservative crash-wait signature observed in the hover trainer:
        # lost components, engine stopped, and touching ground together.
        return (
            self._is_lost_components_active(state)
            and state.m_anEngineIsRunning <= self.reward_config.engine_running_threshold
            and self._is_touching_ground(state)
        )

    def _is_reset_recovery_signature(self, state: FlightAxisState) -> bool:
        # Conservative reset-ready signature observed after trainer recovery:
        # components restored, engine running, and no longer touching ground.
        return (
            not self._is_lost_components_active(state)
            and state.m_anEngineIsRunning >= self.reward_config.engine_running_threshold
            and not self._is_touching_ground(state)
            and self.compute_episode_start_status(state).ready
        )

    def _is_controller_active(self, state: FlightAxisState) -> bool:
        threshold = self.ready_controller_active_threshold
        if threshold is None:
            return False
        return state.m_flightAxisControllerIsActive >= threshold

    def _is_engine_running(self, state: FlightAxisState) -> bool:
        threshold = self.ready_running_threshold
        if threshold is None:
            return False
        return state.m_anEngineIsRunning >= threshold

    def _reactivated(
        self,
        previous_state: FlightAxisState,
        state: FlightAxisState,
        *,
        selector: Callable[[FlightAxisState], bool],
    ) -> bool:
        return not selector(previous_state) and selector(state)

    def _format_readiness_diagnostics(self, state: FlightAxisState) -> str:
        return (
            f"locked={state.m_isLocked:.1f} lost={state.m_hasLostComponents:.1f} "
            f"engine={state.m_anEngineIsRunning:.1f} ctrl={state.m_flightAxisControllerIsActive:.1f} "
            f"ground={state.m_isTouchingGround:.1f} status={state.m_currentAircraftStatus:.1f} "
            f"physics_t={state.m_currentPhysicsTime_SEC:.3f}"
        )

    def _looks_like_reset_teleport(self, previous_state: FlightAxisState, state: FlightAxisState) -> bool:
        # RealFlight Hover Trainer does not always expose a dedicated reset flag.
        # As a fallback, treat a sudden jump out of a crash-wait state as a
        # trainer-driven reposition rather than an out-of-bounds failure.
        if self._planar_distance(previous_state, state) < self.reset_teleport_distance_m:
            return False
        if not self.compute_episode_start_status(state).ready:
            return False
        if self._is_low_altitude_wait_state(previous_state) and not self._is_low_altitude_wait_state(state):
            return True
        return (
            self._is_reset_like_stationary_state(previous_state)
            and self._is_reset_like_stationary_state(state)
            and self._is_inactive_reset_state(state)
        )

    def _can_start_episode_from_state(self, state: FlightAxisState, reset_reason: str | None = None) -> bool:
        readiness = self.compute_episode_start_status(state)
        if not readiness.ready:
            return False
        if self._is_low_altitude_wait_state(state):
            return False
        if reset_reason in TRAINER_RESET_REASONS:
            return True
        if not self._is_start_stable_state(state):
            return False
        if self.allow_reset_like_stationary_start:
            return True
        return not (
            self._is_reset_like_stationary_state(state)
            and self._is_inactive_reset_state(state)
        )

    def _requires_reset_boundary_sync(self, state: FlightAxisState) -> bool:
        return (
            self._is_low_altitude_wait_state(state)
            or (
                self._is_inactive_reset_state(state)
                and not self._is_start_stable_state(state)
            )
            or (
                self._is_reset_like_stationary_state(state)
                and self._is_inactive_reset_state(state)
            )
        )

    def _is_low_altitude_wait_state(self, state: FlightAxisState) -> bool:
        # The hover trainer's pre-reset crash states often remain very close to the
        # ground. Treat low AGL itself as a strong "not started yet" signal even if
        # the aircraft still has some residual motion.
        return state.m_altitudeAGL_MTR <= self.minimum_start_altitude_agl_m

    def _is_start_stable_state(self, state: FlightAxisState) -> bool:
        # A valid episode start should not begin mid-crash or mid-fall. Require a
        # modestly stable state before declaring the episode active.
        return (
            state.m_groundspeed_MPS <= self.start_groundspeed_threshold_mps
            and state.m_airspeed_MPS <= self.start_airspeed_threshold_mps
            and abs(state.m_pitchRate_DEGpSEC) <= self.start_body_rate_threshold_deg_s
            and abs(state.m_rollRate_DEGpSEC) <= self.start_body_rate_threshold_deg_s
            and abs(state.m_yawRate_DEGpSEC) <= self.start_body_rate_threshold_deg_s
        )

    def _looks_like_repositioned_ready_state(self, state: FlightAxisState) -> bool:
        readiness = self.compute_episode_start_status(state)
        if not readiness.ready:
            return False
        if not self._is_reset_like_stationary_state(state):
            return False
        return self._is_inactive_reset_state(state) and self._is_near_hover_target(state)

    def _is_near_hover_target(self, state: FlightAxisState) -> bool:
        x_margin = max(0.5, self.reward_config.max_abs_x_m * self.reposition_position_margin_ratio)
        y_margin = max(0.5, self.reward_config.max_abs_y_m * self.reposition_position_margin_ratio)
        altitude_span = max(
            0.5,
            self.reward_config.max_altitude_agl_m - self.reward_config.min_altitude_agl_m,
        )
        altitude_margin = altitude_span * self.reposition_altitude_margin_ratio
        return (
            abs(state.m_aircraftPositionX_MTR - self.reward_config.target_x_m) <= x_margin
            and abs(state.m_aircraftPositionY_MTR - self.reward_config.target_y_m) <= y_margin
            and abs(state.m_altitudeAGL_MTR - self.reward_config.target_altitude_agl_m) <= altitude_margin
        )

    def _is_reset_like_stationary_state(self, state: FlightAxisState) -> bool:
        rate_threshold_deg_s = 5.0
        return (
            state.m_groundspeed_MPS <= self.reposition_speed_threshold_mps
            and state.m_airspeed_MPS <= self.reposition_speed_threshold_mps
            and abs(state.m_pitchRate_DEGpSEC) <= rate_threshold_deg_s
            and abs(state.m_rollRate_DEGpSEC) <= rate_threshold_deg_s
            and abs(state.m_yawRate_DEGpSEC) <= rate_threshold_deg_s
        )

    def _is_inactive_reset_state(self, state: FlightAxisState) -> bool:
        return (
            state.m_flightAxisControllerIsActive <= BOOL_FIELD_THRESHOLD
            and state.m_anEngineIsRunning <= BOOL_FIELD_THRESHOLD
        )

    def _planar_distance(self, previous_state: FlightAxisState, state: FlightAxisState) -> float:
        dx = state.m_aircraftPositionX_MTR - previous_state.m_aircraftPositionX_MTR
        dy = state.m_aircraftPositionY_MTR - previous_state.m_aircraftPositionY_MTR
        return float((dx * dx + dy * dy) ** 0.5)

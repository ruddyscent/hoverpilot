import unittest

from hoverpilot.rflink.models import FlightAxisState
from hoverpilot.training.hover import RewardConfig, compute_reward, compute_termination


class HoverTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(
            target_x_m=0.0,
            target_y_m=0.0,
            target_altitude_agl_m=1.5,
            max_abs_x_m=10.0,
            max_abs_y_m=12.0,
            min_altitude_agl_m=0.5,
            max_altitude_agl_m=5.0,
            boundary_proximity_weight=2.0,
            terminal_failure_reward=-50.0,
        )

    def _state(self, **overrides):
        state = FlightAxisState(
            m_aircraftPositionX_MTR=0.0,
            m_aircraftPositionY_MTR=0.0,
            m_altitudeAGL_MTR=1.5,
            m_roll_DEG=0.0,
            m_inclination_DEG=0.0,
            m_flightAxisControllerIsActive=1.0,
            m_hasLostComponents=0.0,
        )
        for name, value in overrides.items():
            setattr(state, name, value)
        return state

    def test_inside_boundary_state_is_not_terminated(self):
        result = compute_termination(self._state(), self.config)

        self.assertFalse(result.terminated)
        self.assertIsNone(result.termination_reason)

    def test_near_boundary_state_gets_higher_penalty(self):
        centered = compute_reward(self._state(m_aircraftPositionX_MTR=0.0), self.config)
        near_x_edge = compute_reward(self._state(m_aircraftPositionX_MTR=9.4), self.config)

        self.assertGreater(
            near_x_edge.boundary_proximity_penalty,
            centered.boundary_proximity_penalty,
        )
        self.assertLess(near_x_edge.reward, centered.reward)
        self.assertFalse(near_x_edge.terminated)

    def test_outside_boundary_state_is_terminated(self):
        result = compute_termination(self._state(m_aircraftPositionX_MTR=10.1), self.config)
        reward = compute_reward(self._state(m_aircraftPositionX_MTR=10.1), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "out_of_bounds_x")
        self.assertTrue(reward.terminated)
        self.assertEqual(reward.termination_reason, "out_of_bounds_x")
        self.assertLess(reward.reward, -40.0)

    def test_low_altitude_boundary_termination(self):
        result = compute_termination(self._state(m_altitudeAGL_MTR=0.2), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "altitude_too_low")

    def test_high_altitude_boundary_termination(self):
        result = compute_termination(self._state(m_altitudeAGL_MTR=5.2), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "altitude_too_high")

    def test_lost_components_termination_uses_state_flag(self):
        result = compute_termination(self._state(m_hasLostComponents=1.0), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "lost_components")

    def test_vehicle_locked_is_terminal(self):
        result = compute_termination(self._state(m_isLocked=1.0), self.config)

        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "vehicle_locked")

    def test_touching_ground_only_terminates_after_episode_start(self):
        before_start = compute_termination(
            self._state(m_isTouchingGround=1.0),
            RewardConfig(ground_contact_grace_seconds=0.0),
            episode_started=False,
        )
        after_start = compute_termination(
            self._state(m_isTouchingGround=1.0),
            RewardConfig(ground_contact_grace_seconds=0.0),
            episode_started=True,
            ground_contact_duration_s=0.0,
        )

        self.assertFalse(before_start.terminated)
        self.assertTrue(after_start.terminated)
        self.assertEqual(after_start.termination_reason, "touching_ground")

    def test_engine_stopped_only_terminates_when_enabled(self):
        disabled = compute_termination(self._state(m_anEngineIsRunning=0.0), self.config)
        enabled = compute_termination(
            self._state(m_anEngineIsRunning=0.0),
            RewardConfig(terminate_on_engine_stopped=True),
        )

        self.assertFalse(disabled.terminated)
        self.assertTrue(enabled.terminated)
        self.assertEqual(enabled.termination_reason, "engine_stopped")

    def test_controller_inactive_only_terminates_when_threshold_is_enabled(self):
        disabled = compute_termination(self._state(m_flightAxisControllerIsActive=0.0), self.config)
        enabled = compute_termination(
            self._state(m_flightAxisControllerIsActive=0.0),
            RewardConfig(controller_active_threshold=0.5),
        )

        self.assertFalse(disabled.terminated)
        self.assertTrue(enabled.terminated)
        self.assertEqual(enabled.termination_reason, "controller_inactive")


if __name__ == "__main__":
    unittest.main()

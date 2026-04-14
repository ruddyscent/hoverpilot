import unittest

import numpy as np

from hoverpilot.rflink.models import FlightAxisState
from hoverpilot.utils.logger import format_action, format_state, format_step_log


class LoggerTests(unittest.TestCase):
    def test_format_action_labels_controls(self):
        formatted = format_action(np.asarray([0.1, -0.2, 0.8, 0.3], dtype=np.float32))

        self.assertIn("TX", formatted)
        self.assertIn("ail=+0.100", formatted)
        self.assertIn("ele=-0.200", formatted)
        self.assertIn("thr=+0.800", formatted)
        self.assertIn("rud=+0.300", formatted)

    def test_format_state_includes_core_hover_fields(self):
        state = FlightAxisState(
            m_currentPhysicsTime_SEC=12.5,
            m_aircraftPositionX_MTR=1.2,
            m_aircraftPositionY_MTR=-3.4,
            m_altitudeAGL_MTR=2.1,
            m_airspeed_MPS=4.5,
            m_groundspeed_MPS=3.2,
            m_azimuth_DEG=90.0,
            m_inclination_DEG=1.5,
            m_roll_DEG=-2.5,
            m_pitchRate_DEGpSEC=0.5,
            m_rollRate_DEGpSEC=-1.0,
            m_yawRate_DEGpSEC=2.0,
        )

        formatted = format_state(state)

        self.assertIn("RX", formatted)
        self.assertIn("t=  12.500s", formatted)
        self.assertIn("agl= 2.100m", formatted)
        self.assertIn("air= 4.50m/s", formatted)
        self.assertIn("rates=", formatted)

    def test_format_step_log_combines_action_reward_and_reason(self):
        info = {
            "termination_reason": None,
            "state_summary": "time=1.000s pos=(0.00, 0.00)m alt_asl=0.00m gs=0.00m/s air=0.00m/s att=(az=0.0, inc=0.0, roll=0.0)deg rates=(p=0.0, r=0.0, y=0.0)deg/s quat=(0.000, 0.000, 0.000, 0.000)",
        }
        formatted = format_step_log(
            action=np.asarray([0.0, 0.0, 0.8, 0.0], dtype=np.float32),
            info=info,
            reward=-1.25,
            terminated=False,
            truncated=False,
        )

        self.assertIn("TX", formatted)
        self.assertIn("reward=-1.250", formatted)
        self.assertIn("terminated=False", formatted)
        self.assertIn("reason=None", formatted)
        self.assertIn("time=1.000s", formatted)


if __name__ == "__main__":
    unittest.main()

import unittest

from hoverpilot.rflink.models import RFControlAction


class RFControlActionTests(unittest.TestCase):
    def test_default_mapping_generates_12_channels(self):
        action = RFControlAction(
            throttle=0.55,
            aileron=0.0,
            elevator=0.0,
            rudder=0.0,
        )

        channel_values = action.to_channel_values()

        self.assertEqual(len(channel_values), 12)
        self.assertEqual(channel_values[:4], [0.5, 0.5, 0.55, 0.5])
        self.assertTrue(all(value == 0.0 for value in channel_values[4:]))

    def test_bidirectional_controls_are_remapped_to_zero_to_one(self):
        action = RFControlAction(
            throttle=1.0,
            aileron=-1.0,
            elevator=1.0,
            rudder=0.25,
        )

        channel_values = action.to_channel_values()

        self.assertEqual(channel_values[0], 0.0)
        self.assertEqual(channel_values[1], 1.0)
        self.assertEqual(channel_values[2], 1.0)
        self.assertAlmostEqual(channel_values[3], 0.625)

    def test_custom_channel_map_is_respected(self):
        action = RFControlAction(
            throttle=0.4,
            aileron=1.0,
            elevator=-1.0,
            rudder=0.0,
        )

        channel_values = action.to_channel_values(
            {
                "aileron": 3,
                "elevator": 2,
                "throttle": 0,
                "rudder": 1,
            }
        )

        self.assertEqual(channel_values[:4], [0.4, 0.5, 0.0, 1.0])

    def test_out_of_range_values_are_clamped(self):
        action = RFControlAction(
            throttle=3.0,
            aileron=-2.0,
            elevator=9.0,
            rudder=-0.5,
            channel_overrides={5: 3.0},
        )

        channel_values = action.to_channel_values()

        self.assertEqual(channel_values[0], 0.0)
        self.assertEqual(channel_values[1], 1.0)
        self.assertEqual(channel_values[2], 1.0)
        self.assertEqual(channel_values[3], 0.25)
        self.assertEqual(channel_values[5], 1.0)

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            RFControlAction(channel_overrides={12: 0.5})

        with self.assertRaises(ValueError):
            RFControlAction(throttle=float("nan"))


if __name__ == "__main__":
    unittest.main()

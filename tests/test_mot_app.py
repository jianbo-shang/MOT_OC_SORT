import unittest

from mot_app import target_frame_interval


class PlaybackSpeedTests(unittest.TestCase):
    def test_source_rate_playback_interval(self):
        self.assertAlmostEqual(target_frame_interval(10.0, 1.0), 0.1)
        self.assertAlmostEqual(target_frame_interval(10.0, 0.5), 0.2)
        self.assertAlmostEqual(target_frame_interval(10.0, 2.0), 0.05)

    def test_unlimited_or_invalid_rate_has_no_delay(self):
        self.assertEqual(target_frame_interval(10.0, 0.0), 0.0)
        self.assertEqual(target_frame_interval(0.0, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()

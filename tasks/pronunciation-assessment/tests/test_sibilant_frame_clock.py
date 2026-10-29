import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import numpy as np
from mdd.sibilant_frame_clock import stacked_window_centers


class FrameClockTest(unittest.TestCase):
    def test_stacked_support_midpoints_exclude_partial_pair(self):
        centers = stacked_window_centers(100, 10105, 30)
        self.assertEqual(float(centers[0]), 379.5)
        self.assertEqual(float(centers[-1]), 9659.5)
        np.testing.assert_array_equal(np.diff(centers), np.full(29, 320.))
        self.assertLess(float(centers[-1] + 279.5), 10105.)

    def test_wrong_mask_or_short_clip_is_rejected(self):
        for args in [(0, 10005, 31), (0, 400, 1), (-1, 10000, 30)]:
            with self.assertRaises(ValueError):
                stacked_window_centers(*args)


if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.sibilant_cmvn import statistics, stack_normalized


class CmvnTest(unittest.TestCase):
    def test_odd_unstacked_frame_still_contributes_to_statistics(self):
        raw = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [100., 100.]])
        before = raw.copy()
        output = stack_normalized(raw, *statistics(raw), length=2)
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue((output < 0).all())
        np.testing.assert_array_equal(before, raw)

    def test_outside_context_changes_only_the_reference_statistics(self):
        clip = np.arange(16, dtype=float).reshape(8, 2)
        saved = clip.copy()
        full_a = np.vstack([np.ones((8, 2)), clip])
        full_b = np.vstack([np.full((8, 2), 100.), clip])
        a = stack_normalized(clip, *statistics(full_a), length=4)
        b = stack_normalized(clip, *statistics(full_b), length=4)
        self.assertGreater(float(np.max(np.abs(a - b))), .1)
        np.testing.assert_array_equal(saved, clip)
        with self.assertRaises(ValueError):
            statistics(clip[:1])


if __name__ == "__main__":
    unittest.main()

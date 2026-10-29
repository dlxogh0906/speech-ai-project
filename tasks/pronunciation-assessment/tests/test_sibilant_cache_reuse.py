import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import numpy as np
from mdd.sibilant_cache_reuse import pair_reference_equal, local_relative


class CacheReuseTest(unittest.TestCase):
    def test_only_the_paired_target_reference_can_change(self):
        before = list("ㅅㅣㅅㅜ")
        self.assertTrue(pair_reference_equal(before, list("ㅆㅣㅅㅜ"), 0))
        self.assertFalse(pair_reference_equal(before, list("ㅆㅣㅆㅜ"), 0))
        self.assertFalse(pair_reference_equal(before, list("ㅆㅣㅅ"), 0))
        self.assertFalse(pair_reference_equal(before, before, 1))

    def test_clipped_input_keeps_the_original_anchor_clock(self):
        relative = local_relative(0, 8000, 25, 10, .02, .31)
        self.assertEqual(relative.dtype, np.float32)
        self.assertEqual(float(relative[10]), 0.)
        self.assertNotEqual(float(relative[12]), 0.)
        self.assertTrue(bool((np.diff(relative) > 0).all()))
        with self.assertRaises(ValueError):
            local_relative(0, 8000, 0, 10, .02, .31)


if __name__ == "__main__":
    unittest.main()

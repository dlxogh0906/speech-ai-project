import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles


class TagSupportTests(unittest.TestCase):
    def test_group_before_target_changes_index_not_target_interval(self):
        intervals = single_phone_intervals([(0, .2, "ㅗㄱ"), (.2, .3, "ㅅ"), (.3, .5, "ㅣ")], list("ㅗㄱㅅㅣ"))
        self.assertEqual(intervals[:2], [None, None])
        self.assertEqual(intervals[2], (.2, .3, "ㅅ"))
        phases = coarse_roles(np.array([.25, .325, .395]), intervals, 2, {"ㅣ"})
        np.testing.assert_array_equal(phases, [1, 2, 3])

    def test_grouped_target_is_not_silently_split(self):
        intervals = single_phone_intervals([(0, .2, "ㅅㅣ")], list("ㅅㅣ"))
        self.assertEqual(intervals, [None, None])
        np.testing.assert_array_equal(coarse_roles(np.array([.05, .15]), intervals, 0, {"ㅣ"}), [-100, -100])

    def test_mismatch_and_missing_target_stay_unknown(self):
        self.assertIsNone(single_phone_intervals([(0, .1, "ㅅ")], list("ㅆ")))
        np.testing.assert_array_equal(coarse_roles(np.array([.1]), None, None, {"ㅣ"}), [-100])

    def test_internal_whitespace_does_not_invalidate_unrelated_known_intervals(self):
        intervals = single_phone_intervals([(0, .2, "ㅡ ㄷ"), (.2, .3, "ㅅ"), (.3, .5, "ㅣ")], list("ㅡㄷㅅㅣ"))
        self.assertEqual(intervals[:2], [None, None])
        self.assertEqual(intervals[2:], [(.2, .3, "ㅅ"), (.3, .5, "ㅣ")])

    def test_unknown_non_whitespace_symbol_is_not_silently_discarded(self):
        self.assertIsNone(single_phone_intervals([(0, .2, "ㅅ?"), (.2, .3, "ㅣ")], list("ㅅㅣ")))


if __name__ == "__main__":
    unittest.main()

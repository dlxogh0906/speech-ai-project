import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.sibilant_union_anchor import union_anchor
from mdd.sibilant_transition import paired_anchor


class UnionAnchorTest(unittest.TestCase):
    def fixture(self):
        p = np.full((12, 4), .001)
        p[:, 0] = .997
        p[2] = [.1, .001, .89, .009]
        p[8] = [.1, .88, .001, .019]
        p[10] = [.01, .001, .001, .988]
        return np.log(p)

    def test_disjoint_paths_are_not_averaged(self):
        lp = self.fixture()
        old = paired_anchor(lp, [1, 3], 0, [1, 2], 0)
        new = union_anchor(lp, [1, 3], 0, [1, 2], 0)
        self.assertEqual(old['hypothesis_frames'], [8., 2.])
        self.assertEqual(old['frame'], 5)
        self.assertEqual(new['frame'], 2)
        self.assertEqual(new['segment'], [2])

    def test_symmetric_in_identity(self):
        lp = self.fixture()
        a = union_anchor(lp, [1, 3], 0, [1, 2], 0)
        b = union_anchor(lp[:, [0, 2, 1, 3]], [2, 3], 0, [1, 2], 0)
        self.assertEqual(a, b)

    def test_repetition_guard_and_impossible_alignment(self):
        self.assertIsNone(union_anchor(self.fixture(), [1, 2, 3], 1, [1, 2], 0))
        self.assertIsNone(union_anchor(self.fixture()[:1], [1, 3], 0, [1, 2], 0))


if __name__ == '__main__':
    unittest.main()

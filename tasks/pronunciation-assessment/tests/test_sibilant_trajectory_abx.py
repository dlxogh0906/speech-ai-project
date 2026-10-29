"""Meaningful metric controls, without loading any speech corpus."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
import unittest
import numpy as np
from mdd.sibilant_trajectory_abx import (unit_rows, angular_matrix, permutation,
    dtw, class_balanced_abx, summarize)


class TrajectoryTest(unittest.TestCase):
    def test_order_and_rate_controls(self):
        a = unit_rows(np.eye(4))
        self.assertAlmostEqual(dtw(angular_matrix(a, a)), 0.)
        self.assertAlmostEqual(dtw(angular_matrix(a, np.repeat(a, 2, axis=0))), 0.)
        self.assertGreater(dtw(angular_matrix(a, a[::-1])), .1)
        np.testing.assert_array_equal(a.mean(0), a[::-1].mean(0))
        np.testing.assert_allclose(unit_rows(7*a), a)
        for cost in [np.empty((0, 3)), np.array([[np.nan]]), np.array([[-1.]])]:
            with self.assertRaises(ValueError):
                dtw(cost)

    def test_balanced_all_reference_triplets(self):
        # A large same-class majority must not replace actual-class macro averaging.
        left, right = np.array([0, 0, 1]), np.array([0, 1, 1])
        perfect = (left[:, None] != right[None]).astype(float)
        self.assertEqual(class_balanced_abx(perfect, left, right)[0], 1.)
        self.assertEqual(class_balanced_abx(1-perfect, left, right)[0], 0.)
        self.assertEqual(class_balanced_abx(np.ones((3, 3)), left, right)[0], .5)
        with self.assertRaises(ValueError):
            class_balanced_abx(perfect, left, np.zeros(3))

    def test_gate_failures_and_identity_permutation(self):
        np.testing.assert_array_equal(permutation('uid/0', 32, 20261025), permutation('uid/0', 32, 20261025))
        self.assertFalse(np.array_equal(permutation('uid/0', 32, 20261025), permutation('uid/1', 32, 20261025)))
        for bad in [np.zeros((1, 4)), np.array([[np.inf, 0]])]:
            with self.assertRaises(ValueError):
                unit_rows(bad)
        values = np.tile([.7, .6, .6], (12, 1))
        self.assertTrue(summarize(values, ['ㅣ']*12, 1)['proceed_trajectory_detector'])
        self.assertFalse(summarize(values, ['ㅡ']*12, 1)['proceed_trajectory_detector'])
        values[:, 1] = .71
        self.assertFalse(summarize(values, ['ㅣ']*12, 1)['proceed_trajectory_detector'])


if __name__ == '__main__':
    unittest.main()

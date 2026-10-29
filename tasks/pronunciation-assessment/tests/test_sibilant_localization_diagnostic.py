import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.sibilant_localization_diagnostic import localized_summary
from mdd.sibilant_linear import temporal_summary


class LocalizationTests(unittest.TestCase):
    def fixture(self):
        x = np.zeros((1, 31, 64))
        x[:, 10:13] = 1.
        x[:, 13:15] = 2.
        x[:, 15:18] = 3.
        phases = np.zeros((1, 3, 31), bool)
        phases[:, 0, 10:13] = True
        phases[:, 1, 13:15] = True
        phases[:, 2, 15:18] = True
        return x, np.ones((1, 31), bool), np.zeros((1, 24)), phases

    def test_outside_context_does_not_enter_local_means(self):
        x, valid, ctx, phases = self.fixture()
        first, available = localized_summary(x, valid, ctx, phases)
        before = temporal_summary(x, valid, ctx, True)
        x[:, :10] = 1000.
        second, _ = localized_summary(x, valid, ctx, phases)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(before, temporal_summary(x, valid, ctx, True)))
        self.assertEqual(first.shape, (1, 219))
        self.assertTrue(available[0])
        np.testing.assert_array_equal(first[0, 64:128], -1.)
        np.testing.assert_array_equal(first[0, 128:192], 1.)

    def test_missing_phase_remains_unavailable(self):
        x, valid, ctx, phases = self.fixture()
        phases[:, 2] = False
        _, available = localized_summary(x, valid, ctx, phases)
        self.assertFalse(available[0])

    def test_invalid_or_overlapping_support_rejected(self):
        x, valid, ctx, phases = self.fixture()
        valid[:, 10] = False
        with self.assertRaises(ValueError):
            localized_summary(x, valid, ctx, phases)
        valid[:, 10] = True
        phases[:, 1, 10] = True
        with self.assertRaises(ValueError):
            localized_summary(x, valid, ctx, phases)


if __name__ == "__main__":
    unittest.main()

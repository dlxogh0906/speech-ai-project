import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.sibilant_linear import temporal_summary, predict_readout


class LinearReadoutTests(unittest.TestCase):
    def test_order_contrasts_distinguish_equal_means(self):
        x = np.zeros((2, 31, 1))
        x[0, :10, 0], x[1, 21:, 0] = 3, 3
        mask = np.ones((2, 31), bool)
        context = np.zeros((2, 1))
        mean = temporal_summary(x, mask, context, False)
        temporal = temporal_summary(x, mask, context, True)
        np.testing.assert_array_equal(mean[0], mean[1])
        self.assertFalse(np.array_equal(temporal[0], temporal[1]))

    def test_padding_values_cannot_become_acoustic_evidence(self):
        x = np.ones((1, 31, 2))
        mask = np.ones((1, 31), bool)
        mask[:, :15] = False
        before = temporal_summary(x, mask, np.zeros((1, 1)), True)
        x[:, :15] = np.nan
        after = temporal_summary(x, mask, np.zeros((1, 1)), True)
        np.testing.assert_array_equal(before, after)

    def test_runtime_readout_has_stable_extreme_scores(self):
        artifact = dict(mean=np.zeros(1), scale=np.ones(1), coefficient=np.ones(1), intercept=0.)
        score = predict_readout(np.array([[-1000.], [0], [1000]]), artifact)
        np.testing.assert_array_equal(score, [0., .5, 1.])


if __name__ == "__main__":
    unittest.main()

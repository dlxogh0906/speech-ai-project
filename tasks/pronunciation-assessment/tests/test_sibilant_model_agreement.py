import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import numpy as np
from mdd.sibilant_model_agreement import confirmed_availability


class ModelAgreementTest(unittest.TestCase):
    def test_identity_and_both_observations_required_without_score_changes(self):
        full = np.array([[.9, .05, .03, .02], [.9, .05, .03, .02], [.9, .05, .03, .02], [.01, .01, .95, .03]])
        other = full.copy()
        other[1] = [.05, .9, .03, .02]
        before = full.copy()
        available = np.array([True] * 4)
        result = confirmed_availability(full, available, other, [True, True, False, True])
        np.testing.assert_array_equal(result, [True, False, False, False])
        np.testing.assert_array_equal(full, before)
        np.testing.assert_array_equal(available, [True] * 4)

    def test_unmatched_or_nonfinite_model_outputs_cannot_confirm(self):
        with self.assertRaises(ValueError):
            confirmed_availability(np.zeros((2, 4)), [True] * 2, np.zeros((3, 4)), [True] * 3)
        with self.assertRaises(ValueError):
            confirmed_availability(np.array([[np.nan, 0., 0., 0.]]), [True], np.zeros((1, 4)), [True])


if __name__ == "__main__":
    unittest.main()

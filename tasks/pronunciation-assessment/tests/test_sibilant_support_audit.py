import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.sibilant_support_audit import support_measurements


class SupportAuditTest(unittest.TestCase):
    def test_correct_peak_does_not_imply_concentrated_pooling(self):
        p = np.tile([.7, .1, .1, .1], (30, 1))
        roles = np.zeros(30, np.int64)
        for frame, role in ((2, 1), (5, 2), (8, 3)):
            roles[frame] = role
            p[frame] = .05
            p[frame, role] = .85
        audit = support_measurements(p, roles, np.ones(30, bool))
        self.assertEqual(audit["peak_hits"], [True] * 3)
        self.assertTrue(all(v["own_known"] < .25 for v in audit["soft_mass"]))
        self.assertEqual(audit["confident_mass"], [1.] * 3)

    def test_unknown_boundaries_and_padding_are_not_true_background(self):
        p = np.tile([.1, .7, .1, .1], (5, 1))
        roles, mask = np.array([1, -100, 2, 3, 0]), np.array([True] * 4 + [False])
        p[4] = np.nan
        result = support_measurements(p, roles, mask)
        self.assertEqual(np.array(result["confusion"]).sum(), 3)
        self.assertAlmostEqual(result["soft_mass"][0]["unknown"], .25)
        self.assertTrue(result["complete_observed_roles"])


if __name__ == "__main__":
    unittest.main()

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
from mdd.sibilant_location_abx import crop_indices, retained_abx, location_result
from mdd.sibilant_trajectory_abx import class_balanced_abx


class LocationTest(unittest.TestCase):
    def test_crop_preserves_length_and_source(self):
        t = .01+.02*np.arange(31)
        tag, control, state = crop_indices(t, (.1, .2, 'ㅅ'), (.2, .5, 'ㅏ'), .4)
        self.assertEqual(state, 'available')
        self.assertEqual(len(tag), len(control))
        self.assertTrue((t[tag] >= .1).all() and (t[tag] < .32).all())
        self.assertGreater(control[0], tag[0])
        for ref in [-10., 10.]:
            a, b, state = crop_indices(t, (.1, .2), (.2, .5), ref)
            self.assertEqual(len(a), len(b))
            self.assertTrue((b >= 0).all() and (b < len(t)).all())
        self.assertEqual(crop_indices(t, (.1, .2), (.2, .201), .2)[2], 'no_observed_initialV_frame')
        self.assertEqual(crop_indices(t, None, (.2, .5), .2)[2], 'missing_interval')

    def test_unavailable_triplets_stay_in_denominator(self):
        a, b = np.array([0, 0, 1]), np.array([0, 1, 1])
        d = (a[:, None] != b[None]).astype(float)
        x = retained_abx(d, a, b, np.ones(3, bool), np.ones(3, bool))
        self.assertEqual(x['retained_accuracy'], class_balanced_abx(d, a, b)[0])
        self.assertEqual(x['triplet_coverage'], 1.)
        x = retained_abx(d, a, b, np.zeros(3, bool), np.ones(3, bool))
        self.assertEqual(x['retained_accuracy'], .5)
        self.assertIsNone(x['valid_only_accuracy'])
        self.assertEqual(x['triplet_coverage'], 0.)
        x = retained_abx(d, a, b, np.array([1, 0, 1]), np.ones(3, bool))
        self.assertGreater(x['retained_accuracy'], .5)
        self.assertLess(x['retained_accuracy'], 1.)
        self.assertEqual(x['valid_only_accuracy'], 1.)
        random = np.random.RandomState(17).rand(3, 3)
        self.assertAlmostEqual(retained_abx(random, a, b, np.ones(3, bool), np.ones(3, bool))['retained_accuracy'],
                               class_balanced_abx(random, a, b)[0])

    def test_gate_requires_gain_over_both_and_coverage(self):
        x = np.tile([.7, .6, .6], (20, 1))
        self.assertTrue(location_result(x, ['ㅣ']*20, {'all':1., 'ㅣ':1.}, 1)['proceed_automatic_localization_candidate'])
        self.assertFalse(location_result(x, ['ㅣ']*20, {'all':1., 'ㅣ':.89}, 1)['proceed_automatic_localization_candidate'])
        x[:, 1] = .69
        self.assertFalse(location_result(x, ['ㅣ']*20, {'all':1., 'ㅣ':1.}, 1)['proceed_automatic_localization_candidate'])


if __name__ == '__main__':
    unittest.main()

"""Prevent no-signal inputs from becoming F0 evidence in the optional detector."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import unittest
import numpy as np
from mdd.sz_detector import yin, site_features


class LegacyYinSignalValidity(unittest.TestCase):
    def test_silence_and_constant_offset_are_not_periodic(self):
        for length in (640, 1600):
            for value in (0., .2, -.3):
                self.assertIsNone(yin(np.full(length, value)))

    def test_nonfinite_samples_do_not_create_pitch(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            x = np.sin(2 * np.pi * 200 * np.arange(640) / 16000)
            x[100] = value
            self.assertIsNone(yin(x))

    def test_quiet_periodic_voice_is_retained_and_pcm_unchanged(self):
        for gain in (.1, 1e-6):
            x = .2 + gain * np.sin(2 * np.pi * 200 * np.arange(640) / 16000)
            before = x.copy()
            self.assertAlmostEqual(yin(x), 200., delta=.05)
            np.testing.assert_array_equal(x, before)

    def test_silent_following_vowel_sets_existing_missingness_flag(self):
        canon = ['ㅅ', 'ㅏ']
        ids = {'ㅅ': 0, 'ㅆ': 1, 'ㅏ': 2}
        log_probs = np.full((10, 3), -2.)
        features = site_features(np.zeros(3200), log_probs, [[1], [3, 4]], canon,
                                 0, ids.__getitem__, .02, 200.)
        self.assertEqual(features['f0miss'], 1.)
        # Existing model-input defaults remain; they are not measured H1-H2/F0.
        self.assertEqual(features['f0st'], 0.)
        self.assertEqual(features['h1h2'], 0.)


if __name__ == '__main__':
    unittest.main()

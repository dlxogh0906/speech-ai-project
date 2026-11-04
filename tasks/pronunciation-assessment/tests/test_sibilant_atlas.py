"""Guard physical support, silent inputs, and non-cherry-picked context pairs."""
import unittest
import numpy as np

from mdd.sibilant_atlas import extract, periodicity_curve, paired_cases


class TestSibilantAtlas(unittest.TestCase):
    def test_known_band_and_vowel_growth_preserve_amplitude_invariant_features(self):
        sr = 16000; t = np.arange(6400) / sr
        x = np.zeros_like(t)
        consonant = (t >= .1) & (t < .2)
        x[consonant] = .04 * np.sin(2 * np.pi * 5000 * t[consonant])
        vowel = (t >= .2) & (t < .4)
        x[vowel] = .08 * np.sin(2 * np.pi * 200 * t[vowel])
        x[(t >= .2) & (t < .23)] *= .5
        a = extract(x, sr, [.1, .2], [.2, .4])
        b = extract(x * .5, sr, [.1, .2], [.2, .4])
        self.assertAlmostEqual(a['features']['centroid_khz'], 5., places=2)
        self.assertAlmostEqual(a['features']['vowel_rise_db'], 20 * np.log10(2), places=6)
        for k in a['features']:
            self.assertAlmostEqual(a['features'][k], b['features'][k], places=6)
        self.assertGreaterEqual(a['high_run'][0], .1 - 1e-9)
        self.assertLessEqual(a['high_run'][1], .2 + 1e-9)

    def test_silence_and_short_vowel_are_missing_not_zero(self):
        a = extract(np.zeros(8000), 16000, [.1, .2], [.2, .22])
        self.assertAlmostEqual(a['features']['tag_duration_ms'], 100.)
        for k, v in a['features'].items():
            if k != 'tag_duration_ms': self.assertIsNone(v)
        self.assertIsNone(a['high_run']); self.assertIsNone(a['periodic_run'])
        self.assertTrue(all(r['value'] is None for r in periodicity_curve(np.zeros(8000), 16000)))

    def test_matched_pairs_require_preceding_phone_word_position_and_same_speaker(self):
        def c(key, actual, previous='ㅏ', word='어절 내부', speaker='a'):
            return dict(case_id=key, actual=actual, previous=previous, following='ㅣ',
                        position='초성 추정', word_position=word, speaker=speaker, predicted='ㅅ')
        rows = [c('s1','ㅅ'), c('s2','ㅅ'), c('ss1','ㅆ'), c('ss2','ㅆ','ㅓ'),
                c('ss3','ㅆ',word='어절 처음'), c('ss4','ㅆ',speaker='b')]
        pairs = paired_cases(rows)
        self.assertEqual(len(pairs), 1); self.assertEqual(pairs[0]['ss'], 'ss1')
        for r in rows: r['predicted'] = 'ㅆ'
        self.assertEqual(paired_cases(list(reversed(rows))), pairs)


if __name__ == '__main__': unittest.main()

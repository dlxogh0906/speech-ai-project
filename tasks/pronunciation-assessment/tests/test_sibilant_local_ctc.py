import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
from infer_sibilant_local_ctc import classify


class LocalCTCTest(unittest.TestCase):
    dictionary = ['<blank>', 'ㅅ', 'ㅆ', 'ㅣ', 'ㅜ', 'ㄴ']

    def sequence(self, tokens):
        z = np.full((len(tokens), 6), -12.)
        for t, token in enumerate(tokens):
            z[t, self.dictionary.index(token)] = 0.
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    def test_reference_consonant_does_not_change_observed_recognition(self):
        lp = self.sequence(['<blank>', 'ㅅ', '<blank>', 'ㅣ', '<blank>', 'ㅆ', '<blank>', 'ㅜ'])
        f = {'observed_V': 'ㅣ', 'canonical': 'ㅆ'}
        a, detail = classify(lp, f, np.arange(8) - 3, self.dictionary, 0)
        b, _ = classify(lp, dict(f, canonical='ㅅ'), np.arange(8) - 3, self.dictionary, 0)
        self.assertEqual(detail['actual'], 'ㅅ')
        self.assertEqual(detail['coarse_class'], 0)
        np.testing.assert_array_equal(a, b)

    def test_missing_C_token_is_an_observation_not_a_new_deletion_call(self):
        lp = self.sequence(['<blank>', '<blank>', 'ㅣ', '<blank>'])
        p, detail = classify(lp, {'observed_V': 'ㅣ'}, np.arange(4) - 2, self.dictionary, 0)
        self.assertEqual(detail['state'], 'local_V_without_C_token')
        self.assertIsNone(detail['actual'])
        self.assertEqual(int(p.argmax()), 4)

    def test_ambiguous_or_missing_V_abstains(self):
        lp = self.sequence(['<blank>', '<blank>', 'ㅣ', '<blank>', 'ㅅ', '<blank>', 'ㅣ', '<blank>'])
        self.assertIsNone(classify(lp, {'observed_V': 'ㅣ'}, np.arange(8) - 4, self.dictionary, 0)[0])
        self.assertIsNone(classify(lp, {'observed_V': 'ㅜ'}, np.arange(8) - 4, self.dictionary, 0)[0])


if __name__ == '__main__':
    unittest.main()

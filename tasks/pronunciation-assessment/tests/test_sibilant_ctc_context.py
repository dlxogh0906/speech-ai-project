import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
from score_sibilant_ctc_context import score_batch, hypotheses


class CTCContextTest(unittest.TestCase):
    vocab = {'ㅅ': 1, 'ㅆ': 2, 'ㅣ': 3}

    def test_summing_paths_can_reverse_a_greedy_phone_choice(self):
        p = np.array([[[.05, .45, .5, 1e-8], [.51, .49, 1e-8, 1e-8], [1e-8, 1e-8, 1e-8, 1.]]])
        lp = np.log(p / p.sum(-1, keepdims=True))
        q = {'state': 'local_C_V', 'actual': 'ㅆ', 'local_V_index': 1,
             'tokens': ['ㅆ', 'ㅣ'], 'spans': [[0, 1], [2, 3]]}
        ps, av, detail = score_batch(lp, [3], [q], self.vocab, 0)
        self.assertTrue(av[0])
        self.assertEqual(detail[0]['chosen'], 'ㅅ')
        self.assertGreater(float(ps[0, 0]), .5)

    def test_no_C_competes_instead_of_forcing_S_or_SS(self):
        p = np.array([[[.99, .003, .003, .004], [.01, .001, .001, .988], [.99, .003, .003, .004]]])
        q = {'state': 'local_V_without_C_token', 'actual': None, 'local_V_index': 0,
             'tokens': ['ㅣ'], 'spans': [[1, 2]]}
        ps, av, detail = score_batch(np.log(p), [3], [q], self.vocab, 0)
        self.assertTrue(av[0])
        self.assertIsNone(detail[0]['chosen'])
        self.assertGreater(float(ps[0, 4]), .9)

    def test_missing_vowel_abstains_and_impossible_counterfactual_is_not_zero_masked(self):
        self.assertEqual(hypotheses({'state': 'local_V_missing'}, self.vocab, 1), [])
        q = {'state': 'local_V_without_C_token', 'actual': None, 'local_V_index': 0,
             'tokens': ['ㅣ'], 'spans': [[0, 1]]}
        self.assertEqual(hypotheses(q, self.vocab, 1), [(None, [3])])


if __name__ == '__main__':
    unittest.main()

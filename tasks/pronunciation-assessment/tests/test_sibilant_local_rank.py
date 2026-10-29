import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/data'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
from prepare_sibilant_local_rank import rank_target


class LocalRankTest(unittest.TestCase):
    vocab = {'ㅅ': 1, 'ㅆ': 2, 'ㅣ': 3}
    tag = [(0, .2, 'ㅅ'), (.2, .4, 'ㅆ'), (.4, .6, 'ㅣ')]

    def label(self, actual, a, expected):
        return {'actual': actual, 'a': a, 'alignment_ambiguous': False, 'expected': expected}

    def test_actual_error_is_not_restored_and_normal_gets_same_opposite(self):
        target = {'TAG_indices': [1, 2], 'tokens': ['ㅆ', 'ㅣ'], 'ids': [2, 3], 'roi': [0, 6]}
        error, _ = rank_target(self.label('ㅆ', 1, 'ㅅ'), target, self.tag, list('ㅅㅆㅣ'), self.vocab)
        normal, _ = rank_target(self.label('ㅆ', 1, 'ㅆ'), target, self.tag, list('ㅅㅆㅣ'), self.vocab)
        self.assertEqual(error, normal)
        self.assertEqual(error['actual_id'], 2)
        self.assertEqual(error['alternate_id'], 1)

    def test_counterfactual_repeat_requires_extra_CTC_frame(self):
        target = {'TAG_indices': [0, 1], 'tokens': ['ㅅ', 'ㅆ'], 'ids': [1, 2], 'roi': [0, 2]}
        value, state = rank_target(self.label('ㅆ', 1, 'ㅅ'), target, self.tag, list('ㅅㅆㅣ'), self.vocab)
        self.assertIsNone(value)
        self.assertEqual(state, 'counterfactual_CTC_path_impossible')

    def test_unobserved_or_ambiguous_C_has_no_rank_target(self):
        target = {'TAG_indices': [2], 'tokens': ['ㅣ'], 'ids': [3], 'roi': [0, 5]}
        self.assertIsNone(rank_target(self.label('ㅆ', 1, 'ㅅ'), target, self.tag, list('ㅅㅆㅣ'), self.vocab)[0])
        label = dict(self.label('ㅆ', 1, 'ㅅ'), alignment_ambiguous=True)
        self.assertIsNone(rank_target(label, target, self.tag, list('ㅅㅆㅣ'), self.vocab)[0])


if __name__ == '__main__':
    unittest.main()

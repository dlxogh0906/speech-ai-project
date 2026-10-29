import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.sibilant_observed_anchor import observed_tokens, linked_anchor


class ObservedAnchorTest(unittest.TestCase):
    def test_blank_separated_repetitions_and_special_tokens(self):
        path = [0, 1, 1, 0, 1, 3, 2, 2]
        lp = np.full((len(path), 4), -10.)
        lp[np.arange(len(path)), path] = 0
        tokens, spans = observed_tokens(lp, ['<pad>', 'ㅅ', 'ㅣ', '<unk>'], 0)
        self.assertEqual(tokens, ['ㅅ', 'ㅅ', 'ㅣ'])
        self.assertEqual(spans, [(1, 3), (4, 5), (6, 8)])
        self.assertEqual(linked_anchor(tokens, spans, {'j': 1, 'pred': 'ㅅ'})['frame'], 4)

    def test_unlinked_never_borrows_next_token(self):
        self.assertIsNone(linked_anchor(['ㅣ'], [(8, 9)], {'j': None, 'pred': None}))
        with self.assertRaises(ValueError):
            linked_anchor(['ㅣ'], [(8, 9)], {'j': 0, 'pred': 'ㅅ'})


if __name__ == '__main__':
    unittest.main()

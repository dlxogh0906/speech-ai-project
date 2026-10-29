import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.sibilant_local_ctc_targets import target_region


class LocalCTCTargetTest(unittest.TestCase):
    feature = {'left': 0, 'right': 10320, 'input_frames': 31}
    vocab = {'ㅅ': 1, 'ㅏ': 2, 'ㅣ': 3, 'ㄴ': 4}

    def test_empty_interval_has_no_phone_label_but_can_have_loss_frames(self):
        value, state = target_region(self.feature, [[0, .2, ''], [.2, .4, 'ㅣ'], [.4, .64, 'ㄴ']], ['ㅣ', 'ㄴ'], self.vocab)
        self.assertEqual(state, 'supported')
        self.assertEqual(value['tokens'], ['ㅣ', 'ㄴ'])
        self.assertEqual(value['roi'][0], 0)
        self.assertNotIn('ㅅ', value['tokens'])

    def test_grouped_known_phones_need_no_invented_internal_boundary(self):
        value, state = target_region(self.feature, [[0, .64, 'ㅅㅏ']], ['ㅅ', 'ㅏ'], self.vocab)
        self.assertEqual(state, 'supported')
        self.assertEqual(value['tokens'], ['ㅅ', 'ㅏ'])
        self.assertEqual(value['TAG_indices'], [0])

    def test_unknown_symbols_and_sequence_mismatch_do_not_become_targets(self):
        self.assertIsNone(target_region(self.feature, [[0, .3, 'spn'], [.3, .64, 'ㅏ']], ['ㅏ'], self.vocab)[0])
        self.assertIsNone(target_region(self.feature, [[0, .64, 'ㅏ']], ['ㅅ'], self.vocab)[0])

    def test_short_internal_phone_is_preserved_without_fake_frame_boundary(self):
        value, state = target_region(self.feature, [[0, .2, 'ㅏ'], [.2, .21, 'ㄴ'], [.21, .64, 'ㅣ']],
                                     ['ㅏ', 'ㄴ', 'ㅣ'], self.vocab)
        self.assertEqual(state, 'supported')
        self.assertEqual(value['tokens'], ['ㅏ', 'ㄴ', 'ㅣ'])

    def test_whitespace_groups_are_preserved_but_partial_groups_are_unknown(self):
        value, state = target_region(self.feature, [[0, .64, 'ㅅ ㅏ']], ['ㅅ', 'ㅏ'], self.vocab)
        self.assertEqual(state, 'supported')
        self.assertEqual(value['tokens'], ['ㅅ', 'ㅏ'])
        self.assertIsNone(target_region(self.feature, [[-.3, .3, 'ㅅㅏ']], ['ㅅ', 'ㅏ'], self.vocab)[0])


if __name__ == '__main__':
    unittest.main()

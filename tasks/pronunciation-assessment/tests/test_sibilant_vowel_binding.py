import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
from audit_sibilant_vowel_binding import crop_actual_indices, sequence_binding


class VowelBindingTest(unittest.TestCase):
    def test_grouped_whitespace_keeps_original_indices_without_inner_boundaries(self):
        tag = [(0, .2, 'ㅡ ㄷ'), (.2, .3, 'ㅅ'), (.3, .5, 'ㅣ')]
        self.assertEqual(crop_actual_indices(tag, list('ㅡㄷㅅㅣ'), [1, 2]), [2, 3])

    def test_same_vowel_in_other_slot_is_not_target_binding(self):
        actual = list('ㅅㅣㄴㅣ')
        query = {'state': 'local_C_V', 'tokens': actual, 'spans': [(1, 2), (3, 4), (5, 6), (7, 8)],
                 'local_V_index': 3}
        self.assertEqual(sequence_binding(actual, [0, 1, 2, 3], 1, query, [0, 9])[0],
                         'selected_other_sequence_slot')

    def test_ambiguous_alignment_or_unobserved_target_stays_unknown(self):
        query = {'state': 'local_C_V', 'tokens': list('ㅅㅣㅣ'), 'spans': [(1, 2), (3, 4), (5, 6)],
                 'local_V_index': 1}
        self.assertEqual(sequence_binding(list('ㅅㅣ'), [0, 1], 1, query, [0, 8])[0], 'sequence_mapping_ambiguous')
        self.assertEqual(sequence_binding(list('ㅅㅣㅏ'), [0, 1], 2, query, [0, 8])[0],
                         'intended_V_outside_supported_sequence')


if __name__ == '__main__':
    unittest.main()

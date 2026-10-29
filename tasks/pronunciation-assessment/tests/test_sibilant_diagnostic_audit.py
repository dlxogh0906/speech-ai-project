"""Check exclusive accounting and paired speaker uncertainty."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
from mdd.sibilant_diagnostic_audit import error_stage, false_warning_stage, grouped_bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
from audit_sibilant_diagnostic_context import spelling_context


class DiagnosticAuditTest(unittest.TestCase):
    def test_rejection_is_not_decoder_error(self):
        z = {'expected': 'ㅆ', 'actual': 'ㅅ', 'label_kind': 'pair_error'}
        self.assertEqual(error_stage(z, {'state': 'input_rejected', 'phone': None}), 'input_rejected')
        v = {'state': 'available', 'phone': {'alpha_jamo': 'ㅆ', 'actual_jamo': 'ㅅ', 'uncertain': True}}
        self.assertEqual(error_stage(z, v), 'TAG_actual_in_final_phone_but_warning_suppressed')

    def test_reference_disagreement_precedes_acoustic_claim(self):
        z = {'expected': 'ㅅ', 'actual': 'ㅅ', 'label_kind': 'correct'}
        v = {'state': 'runtime_canonical_disagrees', 'phone': {'alpha_jamo': 'ㅆ', 'actual_jamo': 'ㅅ', 'errorType': 'sub'}}
        self.assertEqual(false_warning_stage(z, v), 'reference_disagreement_requires_normative_review')

    def test_identical_policy_has_zero_paired_difference(self):
        rows = [{'speaker': str(i), 'delta': 0, 'den': i % 3} for i in range(12)]
        z = grouped_bootstrap(rows, {'paired': ('delta', 'den')}, 200, 7)['paired']
        self.assertEqual(z['estimate'], 0)
        self.assertEqual(z['percentile95'], [0., 0.])

    def test_zero_event_speakers_retained(self):
        rows = [{'speaker': str(i), 'num': int(i == 0), 'den': 1} for i in range(10)]
        z = grouped_bootstrap(rows, {'rate': ('num', 'den')}, 200, 7)['rate']
        self.assertEqual(z['speakers'], 10)
        self.assertAlmostEqual(z['estimate'], .1)

    def test_spelling_and_tensification_candidates(self):
        cases = [('싸요', list('ㅆㅏㅛ'), 0, 'written_SS'),
                 ('국수', list('ㄱㅜㄱㅆㅜ'), 3, 'post_obstruent_coda_candidate'),
                 ('할 수', list('ㅎㅏㄹㅆㅜ'), 3, 'adnominal_L_candidate_not_morphological_gold'),
                 ('발생', list('ㅂㅏㄹㅆㅐㅇ'), 3, 'lexical_L_candidate_not_etymological_gold')]
        for prompt, canonical, q, expected in cases:
            row = {'prompt': prompt, 'canonical': canonical}
            label = {'q': q, 'expected': 'ㅆ'}
            self.assertEqual(spelling_context(row, label)[0], expected)


if __name__ == '__main__':
    unittest.main()

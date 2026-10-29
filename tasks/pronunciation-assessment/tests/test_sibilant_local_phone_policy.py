import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
from mdd.sibilant_local_phone_policy import local_phone_decision
from calibrate_sibilant_local_phones import fit


def feature(warning=False, actual='ㅅ'):
    return {'canonical': 'ㅅ', 'baseline_warning': warning, 'baseline_actual': actual}


def query(actual, score):
    return {'state': 'local_C_V', 'actual': actual, 'confidence': score}


def label(i, actual, old='ㅅ', warning=False):
    return {'feature_index': i, 'label_kind': 'third_phone', 'expected': 'ㅅ', 'actual': actual,
            'existing': {'state': 'available', 'phone': {'alpha_jamo': 'ㅅ', 'actual_jamo': old,
                         **({'errorType': 'test'} if warning else {})}}}


class LocalPhonePolicyTest(unittest.TestCase):
    def test_full_mode_retains_actual_third_consonant_identity(self):
        f, q = feature(), query('ㄴ', .99)
        full = fit([f], [label(0, 'ㄴ')], [q], 'full')
        pair = fit([f], [label(0, 'ㄴ')], [q], 'pair')
        result = full['alternatives']['one_percent']
        self.assertTrue(result['feasible'])
        self.assertFalse(pair['alternatives']['one_percent']['feasible'])
        self.assertEqual(local_phone_decision(f, q, result['policy'])['actual'], 'ㄴ')

    def test_new_detection_cannot_pay_for_lost_previously_exact_third_phone(self):
        fs = [feature(True, 'ㄷ'), feature()]
        labels = [label(0, 'ㄷ', 'ㄷ', True), label(1, 'ㄴ')]
        report = fit(fs, labels, [query('ㄴ', .99), query('ㄴ', .98)], 'full')
        self.assertFalse(report['alternatives']['one_percent']['feasible'])

    def test_absent_local_C_is_not_a_new_deletion_call(self):
        p = {'mode': 'full', 'defer': {'ㅅ': .5}, 'actual': {'ㅅ': .5}}
        q = {'state': 'local_V_without_C_token', 'actual': None, 'confidence': .99}
        self.assertIsNone(local_phone_decision(feature(True, None), q, p))


if __name__ == '__main__':
    unittest.main()

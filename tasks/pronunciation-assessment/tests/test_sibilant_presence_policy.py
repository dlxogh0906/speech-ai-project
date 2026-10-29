"""Behavioral controls for the inactive V-reference decision screen."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
from mdd.sibilant_presence_policy import decision
from eval_sibilant_presence_screen import calibrate


def label(i, actual, kind, warning=False, baseline_actual='ㅆ'):
    return {'feature_index': i, 'label_kind': kind, 'expected': 'ㅆ', 'actual': actual,
            'existing': {'state': 'available', 'phone': {'alpha_jamo': 'ㅆ',
            'actual_jamo': baseline_actual, **({'errorType': 'test'} if warning else {})}}}


def feature(warning=False, actual='ㅆ'):
    return {'canonical': 'ㅆ', 'baseline_warning': warning, 'baseline_actual': actual}


class PolicyTest(unittest.TestCase):
    def test_unavailable_and_absence_class_never_invent_phone(self):
        p = {'defer': {'ㅆ': .5}, 'pair': {'ㅆ': .5}}
        self.assertIsNone(decision(feature(), [.9, .05, .03, .02], False, p))
        self.assertIsNone(decision(feature(True, None), [.01, .01, .01, .01, .96], True, p))
        self.assertEqual(decision(feature(True, None), [.01, .96, .01, .01, .01], True, p)['kind'], 'defer')

    def test_joint_budget_can_use_removed_false_alert_without_erasing_error(self):
        # One false old warning is deferred; a missed pair error is detected.
        fs = [feature(True, 'ㅈ'), feature(), feature(True, 'ㅅ')]
        ls = [label(0, 'ㅆ', 'correct', True, 'ㅈ'), label(1, 'ㅅ', 'pair_sub'),
              label(2, 'ㅅ', 'pair_sub', True, 'ㅅ')]
        ps = np.array([[.01, .97, .01, .01], [.98, .01, .005, .005], [.7, .2, .05, .05]])
        r = calibrate(fs, ls, ps, np.ones(3, bool))
        c = r['alternatives']['one_percent']
        self.assertTrue(c['feasible'])
        self.assertEqual(c['counts']['normal_alerts'], 0)
        self.assertEqual(c['changes']['additional_exact'], 1)
        self.assertEqual(c['changes']['lost_exact'], 0)

    def test_unlabelled_high_confidence_query_cannot_silently_modify(self):
        fs = [feature(), feature()]
        ls = [label(0, 'ㅅ', 'pair_sub')]
        ps = np.array([[.9, .05, .03, .02], [.95, .02, .02, .01]])
        r = calibrate(fs, ls, ps, np.ones(2, bool))
        self.assertFalse(r['alternatives']['one_percent']['feasible'])


if __name__ == '__main__':
    unittest.main()

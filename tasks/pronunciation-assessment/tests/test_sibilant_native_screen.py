"""CAL safety: aggregate gains cannot hide changes to ambiguous GT sites."""
import copy
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/eval'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import numpy as np
from score_sibilant_native_conditional import screen


class ScreenTest(unittest.TestCase):
    def fixture(self):
        features = [{'canonical':'ㅅ','baseline_warning':True,'baseline_actual':'ㅆ'},
            {'canonical':'ㅆ','baseline_warning':False,'baseline_actual':'ㅆ'},
            {'canonical':'ㅆ','baseline_warning':True,'baseline_actual':'ㅊ'}]
        labels = []
        for i,(kind,expected,actual,baseline,warning) in enumerate([
            ('correct','ㅅ','ㅅ','ㅆ',True),('pair_error','ㅆ','ㅅ','ㅆ',False),
            ('ambiguous','ㅆ',None,'ㅊ',True)]):
            phone = {'alpha_jamo':expected,'actual_jamo':baseline}
            if warning:phone['errorType']='substitution'
            labels.append({'feature_index':i,'label_kind':kind,'expected':expected,'actual':actual,'existing':{'phone':phone}})
        ps = np.asarray([[.99,.01,0,0,0],[.8,.2,0,0,0],[.9,.1,0,0,0]])
        return features,labels,ps,np.ones(3,bool)

    def test_ambiguous_existing_warning_change_is_blocked_even_when_aggregate_feasible(self):
        f,labels,ps,av = self.fixture();result = screen(f,labels,ps,av)
        self.assertTrue(result['alternatives']['one_percent']['feasible'])
        self.assertFalse(result['strict_followup_qualified'])
        self.assertEqual(result['strict_pairwise_preservation']['actions_with_unknown_or_ambiguous_GT'],1)

    def test_safe_extra_error_and_normal_repair_keep_noinput_denominator(self):
        f,labels,ps,av = self.fixture();ps[2] = [.1,.9,0,0,0]
        labels.append({'feature_index':None,'label_kind':'pair_error','expected':'ㅅ','actual':'ㅆ','existing':{'phone':None}})
        result = screen(f,labels,ps,av)
        self.assertTrue(result['strict_followup_qualified'])
        self.assertEqual(result['baseline']['errors_all'],2)
        self.assertEqual(result['alternatives']['one_percent']['counts']['errors_all'],2)
        self.assertEqual(result['alternatives']['one_percent']['changes']['additional_exact'],1)
        self.assertEqual(result['strict_pairwise_preservation']['lost_real_error_warnings'],0)
        self.assertEqual(result['strict_pairwise_preservation']['lost_exact'],0)


if __name__=='__main__':unittest.main()

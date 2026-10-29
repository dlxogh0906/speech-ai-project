import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import unittest
import numpy as np
from mdd.h0_joint_evaluation import site_counts,gap_counts,calibrate,score_record


def site(status='normal',observed=True,score=2.,pred='ㅗ'):
    return dict(status=status,expected='ㅜ',actual='ㅜ' if status=='normal' else 'ㅗ',
                observed=observed,observation_failure=None if observed else 'annotation_boundary_unresolved',
                score=score,prediction_options=[pred])


class FixedDenominators(unittest.TestCase):
    def test_missing_error_remains_in_denominator(self):
        c=site_counts([site(),site(observed=False),site('substitution'),site('substitution',False)],1.)
        self.assertEqual([c[k] for k in ['normal','error','observed_normal','observed_error','fp','exact','fp_upper','exact_upper']],
                         [2,2,1,1,1,1,2,2])

    def test_calibration_does_not_budget_from_missing_normals(self):
        rows=[site(score=3.),site(score=3.)]+[site(pred='ㅜ') for _ in range(98)]+[site(observed=False) for _ in range(100)]
        c=calibrate([dict(sites=rows)])
        self.assertEqual((c['observed_normal_denominator'],c['budget'],c['actual_fp']),(100,1,0))
        self.assertGreater(c['threshold'],3.)

    def test_gap_exact_sequence_and_missing(self):
        c=gap_counts([dict(actual=[],predicted=['ㄱ']),dict(actual=['ㅗ','ㄹ'],predicted=['ㅗ']),
                      dict(actual=['ㅗ'],predicted=['ㅗ']),dict(actual=['ㄱ'],predicted=None)])
        self.assertEqual((c['error'],c['observed_error'],c['exact'],c['fp'],c['exact_upper']),(3,2,1,1,2))

    def test_truth_does_not_change_prediction_or_GOP(self):
        r=dict(uid='x',speaker='s',item='i',task='RW',language='x',role='cal',reference=['ㅜ'],actual=['ㅜ'],
               reference_owners=[dict(start=0.,end=.1)],actual_owners=[dict(start=0.,end=.1)],
               sites=[dict(q=0,expected='ㅜ',actual='ㅜ',status='normal')],annotation_mask=[None],truth_gaps=[None,None])
        lp=np.log(np.array([[.1,.7,.2],[.1,.7,.2],[.8,.1,.1]]));vocab={'ㅜ':1,'ㅗ':2}
        a=score_record(r,lp,vocab)
        r['actual']=['ㅗ'];r['sites']=[dict(q=0,expected='ㅜ',actual='ㅗ',status='substitution')]
        b=score_record(r,lp,vocab)
        self.assertEqual(a['predicted'],b['predicted']);self.assertEqual(a['sites'][0]['score'],b['sites'][0]['score'])


if __name__=='__main__':unittest.main()

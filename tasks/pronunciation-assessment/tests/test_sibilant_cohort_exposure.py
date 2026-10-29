"""Verify error/normal data factors without losing old cases or changing priors."""
import copy
import collections
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import numpy as np
from mdd.sibilant_cohort_exposure import ARMS,cohort_plans


class CohortTest(unittest.TestCase):
    def fixture(self):
        tokens = ['ㅅ']*4+['ㅆ']*4+['ㅅ','ㅆ']*4
        y = np.asarray([1,0,2,4,0,1,2,4,1,0,0,1,1,0,0,1])
        kinds = ['pair_error','correct','third_phone','deletion']*2+['pair_error']*2+['correct']*2+['pair_error']*2+['correct']*2
        f = [{'canonical':t,'q_runtime':0,'expected_V':'ㅏ'} for t in tokens]
        labs = {i:{'uid':str(i),'expected':t,'following':'ㅏ','label_kind':kinds[i]}
                for i,t in enumerate(tokens)}
        return np.arange(8).reshape(2,4),y,labs,f,[{'target':['x']} for _ in range(12)]

    def test_crossed_factors_retain_originals_and_match_unique_reuse_class_counts(self):
        original,y,labels,f,eligibility = self.fixture()
        plans,report = cohort_plans(original,y,labels,f,eligibility,12,3,steps=5,batch=4)
        for token,first,extra_error,extra_normal in [('ㅅ',0,12,14),('ㅆ',4,13,15)]:
            for arm in ARMS:
                plan = plans[arm][token].ravel()
                np.testing.assert_array_equal(plan[:4],np.arange(first,first+4))
                self.assertEqual(collections.Counter(y[plan].tolist()),collections.Counter(y[plans[ARMS[0]][token].ravel()].tolist()))
                self.assertEqual(int((plan==first+2).sum()),5)
                self.assertEqual(int((plan==first+3).sum()),5)
            full = plans[ARMS[-1]][token].ravel()
            self.assertIn(extra_error,full);self.assertIn(extra_normal,full)
            self.assertIn(first,full);self.assertIn(first+1,full)
            for error_factor in ('old_errors','new_errors'):
                a = plans[error_factor+'_old_normals'][token].ravel()
                b = plans[error_factor+'_new_normals'][token].ravel()
                error_ix = np.asarray([labels[int(i)]['label_kind']=='pair_error' for i in a])
                np.testing.assert_array_equal(a[error_ix],b[error_ix])
            self.assertEqual(report[token]['normal_features_added'],1)

    def test_no_room_or_unmatched_control_pool_is_not_silently_subsampled(self):
        original,y,labels,f,eligibility = self.fixture()
        with self.assertRaises(ValueError):
            cohort_plans(original,y,labels,f,eligibility,12,3,steps=1,batch=4)
        labels = copy.deepcopy(labels);labels[10]['label_kind'] = 'third_phone'
        with self.assertRaises(ValueError):
            cohort_plans(original,y,labels,f,eligibility,12,3,steps=5,batch=4)


if __name__=='__main__':unittest.main()

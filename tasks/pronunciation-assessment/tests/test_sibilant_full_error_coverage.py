"""Full new-error coverage may not be replaced by an easier sampled subset."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/train'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/data')]
import collections
import unittest
import numpy as np
from train_sibilant_all_partial_errors import full_error_plans


class FullCoverageTest(unittest.TestCase):
    def setUp(self):
        self.y = np.asarray([0,0,1,1,0,4,0,0,1,1])
        self.labels = {i:{'uid':str(i)+'-M-x','item':'i','language':'L','following':'ㅏ','position':'initial',
            'label_kind':'correct' if i==4 else 'deletion' if i==5 else 'pair_error',
            'expected':'ㅆ' if self.y[i]==0 else 'ㅅ','actual':'ㅅ' if self.y[i]==0 else 'ㅆ'} for i in range(10)}
        self.features = [{'canonical':self.labels[i]['expected'],'q_runtime':0} for i in range(10)]
        self.eligible = [{'target':['x']} for _ in range(6)];self.eligible[5]['target'] = None
        self.original = np.asarray([[0,2,4,5],[1,3,4,5],[0,2,4,5]])

    def test_all_new_errors_and_exact_repetition_with_untouched_nonerrors(self):
        (a,b),report = full_error_plans(self.original,self.y,self.labels,self.features,self.eligible,6,4)
        self.assertEqual(set(b[:,:2].ravel()),{6,7,8,9})
        self.assertEqual(sorted(collections.Counter(a[:,:2].ravel()).values()),sorted(collections.Counter(b[:,:2].ravel()).values()))
        np.testing.assert_array_equal(a[:,2],self.original[:,2]);np.testing.assert_array_equal(b[:,2],self.original[:,2])
        self.assertTrue((a[:,3]==-1).all() and (b[:,3]==-1).all())
        self.assertTrue(report['all_new_supported_errors_seen_at_least_once'])

    def test_insufficient_class_slots_does_not_silently_subsample_new_errors(self):
        with self.assertRaises(ValueError):full_error_plans(self.original[:1],self.y,self.labels,self.features,self.eligible,6,4)


if __name__ == '__main__':unittest.main()

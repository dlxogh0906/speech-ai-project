"""Paired new-error trials preserve normal identity and exact error repetition."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import collections
import copy
import unittest
import numpy as np
from mdd.sibilant_error_exposure import error_plans


class ExposureTest(unittest.TestCase):
    def setUp(self):
        self.y = np.asarray([1,1,0,4,1,1])
        self.labels = {i:{'uid':str(i)+'-M-x','item':'item','label_kind':'pair_error' if i in (0,1,4,5) else 'correct' if i==2 else 'deletion',
            'expected':'ㅅ','actual':'ㅆ','following':'ㅏ','position':'post_vowel','language':'L'} for i in range(6)}
        self.features = [{'q_runtime':7,'canonical':'ㅅ','expected_V':'ㅏ'} for _ in range(6)]
        self.eligibility = [{'target':['x']} for _ in range(4)];self.eligibility[3]['target'] = None
        self.original = np.asarray([[0,0,2,3],[1,0,2,3]])

    def test_unique_feature_bijection_and_nonerror_identity(self):
        (old,new),report = error_plans(self.original,self.y,self.labels,self.features,self.eligibility,4,3)
        np.testing.assert_array_equal(old[:,2],self.original[:,2]);np.testing.assert_array_equal(new[:,2],self.original[:,2])
        self.assertTrue((old[:,3]==-1).all() and (new[:,3]==-1).all())
        self.assertEqual(sorted(collections.Counter(old[:,:2].ravel()).values()),sorted(collections.Counter(new[:,:2].ravel()).values()))
        self.assertEqual(report['changed_pair_draws'],4)
        self.assertTrue((new[:,:2]>=4).all())

    def test_unmatched_L1_or_sex_does_not_force_new_error_draws(self):
        labels = copy.deepcopy(self.labels)
        for i in (4,5):labels[i]['language'] = 'other'
        (old,new),report = error_plans(self.original,self.y,labels,self.features,self.eligibility,4,3)
        self.assertEqual(report['changed_pair_draws'],0)
        np.testing.assert_array_equal(old,new)
        labels = copy.deepcopy(self.labels)
        for i in (4,5):labels[i]['uid'] = str(i)+'-F-x'
        self.assertEqual(error_plans(self.original,self.y,labels,self.features,self.eligibility,4,3)[1]['changed_pair_draws'],0)

    def test_unknown_supported_original_class_is_rejected(self):
        y = self.y.copy();y[0] = -100
        with self.assertRaises(ValueError):error_plans(self.original,y,self.labels,self.features,self.eligibility,4,3)


if __name__ == '__main__':unittest.main()

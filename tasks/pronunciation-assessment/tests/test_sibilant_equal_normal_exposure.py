"""Unequal source pool sizes must yield identical normal reuse patterns."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import unittest
import collections
import numpy as np
from train_sibilant_equal_normal_exposure import equal_plans


class EqualExposureTest(unittest.TestCase):
    def test_unequal_pools_bijective_reuse_and_error_retention(self):
        features = [{'canonical':'ㅅ'} for _ in range(9)]
        labels = {i:{'uid':str(i),'q':0,'expected':'ㅅ','following':'ㅣ',
            'label_kind':'pair_error' if i == 0 else 'correct'} for i in range(9)}
        rows = {str(i):{'item':'item','utterance_all_correct':i != 0,
            'sites':[{'label_kind':labels[i]['label_kind']}]} for i in range(9)}
        target = np.array([1]+[0]*8)
        original = np.array([[0,1,2,3,4,5],[0,5,4,3,2,1]])
        plans, changed = equal_plans(original,target,labels,features,rows,
            [{'target':True} for _ in range(6)],6,7)
        self.assertEqual(changed,10)
        np.testing.assert_array_equal(plans[0][:,0],plans[1][:,0])
        old,new = [collections.Counter(plan[:,1:].ravel()) for plan in plans]
        self.assertEqual(sorted(old.values()),sorted(new.values()))
        self.assertLessEqual(len(old),3)
        mapping = {}
        for a,b in zip(plans[0][:,1:].ravel(),plans[1][:,1:].ravel()):
            self.assertEqual(mapping.setdefault(a,b),b)


if __name__ == '__main__':
    unittest.main()

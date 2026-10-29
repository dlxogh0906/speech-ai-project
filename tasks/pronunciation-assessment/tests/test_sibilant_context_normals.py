"""Context balancing must preserve all other exposures and retain hard errors."""
import sys
import unittest
import collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import numpy as np
from mdd.sibilant_context_normals import ARMS,normal_context_plans
from mdd.sibilant_source_shift import context


class ContextNormalTest(unittest.TestCase):
    def fixture(self):
        tokens = ['ㅅ']*4+['ㅆ']*4+['ㅅ','ㅆ']*6
        y = np.asarray([1,0,2,4,0,1,2,4,1,0,0,1,1,0,0,1,0,1,0,1])
        kinds = ['pair_error','correct','third_phone','deletion']*2+['pair_error']*2+['correct']*2+['pair_error']*2+['correct']*6
        fs = [{'canonical':t,'q_runtime':0,'expected_V':'ㅏ'} for t in tokens]
        labs = {i:{'uid':str(i)+'-M-x','expected':t,'following':'ㅏ','label_kind':kinds[i],
            'item':'B' if i in (14,15,18,19) else 'A','position':'initial','language':'L'} for i,t in enumerate(tokens)}
        return np.arange(8).reshape(2,4),y,labs,fs,[{'target':['x']} for _ in range(12)]

    def test_context_normals_cover_error_sites_and_error_identity_class_reuse_are_fixed(self):
        original,y,labels,fs,eligible = self.fixture()
        plans,report = normal_context_plans(original,y,labels,fs,eligible,12,16,3,
            sizes=(1,1),steps=5,batch=4,minimum_coverage=1.)
        self.assertEqual(report['normal_context_coverage'],1.)
        for token,first,matched in [('ㅅ',0,16),('ㅆ',4,17)]:
            baseline = plans[ARMS[0]][token].ravel()
            for arm in ARMS:
                values = plans[arm][token].ravel()
                np.testing.assert_array_equal(values[:4],np.arange(first,first+4))
                self.assertEqual(collections.Counter(y[values].tolist()),collections.Counter(y[baseline].tolist()))
                self.assertEqual(int((values==first+2).sum()),5)
                self.assertEqual(int((values==first+3).sum()),5)
            for arm in ARMS[2:]:
                a,b = [plans[name][token].ravel() for name in (ARMS[1],arm)]
                mask = np.asarray([labels[int(i)]['label_kind']=='pair_error' for i in a])
                np.testing.assert_array_equal(a[mask],b[mask])
            self.assertIn(matched,plans[ARMS[-1]][token])
            self.assertEqual(context(labels[matched],fs[matched]),context(labels[first],fs[first]))

    def test_missing_matched_context_is_not_replaced_with_a_different_context(self):
        original,y,labels,fs,eligible = self.fixture();labels[16]['item']='B'
        with self.assertRaises(ValueError):
            normal_context_plans(original,y,labels,fs,eligible,12,16,3,
                sizes=(1,1),steps=5,batch=4,minimum_coverage=1.)


if __name__=='__main__':unittest.main()

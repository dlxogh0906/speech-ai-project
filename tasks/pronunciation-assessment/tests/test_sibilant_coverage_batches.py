import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdd.sibilant_context_batches import make_plan as original_plan
from mdd.sibilant_coverage_batches import make_plan


class CoverageBatchesTest(unittest.TestCase):
    def fixture(self):
        y,labels,features=[],[],[]
        for expected in ('ㅅ','ㅆ'):
            for actual,token in enumerate(('ㅅ','ㅆ')):
                for n in range(30):
                    i=len(y);y.append(actual);features.append({'canonical':expected})
                    labels.append(dict(feature_index=i,item=expected,q=0,expected=expected,following='ㅣ',actual=token,
                                       label_kind='correct' if expected==token else 'pair_error',alignment_ambiguous=False))
        for cls in (2,3):
            for n in range(5):y.append(cls);features.append({'canonical':'ㅅ'})
        return np.array(y),labels,features

    def test_same_contexts_quotas_other_draws_and_determinism(self):
        y,l,f=self.fixture();old,keys,_=original_plan(y,l,f,42,20,'paired_context');new,nkeys,r=make_plan(y,l,f,42,20,'paired_context')
        self.assertEqual(keys,nkeys)
        np.testing.assert_array_equal(y[old],y[new]);np.testing.assert_array_equal(old[:,8:],new[:,8:])
        np.testing.assert_array_equal(new,make_plan(y,l,f,42,20,'paired_context')[0])
        self.assertGreater(r['candidate_coverage']['unique_all'],r['control_coverage']['unique_all'])

    def test_no_repeat_inside_each_context_class_cycle(self):
        y,l,f=self.fixture();plan,keys,_=make_plan(y,l,f,42,20,'paired_context')
        for expected in ('ㅅ','ㅆ'):
            for cls in (0,1):
                draws=[int(plan[s,j+cls*4]) for s,contexts in enumerate(keys) for j,k in enumerate(contexts) if k[2]==expected]
                self.assertEqual(len(set(draws[:30])),30)


if __name__=='__main__':unittest.main()

import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from mdd.h0_p6_feasibility import certificate,solve_qp,solve_interior
from design_h0_p6_holdouts import families,trie_add,trie_matches


class FeasibilityTests(unittest.TestCase):
    def test_lower_bound_arbitrary_dual_and_infeasible_iterate(self):
        Q=sparse.eye(2,format='csr');t=np.array([0.,3.]);A=sparse.csr_matrix([[1.,1.]])
        l=u=np.array([2.]);opt=.25**2+(1.75-3)**2
        for w in (np.array([1.,1.]),np.array([8.,-4.])):
            for y in (-10.,0.,10.):
                c=certificate(Q,t,w,A,l,u,np.array([y]))
                self.assertLessEqual(c['lower_bound'],opt+1e-10)
        self.assertFalse(c['numerically_feasible'])

    def test_sparse_qp_matches_independent_slsqp(self):
        rng=np.random.RandomState(41);Q=sparse.csr_matrix(rng.normal(size=(7,5)));t=rng.normal(size=7)
        A=sparse.csr_matrix([[1,1,1,1,1],[1,0,0,0,0],[0,1,1,0,0]])
        l=np.array([5.,.8,1.]);u=np.array([5.,1.2,2.2])
        w,y,c=solve_qp(Q,t,A,l,u,np.ones(5))
        f=lambda x:float(sum((Q.dot(x)-t)**2))
        ref=minimize(f,np.ones(5),method='SLSQP',bounds=[(.25,4.)]*5,
            constraints=[dict(type='eq',fun=lambda x:x.sum()-5),
                         dict(type='ineq',fun=lambda x:np.r_[A.dot(x)[1:]-l[1:],u[1:]-A.dot(x)[1:]])],
            options=dict(ftol=1e-12,maxiter=1000))
        self.assertTrue(ref.success);self.assertTrue(c['certified_to_tolerance'])
        self.assertAlmostEqual(f(w),ref.fun,places=7)
        self.assertLessEqual(c['lower_bound'],ref.fun+1e-8)
        wi,yi,ci=solve_interior(Q,t,A,l,u)
        self.assertTrue(ci['certified_to_tolerance'])
        self.assertAlmostEqual(f(wi),ref.fun,places=7)

    def test_negative_gap_from_infeasible_point_is_not_convergence(self):
        w=np.array([1-1.5e-6]);Q=sparse.csr_matrix([[100.]])
        t=np.array([99.9]);gradient=2*Q.T.dot(Q.dot(w)-t)
        c=certificate(Q,t,w,sparse.csr_matrix([[1.]]),np.array([1.]),np.array([np.inf]),-gradient)
        self.assertTrue(c['numerically_feasible'])
        self.assertLess(c['upper_minus_lower'],-2e-5)
        self.assertFalse(c['certified_to_tolerance'])

    def test_text_families_and_overlapping_matches(self):
        groups,edges=families({'a':'소리','b':'소리가난다','c':'난다','d':'별개'})
        self.assertIn(['a','b','c'],groups);self.assertIn(['d'],groups)
        trie={}
        for word in ('소리','소리가','리가'):trie_add(trie,word,word)
        self.assertEqual(trie_matches(trie,'소리가'),{'소리','소리가','리가'})

    def test_long_shared_substring_is_one_holdout_family(self):
        common='가나다라마바사아자차카타'
        grouped,_=families({'a':'앞'+common,'b':common+'뒤','c':'다른문장'})
        self.assertIn(['a','b'],grouped)

if __name__=='__main__':unittest.main()

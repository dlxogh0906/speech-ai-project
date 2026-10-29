"""Independent constrained solver check for TRAIN design projection."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tasks/pronunciation-assessment/scripts/eval'))
from prepare_h0_p5_constrained_design import project,project_caps

class ProjectionTest(unittest.TestCase):
    def test_caps_against_independent_quadratic_solver(self):
        c=np.array([.2,.4,.9,.7,.3,.6]);groups=np.array([0,0,1,1,2,2]);sg=np.array([0,0,0]);target=np.array([c.sum()]);caps=np.array([1.1,1.4,1.2]);v=np.array([-2,3,7,-1,2,5.])
        w=project_caps(v,c,groups,caps,sg,target)
        fun=lambda x:.5*np.dot(c,(x-v)**2)
        constraints=[dict(type='eq',fun=lambda x:np.dot(c,x)-target[0])]
        for i in range(3):constraints.append(dict(type='ineq',fun=lambda x,i=i:caps[i]-sum(c[groups==i]*x[groups==i])))
        ref=minimize(fun,np.ones(6),method='SLSQP',bounds=[(.25,4)]*6,constraints=constraints,options={'ftol':1e-11,'maxiter':1000})
        self.assertTrue(ref.success);self.assertAlmostEqual(fun(w),ref.fun,places=8)
        np.testing.assert_allclose(project_caps(w,c,groups,caps,sg,target),w,atol=1e-9)

    def test_exact_prompt_projection_preserves_every_stratum(self):
        rng=np.random.RandomState(17);c=rng.uniform(.01,1,1000);g=np.arange(1000)%7;t=np.bincount(g,weights=c);v=rng.normal(1,5,1000)
        w=project(v,c,g,t);np.testing.assert_allclose(np.bincount(g,weights=c*w),t,atol=1e-8)
        self.assertGreaterEqual(w.min(),.25);self.assertLessEqual(w.max(),4)
        np.testing.assert_allclose(project(w,c,g,t),w,atol=1e-10)

if __name__=='__main__':unittest.main()

"""Checks that marginal conservation cannot hide joint support loss."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'tasks/pronunciation-assessment/src')]
import unittest
import numpy as np
from scipy import sparse
from mdd.h0_joint_weighting import append_joint_constraints, joint_metrics, repair_tiny_residual, weighted_ctc_loss, repaired_certificate
from mdd.h0_p6_feasibility import certificate, solve_qp


class JointSupportTest(unittest.TestCase):
    def fixture(self):
        source = sparse.csr_matrix([[.5, .5, 0, 0], [0, 0, .5, .5]])
        speaker = sparse.csr_matrix([[.5, 0, .5, 0], [0, .5, 0, .5]])
        matrices = {'phone_source': source, 'speaker_source': sparse.eye(4, format='csr')}
        return source, speaker, matrices

    def test_joint_loss_hidden_by_marginals(self):
        source, speaker, m = self.fixture()
        w = np.array([.25, 1.75, 1.75, .25])
        np.testing.assert_array_equal(source.dot(w), [1, 1])
        np.testing.assert_array_equal(speaker.dot(w), [1, 1])
        result = joint_metrics(m, w)
        self.assertEqual(result['speaker_source']['outside'], 2)
        self.assertFalse(result['speaker_source']['passed'])
        A, lo, hi = append_joint_constraints(source, np.ones(2), np.ones(2), m)
        self.assertGreater(np.max(lo-A.dot(w)), 0)

    def test_solve_enforces_joint_bounds(self):
        source, _, m = self.fixture()
        A, lo, hi = append_joint_constraints(source, np.ones(2), np.ones(2), m)
        target = np.array([.25, 1.75, 1.75, .25])
        Q = sparse.eye(4, format='csr')
        w, y, c = solve_qp(Q, target, A, lo, hi, np.ones(4), max_iter=10000)
        self.assertTrue(c['certified_to_tolerance'])
        np.testing.assert_allclose(w, [.5, 1.5, 1.5, .5], atol=2e-6)
        self.assertTrue(all(x['passed'] for x in joint_metrics(m, w).values()))
        self.assertGreater(c['lower_bound'], .24)

    def test_tiny_repair_preserves_source_and_joint_support(self):
        source, _, m = self.fixture()
        A, lo, hi = append_joint_constraints(source, np.ones(2), np.ones(2), m)
        w = np.array([.5-1e-7, 1.5, 1.5+1e-7, .5])
        fixed, report = repair_tiny_residual(w, source, A, lo, hi)
        np.testing.assert_allclose(source.dot(fixed), [1, 1], atol=1e-14)
        self.assertTrue(np.all(A.dot(fixed) >= lo-1e-14))
        self.assertTrue(np.all(A.dot(fixed) <= hi+1e-14))
        self.assertLess(report['max_weight_change'], 1e-6)

    def test_loss_keeps_declared_microbatch_denominator(self):
        import torch
        torch.manual_seed(43)
        logits = torch.randn(2, 8, 4, requires_grad=True)
        lengths = torch.tensor([8, 7]); targets = [[1, 2], [2, 2]]
        one = weighted_ctc_loss(logits, lengths, targets, [1., 1.])
        twice = weighted_ctc_loss(logits, lengths, targets, [2., 2.])
        self.assertEqual(float(twice), 2*float(one))
        first = weighted_ctc_loss(logits[:1], lengths[:1], targets[:1], [1.])
        second = weighted_ctc_loss(logits[1:], lengths[1:], targets[1:], [1.])
        mixed = weighted_ctc_loss(logits, lengths, targets, [.5, 1.5])
        torch.testing.assert_close(mixed, (.5*first+1.5*second)/2)
        g = torch.autograd.grad(mixed, logits, retain_graph=True)[0]
        reference = torch.autograd.grad((.5*first+1.5*second)/2, logits)[0]
        torch.testing.assert_close(g, reference)

    def test_global_bound_can_use_infeasible_anchor(self):
        Q=sparse.eye(1,format='csr');A=sparse.eye(1,format='csr')
        # min x^2 on x >= 1 has optimum 1; the anchor is slightly infeasible.
        anchor=np.array([1.-1e-7]);point=np.array([1.])
        c=repaired_certificate(Q,np.zeros(1),point,anchor,A,np.ones(1),np.array([2.]),np.array([-2.]))
        self.assertLessEqual(c['lower_bound'],1.+1e-14)
        self.assertTrue(c['certified_to_tolerance'])
        self.assertEqual(c['max_constraint_violation'],0.)


if __name__ == '__main__':
    unittest.main()

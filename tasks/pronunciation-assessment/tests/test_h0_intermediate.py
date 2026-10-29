import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT/'tasks/pronunciation-assessment/src')]
import unittest
import torch
from mdd.h0_intermediate import IntermediateHeads, rng_state, restore_rng


class IntermediateTests(unittest.TestCase):
    def test_identity_all_arms_and_no_random_forward(self):
        for arm in ('B0','B1','B2','B3'):
            m=IntermediateHeads(torch.nn.Linear(8,5),arm)
            x,y=torch.randn(2,7,8),torch.randn(2,7,8)
            state=torch.get_rng_state().clone()
            z,aux=m(x,y,True)
            self.assertTrue(torch.equal(y,z))
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertEqual(aux is not None,arm in ('B1','B3'))

    def test_auxiliary_gradient_reaches_middle(self):
        m=IntermediateHeads(torch.nn.Linear(8,5),'B1')
        x,y=torch.randn(2,7,8,requires_grad=True),torch.randn(2,7,8,requires_grad=True)
        z,aux=m(x,y,True); aux.square().mean().backward()
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertIsNone(y.grad)

    def test_zero_gate_learns_without_initial_middle_leak(self):
        m=IntermediateHeads(torch.nn.Linear(8,5),'B2')
        x,y=torch.randn(2,7,8,requires_grad=True),torch.randn(2,7,8,requires_grad=True)
        z,_=m(x,y); (z*y.detach()).sum().backward()
        self.assertGreater(abs(float(m.gate.grad)),0.)
        self.assertEqual(float(x.grad.abs().max()),0.)
        with torch.no_grad(): m.gate.fill_(.4)
        x.grad.zero_(); z,_=m(x,y); z.square().mean().backward()
        self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_state_roundtrip_and_rng_replay(self):
        import io
        m=IntermediateHeads(torch.nn.Linear(8,5),'B3')
        with torch.no_grad(): m.gate.fill_(.7)
        x,y=torch.randn(2,7,8),torch.randn(2,7,8)
        buf=io.BytesIO(); torch.save(m.state_dict(),buf); buf.seek(0)
        n=IntermediateHeads(torch.nn.Linear(8,5),'B3'); n.load_state_dict(torch.load(buf))
        self.assertTrue(torch.equal(m(x,y)[0],n(x,y)[0]))
        s=rng_state(); a=torch.rand(11); restore_rng(s); self.assertTrue(torch.equal(a,torch.rand(11)))


if __name__=='__main__': unittest.main()

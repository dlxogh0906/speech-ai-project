import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_c_trajectory import CTrajectoryReadout, c_summary


class CTrajectoryTest(unittest.TestCase):
    def test_zero_branch_reproduces_existing_head_exactly(self):
        torch.manual_seed(12)
        old = SupportReadout(7, True)
        x, mask, relative = torch.randn(2, 9, 7), torch.ones(2, 9, dtype=torch.bool), torch.randn(2, 9)
        expected = old(x, mask, relative)
        for branch in ("mean", "drift"):
            new = CTrajectoryReadout(7, branch)
            new.load_control(old.state_dict())
            actual = new(x, mask, relative)
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_direction_is_retained_and_padding_cannot_change_it(self):
        mask = torch.tensor([[True, True, True, True, False]])
        x = torch.tensor([[[0.], [1.], [2.], [3.], [9999.]]])
        phase = torch.zeros(1, 5, 4)
        summary, mean, drift = c_summary(x, mask, phase)
        self.assertAlmostEqual(float(mean[0]), 1.5)
        self.assertGreater(float(drift[0]), 0.)
        x[0, :4] = x[0, :4].flip(0)
        _, reverse_mean, reverse = c_summary(x, mask, phase)
        self.assertTrue(torch.equal(mean, reverse_mean))
        self.assertAlmostEqual(float(drift[0]), -float(reverse[0]))
        x[0, 4] = -9999.
        self.assertTrue(torch.equal(reverse, c_summary(x, mask, phase)[2]))

    def test_extra_branch_has_finite_trainable_gradient(self):
        head = CTrajectoryReadout(4, "drift")
        x = torch.randn(2, 8, 4, requires_grad=True)
        logits, roles = head(x, torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8))
        (logits.square().sum() + roles.square().mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(head.extra.weight.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()

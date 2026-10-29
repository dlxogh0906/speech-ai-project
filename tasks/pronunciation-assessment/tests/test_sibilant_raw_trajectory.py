import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_raw_trajectory import RawTrajectoryReadout


class RawTrajectoryTest(unittest.TestCase):
    def test_zero_skip_reproduces_control_and_raw_can_learn(self):
        torch.manual_seed(31)
        old = SupportReadout(6, True)
        h, x = torch.randn(2, 9, 6), torch.randn(2, 9, 4, requires_grad=True)
        mask, rel = torch.ones(2, 9, dtype=torch.bool), torch.randn(2, 9)
        for branch in ("mean", "drift"):
            head = RawTrajectoryReadout(6, branch, 4)
            head.load_control(old.state_dict())
            logits, roles = head(h, mask, rel, x)
            baseline = old(h, mask, rel)
            self.assertTrue(torch.equal(logits, baseline[0]))
            self.assertTrue(torch.equal(roles, baseline[1]))
            logits.square().sum().backward(retain_graph=True)
            self.assertGreater(float(head.extra.weight.grad.abs().sum()), 0.)

    def test_time_grid_and_dimensions_are_checked(self):
        head = RawTrajectoryReadout(6, "drift", 4)
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 8, 6), torch.ones(1, 8, dtype=torch.bool), torch.zeros(1, 8), torch.zeros(1, 7, 4))
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 8, 6), torch.ones(1, 8, dtype=torch.bool), torch.zeros(1, 8), torch.zeros(1, 8, 3))


if __name__ == "__main__":
    unittest.main()

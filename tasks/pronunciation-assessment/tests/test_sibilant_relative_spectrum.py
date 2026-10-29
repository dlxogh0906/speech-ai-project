import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_relative_spectrum import RelativeSpectrumReadout


class RelativeSpectrumTest(unittest.TestCase):
    def test_zero_extra_reproduces_original_and_can_learn(self):
        torch.manual_seed(37)
        old = SupportReadout(6, True)
        h, x = torch.randn(2, 9, 6), torch.randn(2, 9, 4)
        mask, rel = torch.ones(2, 9, dtype=torch.bool), torch.randn(2, 9)
        for branch in ("mean", "relative"):
            head = RelativeSpectrumReadout(6, branch, torch.zeros(4), torch.ones(4))
            head.load_control(old.state_dict())
            logits, roles = head(h, mask, rel, x)
            baseline = old(h, mask, rel)
            self.assertTrue(torch.equal(logits, baseline[0]))
            self.assertTrue(torch.equal(roles, baseline[1]))
            logits.square().sum().backward()
            self.assertGreater(float(head.extra.weight.grad.abs().sum()), 0.)

    def test_relative_spectrum_cancels_constant_offset_and_excludes_padding(self):
        torch.manual_seed(41)
        head = RelativeSpectrumReadout(6, "relative", torch.zeros(4), torch.ones(4))
        torch.nn.init.normal_(head.extra.weight, std=.1)
        h, x = torch.randn(1, 9, 6), torch.randn(1, 9, 4)
        mask = torch.tensor([[True] * 7 + [False] * 2])
        rel = torch.randn(1, 9)
        original = head(h, mask, rel, x)[0]
        changed = x + torch.tensor([1., -2., 3., 4.])
        changed[:, 7:] = 1e4
        torch.testing.assert_close(original, head(h, mask, rel, changed)[0], atol=2e-6, rtol=1e-6)

    def test_invalid_acoustic_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            RelativeSpectrumReadout(6, "relative", torch.zeros(4), torch.zeros(4))
        with self.assertRaises(ValueError):
            RelativeSpectrumReadout(6, "relative", torch.tensor([float("nan")]), torch.ones(1))


if __name__ == "__main__":
    unittest.main()

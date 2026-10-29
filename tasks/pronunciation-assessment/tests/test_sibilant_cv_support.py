import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import torch
from mdd.sibilant_cv_support import SupportReadout, phase_loss, reliable_support


class SupportTests(unittest.TestCase):
    def test_inference_has_no_annotation_input_and_ignores_padding(self):
        torch.manual_seed(1)
        head = SupportReadout(4, True)
        x = torch.randn(1, 9, 4, requires_grad=True)
        mask = torch.tensor([[True] * 6 + [False] * 3])
        relative = torch.linspace(-1, 1, 9)[None]
        expected, _ = head(x, mask, relative)
        other = x.detach().clone()
        other[:, 6:] = 100
        actual, _ = head(other, mask, relative)
        torch.testing.assert_close(expected, actual)
        expected.sum().backward()
        self.assertEqual(float(x.grad[:, 6:].abs().sum()), 0.)

    def test_unknown_phase_is_not_background_supervision(self):
        logits = torch.randn(2, 5, 4, requires_grad=True)
        loss = phase_loss(logits, torch.full((2, 5), -100, dtype=torch.long))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_ignored_boundaries_receive_no_phase_gradient(self):
        x = torch.zeros(1, 4, 4, requires_grad=True)
        phase_loss(x, torch.tensor([[1, -100, 2, 3]])).backward()
        self.assertEqual(float(x.grad[0, 1].abs().sum()), 0.)
        self.assertGreater(float(x.grad[0, 0].abs().sum()), 0.)

    def test_missing_or_reversed_roles_abstain(self):
        logits = torch.full((1, 3, 4), -10.)
        for i in range(3):
            logits[0, i, i + 1] = 10.
        mask = torch.ones(1, 3, dtype=torch.bool)
        self.assertTrue(bool(reliable_support(logits, mask)[0]))
        self.assertFalse(bool(reliable_support(logits.flip(1), mask)[0]))
        mask[0, 2] = False
        self.assertFalse(bool(reliable_support(logits, mask)[0]))


if __name__ == "__main__":
    unittest.main()

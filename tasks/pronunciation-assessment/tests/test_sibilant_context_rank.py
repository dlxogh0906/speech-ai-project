"""Checks for the actual training objective, without model/audio dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import torch
from mdd.sibilant_context_rank import context_rank_loss


class ContextRankTest(unittest.TestCase):
    def test_context_offset_cancels_and_gradient_uses_both_recordings(self):
        actual = torch.arange(4).repeat_interleave(4)
        logits = torch.zeros((16, 4), requires_grad=True)
        loss = context_rank_loss(logits, actual)
        loss.backward()
        self.assertTrue(bool((logits.grad[:4, 0] < 0).all()))
        self.assertTrue(bool((logits.grad[4:8, 1] < 0).all()))
        self.assertEqual(float(logits.grad[8:].abs().sum()), 0.)
        offset = torch.tensor([-.8, .2, 1.4, 2.])
        shifted = logits.detach().clone()
        shifted[:4, 1] += offset
        shifted[4:8, 1] += offset
        torch.testing.assert_close(context_rank_loss(shifted, actual), loss)
        separated = logits.detach().clone()
        separated[:4, 0], separated[4:8, 1] = 1., 1.
        self.assertLess(float(context_rank_loss(separated, actual)), float(loss))

    def test_invalid_pair_order_rejected(self):
        actual = torch.arange(4).repeat_interleave(4)
        actual[0] = 1
        with self.assertRaises(ValueError):
            context_rank_loss(torch.zeros((16, 4)), actual)


if __name__ == "__main__":
    unittest.main()

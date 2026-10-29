import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import torch
from mdd.sibilant_local_encoder import TemporalReadout


class TemporalReadoutTests(unittest.TestCase):
    def test_padding_does_not_change_scores_or_receive_gradient(self):
        torch.manual_seed(3)
        model = TemporalReadout(4)
        hidden = torch.randn(2, 12, 4, requires_grad=True)
        mask = torch.tensor([[1] * 12, [1] * 7 + [0] * 5], dtype=torch.bool)
        expected = model(hidden, mask)
        other = hidden.detach().clone()
        other[1, 7:] = 1000
        torch.testing.assert_close(model(other, mask), expected)
        expected.sum().backward()
        self.assertEqual(float(hidden.grad[1, 7:].abs().sum()), 0.)

    def test_missing_temporal_support_rejected(self):
        model = TemporalReadout(4)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 6, 4), torch.zeros(1, 6, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 6, 4), torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

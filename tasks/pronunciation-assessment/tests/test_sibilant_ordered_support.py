import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import itertools
import unittest
import torch
from mdd.sibilant_ordered_support import ordered_phase_probabilities, OrderedSupportReadout
from mdd.sibilant_cv_support import SupportReadout


class OrderedSupportTest(unittest.TestCase):
    def test_matches_enumerated_valid_paths(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 5, 4)
        p = logits.double().softmax(-1)[0]
        expected = torch.zeros((5, 4), dtype=torch.float64)
        total = 0.
        roles = [0, 1, 2, 3, 0]
        for path in itertools.product(range(5), repeat=5):
            if path[0] not in (0, 1) or path[-1] not in (3, 4) or any(b not in (a, a + 1) for a, b in zip(path, path[1:])):
                continue
            prob = float(torch.stack([p[i, roles[s]] for i, s in enumerate(path)]).prod())
            total += prob
            for i, s in enumerate(path):
                expected[i, roles[s]] += prob
        expected /= total
        actual = ordered_phase_probabilities(logits, torch.ones((1, 5), dtype=torch.bool))[0]
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)

    def test_padding_cannot_supply_events_and_gradients_remain_finite(self):
        torch.manual_seed(2)
        logits = torch.randn((2, 8, 4), requires_grad=True)
        mask = torch.arange(8)[None] < torch.tensor([5, 8])[:, None]
        actual = ordered_phase_probabilities(logits, mask)
        short = ordered_phase_probabilities(logits[:1, :5], mask[:1, :5])
        torch.testing.assert_close(actual[:1, :5], short)
        self.assertEqual(float(actual[0, 5:].abs().sum()), 0.)
        torch.testing.assert_close(actual.sum(-1), mask.float(), atol=2e-6, rtol=1e-5)
        actual.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertEqual(float(logits.grad[0, 5:].abs().sum()), 0.)

    def test_parameter_and_uniform_control_are_unchanged(self):
        torch.manual_seed(3)
        original = SupportReadout(4, False)
        ordered = OrderedSupportReadout(4, False)
        ordered.load_state_dict(original.state_dict())
        h = torch.randn((2, 7, 4))
        mask = torch.arange(7)[None] < torch.tensor([5, 7])[:, None]
        relative = torch.randn((2, 7))
        for a, b in zip(original(h, mask, relative), ordered(h, mask, relative)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

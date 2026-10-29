import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_prior_risk import class_mass_weights, actual_class_loss


class ClassMassTest(unittest.TestCase):
    def test_fixed_draw_weighted_loss_matches_population_class_mass(self):
        # Same class-conditional loss, different empirical replay proportions.
        y = np.array([0] * 8 + [1] * 2 + [-1])
        drawn = [0, 0, 8, 8]
        weights, population, draws = class_mass_weights(y, drawn, classes=2)
        np.testing.assert_array_equal(population, [8, 2])
        np.testing.assert_array_equal(draws, [2, 2])
        logits = torch.tensor([[2., -1.], [2., -1.], [1., -2.], [1., -2.]], requires_grad=True)
        target = torch.tensor([0, 0, 1, 1])
        actual = actual_class_loss(logits, target, weights)
        per_class = torch.nn.functional.cross_entropy(logits[[0, 2]], torch.tensor([0, 1]), reduction='none')
        expected = .8 * per_class[0] + .2 * per_class[1]
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(logits.grad[2:].abs().sum()), 0.)

    def test_control_is_exact_original_cross_entropy(self):
        logits, target = torch.randn(8, 5), torch.tensor([0, 1, 2, 3, 4, 0, 0, 1])
        self.assertTrue(torch.equal(actual_class_loss(logits, target),
                                    torch.nn.functional.cross_entropy(logits, target)))

    def test_unknown_draw_missing_class_and_nonpositive_weights_reject(self):
        for y, drawn in [([0, 1, -1], [2]), ([0, 1], [0]), ([0, 1], [-1])]:
            with self.assertRaises(ValueError):
                class_mass_weights(y, drawn, classes=2)
        with self.assertRaises(ValueError):
            actual_class_loss(torch.zeros(1, 2), torch.tensor([0]), [1., 0.])


if __name__ == '__main__':
    unittest.main()

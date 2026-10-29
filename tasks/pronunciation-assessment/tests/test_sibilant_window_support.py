import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import numpy as np
import torch
from mdd.sibilant_window_support import stacked_support_kernel, window_role_support, balanced_support_loss
from mdd.sibilant_cv_support import phase_loss


class WindowSupportTest(unittest.TestCase):
    def test_known_singleton_and_mixed_support(self):
        kernel = stacked_support_kernel(np.ones(400))
        roles = np.ones(10005, np.int64)
        value = window_role_support(roles, 0, 10005, 30, kernel)
        np.testing.assert_array_equal(value[:, 1], np.ones(30))
        roles[:280] = 2
        value = window_role_support(roles, 0, 10005, 30, kernel)
        self.assertAlmostEqual(float(value[0, 1]), .5)
        self.assertAlmostEqual(float(value[0, 2]), .5)
        np.testing.assert_allclose(value.sum(-1), 1.)
        roles[:] = -100
        self.assertFalse(window_role_support(roles, 0, 10005, 30, kernel).any())

    def test_one_hot_loss_matches_existing_loss_and_gradients(self):
        torch.manual_seed(71)
        logits = torch.randn(2, 7, 4, requires_grad=True)
        target = torch.tensor([[0, 1, 2, 3, -100, 1, 2], [1, 2, 3, 0, 1, -100, 0]])
        support = torch.nn.functional.one_hot(target.clamp(min=0), 4).float()
        support[target == -100] = 0
        a, b = phase_loss(logits, target), balanced_support_loss(logits, support)
        torch.testing.assert_close(a, b)
        ga = torch.autograd.grad(a, logits, retain_graph=True)[0]
        gb = torch.autograd.grad(b, logits)[0]
        torch.testing.assert_close(ga, gb)

    def test_unknown_loss_is_zero_and_bad_mass_is_rejected(self):
        logits = torch.randn(2, 7, 4, requires_grad=True)
        loss = balanced_support_loss(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))
        with self.assertRaises(ValueError):
            balanced_support_loss(logits, torch.ones_like(logits))


if __name__ == '__main__':
    unittest.main()

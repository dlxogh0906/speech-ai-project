import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import torch
from mdd.sibilant_training_mask import masked_actual_loss


class TrainingMaskTest(unittest.TestCase):
    def test_all_valid_is_exact_original_loss_and_gradient(self):
        torch.manual_seed(101)
        x, y = torch.randn(16, 4, requires_grad=True), torch.arange(4).repeat_interleave(4)
        a = torch.nn.functional.cross_entropy(x, y)
        b = masked_actual_loss(x, y, torch.ones(16, dtype=torch.bool))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(torch.autograd.grad(a, x, retain_graph=True)[0], torch.autograd.grad(b, x)[0]))

    def test_unsupported_label_has_no_class_gradient_and_families_keep_equal_weight(self):
        x, y = torch.randn(8, 4, requires_grad=True), torch.arange(4).repeat_interleave(2)
        valid = torch.tensor([False, True, True, True, True, True, True, True])
        loss = masked_actual_loss(x, y, valid)
        expected = torch.stack([torch.nn.functional.cross_entropy(x[(y == k) & valid], y[(y == k) & valid]) for k in range(4)]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(x.grad[0].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()

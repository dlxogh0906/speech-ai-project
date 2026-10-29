import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import torch
from mdd.sibilant_joint_vowel import JointVowelReadout


class VowelProgressPoolTest(unittest.TestCase):
    def test_uniform_vowel_mass_retains_earlier_later_change_without_padding(self):
        head = JointVowelReadout(1, 'vowel_progress')
        for value in head.support.parameters():
            torch.nn.init.zeros_(value)
        h = torch.tensor([[[0.], [1.], [2.], [9999.]]])
        mask = torch.tensor([[True, True, True, False]])
        relative = torch.zeros(1, 4)
        summaries = []
        handle = head.actual.register_forward_pre_hook(lambda module, args: summaries.append(args[0].detach().clone()))
        head(h, mask, relative)
        expected = torch.tensor([[1., -4. / 9., 8. / 9.]])
        torch.testing.assert_close(summaries[-1], expected)
        h[0, 3] = -9999.
        head(h, mask, relative)
        torch.testing.assert_close(summaries[-1], expected)
        handle.remove()

    def test_cumulative_pool_gradients_are_finite_and_ignore_padding(self):
        torch.manual_seed(97)
        head = JointVowelReadout(5, 'vowel_progress')
        h = torch.randn(2, 8, 5, requires_grad=True)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 6:] = False
        logits, _ = head(h, mask, torch.zeros(2, 8))
        logits.square().sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertEqual(float(h.grad[:, 6:].abs().sum()), 0.)
        self.assertGreater(float(h.grad[:, :6].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()

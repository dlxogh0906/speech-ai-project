import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import numpy as np
import torch
from mdd.sibilant_tag_support import coarse_roles
from mdd.sibilant_cv_support import SupportReadout, reliable_support
from mdd.sibilant_joint_vowel import coarse_cv_roles, joint_vowel_loss, JointVowelReadout, strict_or_joint_support


class JointVowelTest(unittest.TestCase):
    def test_only_artificial_midpoint_mask_is_removed(self):
        times = np.arange(0, .201, .001)
        intervals = [(0., .1, 'ㅅ'), (.1, .16, 'ㅣ')]
        split = coarse_roles(times, intervals, 0, {'ㅣ'})
        merged = coarse_cv_roles(times, intervals, 0, {'ㅣ'})
        expected = split.copy()
        expected[expected == 3] = 2
        np.testing.assert_array_equal(merged[split != -100], expected[split != -100])
        self.assertEqual(int(merged[130]), 2)
        self.assertEqual(int(split[130]), -100)
        self.assertEqual(int(merged[100]), -100)

    def test_merged_loss_is_invariant_to_E_L_swap_and_unknowns(self):
        torch.manual_seed(79)
        x = torch.randn(2, 7, 4, requires_grad=True)
        y = torch.tensor([[0, 1, 2, 2, -100, 0, 1], [2, 1, 0, -100, 2, 1, 0]])
        a = joint_vowel_loss(x, y)
        b = joint_vowel_loss(x[..., [0, 1, 3, 2]], y)
        torch.testing.assert_close(a, b)
        a.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[y == -100].abs().sum()), 0.)

    def test_parameter_identity_and_joint_observation(self):
        torch.manual_seed(83)
        old = SupportReadout(5, True)
        head = JointVowelReadout(5, 'learned')
        head.load_state_dict(old.state_dict())
        h, mask, relative = torch.randn(1, 7, 5), torch.ones(1, 7, dtype=torch.bool), torch.randn(1, 7)
        self.assertTrue(torch.equal(old(h, mask, relative)[0], head(h, mask, relative)[0]))
        logits = torch.tensor([[[0., 5., -2., -2.], [-3., -3., 1., 1.], [0., -1., -2., -2.]]])
        observed = torch.tensor([[True, True, False]])
        self.assertFalse(bool(reliable_support(logits, observed)[0]))
        self.assertTrue(bool(strict_or_joint_support(logits, observed)[0]))
        observed[0, 1] = False
        self.assertFalse(bool(strict_or_joint_support(logits, observed)[0]))


if __name__ == '__main__':
    unittest.main()

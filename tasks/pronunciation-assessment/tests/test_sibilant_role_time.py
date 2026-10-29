import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_role_time import RoleTimeReadout, temporal_moments


class RoleTimeTest(unittest.TestCase):
    def test_zero_initialization_preserves_old_logits_and_isolates_new_gradient(self):
        torch.manual_seed(44)
        old = SupportReadout(5, True)
        heads = [RoleTimeReadout(5, active) for active in (False, True)]
        hidden = torch.randn(2, 9, 5)
        mask = torch.arange(9)[None, :] < torch.tensor([9, 6])[:, None]
        relative = torch.linspace(-1, 1, 9).expand(2, -1)
        expected, roles = old(hidden, mask, relative)
        for head in heads:
            head.load_base(old.state_dict())
            value, phase = head(hidden, mask, relative)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(phase, roles, rtol=0, atol=0)
            torch.nn.functional.cross_entropy(value, torch.tensor([0, 1])).backward()
        self.assertEqual(int(heads[0].temporal.weight.grad.count_nonzero()), 0)
        self.assertGreater(int(heads[1].temporal.weight.grad.count_nonzero()), 0)

    def test_padding_cannot_add_duration_or_move_centers(self):
        torch.manual_seed(22)
        logits = torch.randn(1, 7, 4)
        original = temporal_moments(logits, torch.ones(1, 7, dtype=torch.bool))
        padded = torch.cat([logits, torch.full((1, 4, 4), 100.)], 1)
        mask = torch.arange(11)[None, :] < 7
        torch.testing.assert_close(temporal_moments(padded, mask), original, rtol=1e-6, atol=1e-7)

    def test_known_grid_roles_have_expected_mass_and_relative_centers(self):
        roles = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 3, 0])
        logits = torch.full((1, 10, 4), -20.)
        logits[0, torch.arange(10), roles] = 20.
        stats = temporal_moments(logits, torch.ones(1, 10, dtype=torch.bool))[0]
        torch.testing.assert_close(stats[:3], torch.tensor([3., 2., 3.]) * (.02/.31))
        torch.testing.assert_close(stats[3:6], torch.tensor([-2.5, 0., 2.5]) * (.02/.31), atol=1e-7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()

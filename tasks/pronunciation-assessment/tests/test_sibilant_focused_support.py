import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_focused_support import SquaredSupportReadout


class FocusedSupportTest(unittest.TestCase):
    def test_uniform_control_is_identical(self):
        torch.manual_seed(2)
        base, focused = SupportReadout(4, False), SquaredSupportReadout(4, False)
        focused.load_state_dict(base.state_dict())
        x, relative = torch.randn(2, 8, 4), torch.randn(2, 8)
        mask = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
        a, b = base(x, mask, relative), focused(x, mask, relative)
        for old, new in zip(a, b):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_padding_cannot_enter_focused_pool_or_gradient(self):
        torch.manual_seed(2)
        head = SquaredSupportReadout(4, True)
        x = torch.randn(1, 8, 4, requires_grad=True)
        relative = torch.randn(1, 8)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        y, _ = head(x, mask, relative)
        other = x.detach().clone()
        other[:, 5:] = 1000
        second, _ = head(other, mask, relative)
        torch.testing.assert_close(y, second, rtol=0, atol=0)
        y.sum().backward()
        self.assertEqual(float(x.grad[:, 5:].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()

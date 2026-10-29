import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_context_diagnostic import attenuate_outside, OriginalGroupNorm


class ContextDiagnosticTest(unittest.TestCase):
    def test_only_outside_protected_samples_change_and_identity_gain_exact(self):
        x = np.random.RandomState(91).normal(size=500).astype(np.float32)
        for gain in [0., .1, 1.]:
            y = attenuate_outside(x, 130, 350, gain, 40)
            np.testing.assert_array_equal(y[130:350], x[130:350])
            np.testing.assert_array_equal(y[:90], x[:90]*gain)
            if gain == 1.:
                np.testing.assert_array_equal(x, y)
        with self.assertRaises(ValueError):
            attenuate_outside(x, 350, 130)

    def test_held_stats_native_identity_locality_and_exception_cleanup(self):
        torch.manual_seed(9)
        module = torch.nn.GroupNorm(2, 4).eval()
        with torch.no_grad():
            module.weight.copy_(torch.tensor([.7, 1.2, .9, 1.4]))
        x = torch.randn(1, 4, 30)
        original = module(x)
        reference = OriginalGroupNorm(module, x, original)
        changed = x.clone()
        changed[:, :, :8] *= .1
        with reference.hold():
            self.assertTrue(torch.equal(module(x), original))
            held = module(changed)
            self.assertTrue(torch.equal(held[:, :, 8:], original[:, :, 8:]))
        self.assertGreater(float((module(changed)[:, :, 8:]-original[:, :, 8:]).abs().max()), .01)
        with self.assertRaises(RuntimeError):
            with reference.hold():
                raise RuntimeError('cleanup')
        self.assertTrue(torch.equal(module(x), original))
        self.assertEqual(len(module._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()

import itertools
import math
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.sibilant_null_path import null_path_evidence


def brute(emission, present, anchor):
    roles = [0, 1, 2, 0] if present else [0, 2, 0]
    v = 2 if present else 1
    paths = []
    for path in itertools.product(range(len(roles)), repeat=len(emission)):
        if path[0] > 1 or path[-1] < len(roles) - 2:
            continue
        if any(b - a not in (0, 1) for a, b in zip(path, path[1:])):
            continue
        if anchor >= 0 and path[anchor] != v:
            continue
        paths.append(sum(float(emission[t, roles[s]]) for t, s in enumerate(path)))
    return torch.logsumexp(torch.tensor(paths, dtype=torch.float64), 0).item()


class NullPathTest(unittest.TestCase):
    def test_matches_enumerated_paths_and_has_finite_gradient(self):
        torch.manual_seed(9)
        x = torch.randn(2, 5, 4, requires_grad=True)
        mask = torch.ones(2, 5, dtype=torch.bool)
        got = null_path_evidence(x, mask, torch.tensor([-1, 3]))
        lp = x.detach().log_softmax(-1)
        em = torch.stack([lp[..., 0], lp[..., 1], lp[..., 2:].logsumexp(-1)], -1)
        for i, anchor in enumerate([-1, 3]):
            wanted = (brute(em[i], True, anchor) - brute(em[i], False, anchor)
                      - brute(torch.zeros(5, 3), True, anchor) + brute(torch.zeros(5, 3), False, anchor))
            self.assertAlmostEqual(float(got[i]), wanted, places=5)
        got.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_uniform_roles_have_no_length_or_path_count_advantage(self):
        x = torch.tensor([0., 0., -math.log(2), -math.log(2)]).repeat(3, 9, 1)
        mask = torch.arange(9)[None, :] < torch.tensor([3, 6, 9])[:, None]
        torch.testing.assert_close(null_path_evidence(x, mask, torch.tensor([1, 4, 7])), torch.zeros(3), atol=2e-6, rtol=0)
        y = x.clone()
        y[~mask] = 1000
        torch.testing.assert_close(null_path_evidence(x, mask), null_path_evidence(y, mask), atol=0, rtol=0)

    def test_V_only_does_not_require_or_invent_a_C(self):
        logits = torch.full((2, 8, 4), -6.)
        logits[:, :, 2] = 6.
        logits[0, :3, 1] = 12.
        logits[0, :3, 2] = -6.
        result = null_path_evidence(logits, torch.ones(2, 8, dtype=torch.bool), torch.tensor([4, 4]))
        self.assertGreater(float(result[0]), 5.)
        self.assertLess(float(result[1]), -5.)


if __name__ == '__main__':
    unittest.main()

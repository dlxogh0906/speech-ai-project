"""Compare all paths, best paths and occupancy with exhaustive enumeration."""
import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.ctc_evidence import trellis, occurrence


def collapse(path):
    return [x for x, _ in itertools.groupby(path) if x != 0]


class CTCEvidenceTest(unittest.TestCase):
    def test_bruteforce_with_repeats_and_empty(self):
        rng = np.random.RandomState(17)
        probs = rng.dirichlet(np.ones(3), size=5)
        lp = np.log(probs)
        for tokens in ([], [1], [1, 1], [1, 2], [2, 1, 2]):
            paths = [p for p in itertools.product(range(3), repeat=5) if collapse(p) == tokens]
            scores = np.array([sum(lp[t, p[t]] for t in range(5)) for p in paths])
            z = trellis(lp, tokens)
            self.assertAlmostEqual(z['logsum'], np.logaddexp.reduce(scores), places=12)
            self.assertAlmostEqual(z['best'], scores.max(), places=12)
            np.testing.assert_allclose(z['posterior'].sum(1), 1., atol=1e-12)
            # The token occupancy recovered from all alignment states equals
            # exhaustive conditioned path occupancy at every frame.
            ext = np.zeros(2 * len(tokens) + 1, int); ext[1::2] = tokens
            weights = np.exp(scores - z['logsum'])
            for v in range(3):
                brute = sum(w * (np.asarray(p) == v) for w, p in zip(weights, paths))
                np.testing.assert_allclose(z['posterior'][:, ext == v].sum(1), brute, atol=1e-12)
            for q in range(len(tokens)):
                self.assertIsNotNone(occurrence(z, q))

    def test_impossible_repeat_and_zero_frames(self):
        z = trellis(np.log(np.full((2, 3), 1 / 3)), [1, 1])
        self.assertEqual(z['logsum'], -np.inf)
        self.assertIsNone(z['posterior'])
        self.assertEqual(trellis(np.empty((0, 3)), [])['logsum'], 0.)
        self.assertEqual(trellis(np.empty((0, 3)), [1])['logsum'], -np.inf)

    def test_exact_ties_and_hard_zeros(self):
        lp = np.array([[0., -np.inf, -np.inf], [-np.inf, 0., -np.inf]])
        self.assertEqual(trellis(lp, [1])['logsum'], 0.)
        self.assertEqual(trellis(lp, [2])['logsum'], -np.inf)

    def test_reject_blank_target(self):
        with self.assertRaises(ValueError):
            trellis(np.zeros((4, 3)), [0])


if __name__ == '__main__':
    unittest.main()

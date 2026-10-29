import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_frozen_probe import FrozenProbe, load_probe
from mdd.sibilant_temporal_residual import (TemporalResidualProbe, make_permutations,
    shuffle_observed, temporal_artifact, load_temporal_probe)


class TemporalResidualTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        self.projection = {'mean': torch.zeros(6), 'scale': torch.ones(6),
                           'components': torch.linalg.qr(torch.randn(6, 3)).Q,
                           'explained_variance_ratio': .9}
        self.hidden = torch.randn(2, 5, 6)
        self.mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
        self.relative = torch.zeros(2, 5)

    def model(self, mode='ordered'):
        torch.manual_seed(29)
        return TemporalResidualProbe(self.projection, mode).eval()

    def train_model(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(self.hidden, self.mask, self.relative)
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            optimizer.step()
        return model

    def test_zero_branch_preserves_base_outputs_for_both_modes(self):
        torch.manual_seed(29)
        base = FrozenProbe(self.projection).eval()
        expected, phase = base(self.hidden, self.mask, self.relative)
        ordered, shuffled = self.model(), self.model('shuffled')
        permutation = torch.tensor([[4, 1, 3, 0, 2], [2, 0, 1, 3, 4]])
        for model, perm in [(ordered, None), (shuffled, permutation)]:
            actual, role = model(self.hidden, self.mask, self.relative, perm)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(role, phase))

    def test_learned_branch_handles_nan_and_appended_padding(self):
        model = self.train_model()
        self.assertGreater(float(model.branch.layers[0].weight.grad.abs().sum()), 0.)
        self.assertFalse(model.components.requires_grad)
        before = model(self.hidden, self.mask, self.relative)[0]
        poisoned = self.hidden.clone()
        poisoned[~self.mask] = float('nan')
        expanded = torch.cat([poisoned, torch.full((2, 3, 6), float('nan'))], 1)
        mask = torch.cat([self.mask, torch.zeros(2, 3, dtype=torch.bool)], 1)
        relative = torch.cat([self.relative, torch.zeros(2, 3)], 1)
        after = model(expanded, mask, relative)[0]
        self.assertTrue(torch.isfinite(after).all())
        torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)

    def test_typed_artifact_reloads_trained_branch_and_rejects_plain_loader(self):
        model = self.train_model()
        artifact = temporal_artifact(model, {'backbone_key': 'korean'})
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        restored = load_temporal_probe(torch.load(buffer), {'backbone_key': 'korean'}, 'cpu')
        self.assertTrue(torch.equal(model(self.hidden, self.mask, self.relative)[0],
                                    restored(self.hidden, self.mask, self.relative)[0]))
        with self.assertRaises(ValueError):
            load_probe(artifact, {'backbone_key': 'korean'}, 'cpu')
        with self.assertRaises(ValueError):
            load_temporal_probe(artifact, {'backbone_key': 'other'}, 'cpu')
        artifact['binding']['classes'].reverse()
        with self.assertRaises(ValueError):
            load_temporal_probe(artifact, {'backbone_key': 'korean'}, 'cpu')

    def test_permutations_ignore_labels_and_keep_padding_outside_prefix(self):
        features = [{'uid': 'a', 'q_runtime': 2, 'actual': 'ㅅ'}, {'uid': 'b', 'q_runtime': 4}]
        first = make_permutations(features, [5, 3], 5, 7)
        features[0]['actual'] = 'ㅆ'
        np.testing.assert_array_equal(first, make_permutations(features, [5, 3], 5, 7))
        shuffled = shuffle_observed(self.hidden, self.mask, torch.as_tensor(first))
        self.assertTrue(torch.equal(shuffled[1, 3:], self.hidden[1, 3:]))
        broken = torch.tensor([[4, 1, 3, 0, 2], [3, 0, 1, 2, 4]])
        with self.assertRaises(ValueError):
            shuffle_observed(self.hidden, self.mask, broken)


if __name__ == '__main__':
    unittest.main()

import copy
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
from mdd.sibilant_discarded_axes import (SCHEMA, redistribute_pair, fit_axes, axis_observations,
    DiscardedAxisResidual, axis_artifact, load_axis_residual, normalization_digest)
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest, READOUT
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class DiscardedAxesTest(unittest.TestCase):
    def test_zero_identity_and_nonzero_pairmass_other_probabilities(self):
        torch.manual_seed(19)
        logits = torch.randn(9, 5)
        self.assertTrue(torch.equal(logits, redistribute_pair(logits, torch.zeros(9))))
        delta = torch.linspace(-5, 5, 9).requires_grad_()
        changed = redistribute_pair(logits, delta)
        a, b = logits.softmax(-1), changed.softmax(-1)
        torch.testing.assert_close(a[:, 2:], b[:, 2:], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a[:, :2].sum(-1), b[:, :2].sum(-1), atol=1e-6, rtol=1e-6)
        grad = torch.autograd.grad(changed.log_softmax(-1)[:, 2].sum(), delta)[0]
        self.assertLess(float(abs(grad).max()), 1e-6)

    def test_crosshalf_axes_stay_outside_original_PCA(self):
        torch.set_num_threads(2)
        q = torch.eye(10)[:, :2]
        deltas = np.zeros((8, 2, 3, 10))
        for i in range(8):
            deltas[i, :, :2, 2+i % 6] = 1+i/10.
        axes, report = fit_axes(deltas, q, 'cpu')
        self.assertEqual(tuple(axes.shape), (2, 10, 4))
        self.assertLess(report['PCA_axis_max_abs_dot'], 2e-6)
        for z in axes:
            torch.testing.assert_close(z.T@z, torch.eye(4), atol=1e-6, rtol=1e-6)

    def test_trained_direct_cached_reload_and_frozen_parent(self):
        torch.manual_seed(41)
        torch.set_num_threads(2)
        q = torch.eye(10)[:, :2]
        projection = {'mean': torch.zeros(10), 'scale': torch.ones(10), 'components': q}
        base = FrozenProbe(projection).eval()
        parent = {'readout': READOUT, 'projection': projection, 'head': copy.deepcopy(base.head.state_dict()),
            'binding': {'backbone_key': 'unit', 'classes': list(CLASSES),
                        'projection_sha256': projection_digest(projection)}}
        axes = torch.stack([torch.eye(10)[:, 2:6]]*2)
        norm = {'schema': SCHEMA, 'partition': 'train', 'mean': [0.]*8, 'scale': [1.]*8}
        model = DiscardedAxisResidual(base, axes, norm, 'signal').eval()
        x, m, r = torch.randn(4, 8, 10), torch.ones(4, 8, dtype=torch.bool), torch.zeros(4, 8)
        groups = torch.arange(4)
        before = tensor_digest(base.state_dict())
        logits, phase = base(x, m, r)
        raw = axis_observations(base, x, m, phase, axes)
        self.assertTrue(torch.equal(model(x, m, r, groups)[0], logits))
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model.from_observations(logits, raw, groups), torch.tensor([0, 1, 2, 4])).backward()
            optimizer.step()
        self.assertEqual(before, tensor_digest(base.state_dict()))
        self.assertEqual(sum(v.numel() for v in model.parameters() if v.requires_grad), 36)
        saved = axis_artifact(model, parent, norm)
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        restored = load_axis_residual(torch.load(stream), {'backbone_key': 'unit'}, 'signal', 'cpu')
        expected = model(x, m, r, groups)[0]
        self.assertTrue(torch.equal(expected, restored(x, m, r, groups)[0]))
        torch.testing.assert_close(expected, model.from_observations(logits, raw, groups), atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            load_axis_residual(saved, {}, 'permuted', 'cpu')
        bad = copy.deepcopy(saved)
        bad['normalization']['mean'][0] += 1
        bad['binding']['normalization_sha256'] = normalization_digest(bad['normalization'])
        with self.assertRaises(ValueError):
            load_axis_residual(bad, {}, 'signal', 'cpu')


if __name__ == '__main__':
    unittest.main()

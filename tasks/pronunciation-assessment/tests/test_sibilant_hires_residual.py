import copy
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_hires_acoustics import (NAMES, observations, coarsen_20ms,
    normalized_input)
from mdd.sibilant_hires_residual import HiresResidual, residual_artifact, load_residual
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest, load_probe, READOUT
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class HiresTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        torch.set_num_threads(2)
        self.projection = {'mean': torch.zeros(6), 'scale': torch.ones(6), 'components': torch.randn(6, 3)}
        self.norm = {'mean': [0.]*len(NAMES), 'scale': [1.]*len(NAMES)}

    def test_actual_window_coordinates_no_spectral_extrapolation_or_PCM_mutation(self):
        x = np.random.RandomState(13).normal(0, .01, 12000).astype(np.float32)
        original = x.copy()
        z, relative, first = observations(x, {'left': 103, 'right': 10599}, .4)
        self.assertEqual(first, 4)
        self.assertEqual(z.shape[1], len(NAMES))
        self.assertAlmostEqual(float(relative[0]), ((128+79.5)/16000-.4)/.31, places=6)
        self.assertTrue(np.isnan(z[0, 3:]).all())
        self.assertTrue(np.isfinite(z[:, :3]).all())
        np.testing.assert_array_equal(x, original)

    def test_coarse_control_keeps_missingness_and_global_partial_bin_identity(self):
        x = np.arange(28, dtype=np.float32).reshape(14, 2)
        x[2, 0], x[10, 1] = np.nan, np.nan
        coarse = coarsen_20ms(x, 7)
        np.testing.assert_array_equal(np.isfinite(x), np.isfinite(coarse))
        self.assertEqual(coarse[0, 1], x[:3, 1].mean())
        self.assertEqual(coarse[-1, 0], x[-1, 0])
        z = normalized_input(np.full((2, len(NAMES)), np.nan, np.float32), self.norm)
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(float(abs(z).sum()), 0.)

    def test_zero_residual_trained_reload_padding_and_binding_rejections(self):
        base = FrozenProbe(self.projection).eval()
        base_artifact = {'readout': READOUT, 'projection': self.projection,
            'head': copy.deepcopy(base.head.state_dict()), 'binding': {
                'backbone_key': 'unit', 'classes': list(CLASSES),
                'projection_sha256': projection_digest(self.projection)}}
        model = HiresResidual(base, 'fine').eval()
        hidden, mask, relative = torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8)
        raw, rm, rr = torch.randn(2, 30, 2*len(NAMES)), torch.ones(2, 30, dtype=torch.bool), torch.zeros(2, 30)
        self.assertTrue(torch.equal(model(hidden, mask, relative, raw, rm, rr)[0], base(hidden, mask, relative)[0]))
        before = tensor_digest(base.state_dict())
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(hidden, mask, relative, raw, rm, rr)[0], torch.tensor([0, 1])).backward()
            optimizer.step()
        self.assertEqual(before, tensor_digest(base.state_dict()))
        self.assertGreater(float(model.layers[0].weight.grad.abs().sum()), 0.)
        artifact = residual_artifact(model, base_artifact, self.norm)
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        restored = load_residual(torch.load(buffer), {'backbone_key': 'unit'}, 'fine', 'cpu')
        expected = model(hidden, mask, relative, raw, rm, rr)[0]
        self.assertTrue(torch.equal(expected, restored(hidden, mask, relative, raw, rm, rr)[0]))
        padded = torch.cat([raw, torch.full((2, 7, 2*len(NAMES)), float('nan'))], 1)
        pm = torch.cat([rm, torch.zeros(2, 7, dtype=torch.bool)], 1)
        pr = torch.cat([rr, torch.zeros(2, 7)], 1)
        torch.testing.assert_close(restored(hidden, mask, relative, padded, pm, pr)[0], expected, atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            load_probe(artifact, {}, 'cpu')
        with self.assertRaises(ValueError):
            load_residual(artifact, {}, 'coarse20ms', 'cpu')
        bad = copy.deepcopy(artifact)
        bad['normalization']['mean'][0] += 1
        with self.assertRaises(ValueError):
            load_residual(bad, {}, 'fine', 'cpu')


if __name__ == '__main__':
    unittest.main()

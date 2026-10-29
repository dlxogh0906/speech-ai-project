import copy
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
from mdd.sibilant_pitchfree_spectrum import (SCHEMA, BINS, FREQUENCIES, BANDS,
    observations, coarse_bands, normalized_input)
from mdd.sibilant_spectral_residual import SpectralResidual, spectral_artifact, load_spectral
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest, READOUT
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class SpectrumTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.norm = {'schema': SCHEMA, 'partition': 'train', 'mean': [0.]*BINS,
                     'scale': [20.]*BINS, 'energy_mean': 0., 'energy_scale': 5.}

    def test_original_ROI_support_gain_offset_invariance_and_missing_not_filled(self):
        wave = np.random.RandomState(43).normal(0, .02, 12000)
        original = wave.copy()
        f = {'left': 103, 'right': 10599}
        x, e, r, first = observations(wave, f, .4)
        self.assertEqual(first, 2)
        self.assertEqual(x.shape[-1], BINS)
        self.assertAlmostEqual(float(r[0]), ((160+319.5)/16000-.4)/.31, places=6)
        np.testing.assert_array_equal(wave, original)
        altered = wave*2+3
        altered[:103] = 999
        altered[10599:] = -999
        check = observations(altered, f, .4)
        np.testing.assert_array_equal(check[0], x)
        np.testing.assert_array_equal(check[1], e)
        for flat in (np.zeros(12000), np.ones(12000)*.2):
            values, energy, relative, _ = observations(flat, f, .4)
            self.assertTrue(np.isnan(values).all())
            z, valid = normalized_input(values, energy, relative, self.norm, 'fine')
            self.assertTrue(np.isfinite(z).all())
            self.assertFalse(valid.any())
            self.assertEqual(float(abs(z[:, :-1]).sum()), 0.)
        short = observations(wave, {'left': 1, 'right': 200}, .1)
        self.assertEqual(len(short[0]), 0)

    def test_control_removes_fine_detail_but_preserves_band_power_validity(self):
        x = np.random.RandomState(4).uniform(-60, 10, (3, BINS)).astype(np.float32)
        x[1] = np.nan
        z = coarse_bands(x)
        np.testing.assert_array_equal(np.isfinite(z), np.isfinite(x))
        for low, high in zip(BANDS[:-1], BANDS[1:]):
            ix = np.flatnonzero((FREQUENCIES >= low) & (FREQUENCIES < high))
            expected = np.mean(10**(x[[0, 2]][:, ix]/10), -1)
            np.testing.assert_allclose(10**(z[[0, 2]][:, ix]/10),
                np.repeat(expected[:, None], len(ix), -1), atol=1e-5, rtol=1e-5)
            self.assertTrue(np.all(z[0, ix] == z[0, ix[0]]))
        # Moving energy within the same band cannot survive this control.
        a = np.full((1, BINS), -60., np.float32)
        b = a.copy()
        a[0, 20], b[0, 25] = 10., 10.
        np.testing.assert_allclose(coarse_bands(a), coarse_bands(b), atol=1e-5)

    def test_trained_serialization_padding_failure_fallback_and_binding(self):
        torch.manual_seed(19)
        projection = {'mean': torch.zeros(6), 'scale': torch.ones(6), 'components': torch.randn(6, 3)}
        base = FrozenProbe(projection).eval()
        artifact = {'readout': READOUT, 'projection': projection,
            'head': copy.deepcopy(base.head.state_dict()), 'binding': {
                'backbone_key': 'unit', 'classes': list(CLASSES),
                'projection_sha256': projection_digest(projection)}}
        model = SpectralResidual(base, 'fine').eval()
        x, m, r = torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8)
        z, v = torch.randn(2, 40, BINS+3), torch.ones(2, 40, dtype=torch.bool)
        self.assertTrue(torch.equal(model(x, m, r, z, v)[0], base(x, m, r)[0]))
        before = tensor_digest(base.state_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x, m, r, z, v)[0], torch.tensor([0, 1])).backward()
            optimizer.step()
        self.assertGreater(float(model.layers[0].weight.grad.abs().sum()), 0.)
        self.assertEqual(before, tensor_digest(base.state_dict()))
        saved = spectral_artifact(model, artifact, self.norm)
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = load_spectral(torch.load(buffer), {'backbone_key': 'unit'}, 'fine', 'cpu')
        expected = model(x, m, r, z, v)[0]
        self.assertTrue(torch.equal(expected, restored(x, m, r, z, v)[0]))
        padded = torch.cat([z, torch.full((2, 9, BINS+3), float('nan'))], 1)
        mask = torch.cat([v, torch.zeros(2, 9, dtype=torch.bool)], 1)
        torch.testing.assert_close(restored(x, m, r, padded, mask)[0], expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(restored(x, m, r, z*float('nan'), ~v)[0], base(x, m, r)[0]))
        with self.assertRaises(ValueError):
            load_spectral(saved, {}, 'coarse_bands', 'cpu')
        bad = copy.deepcopy(saved)
        bad['normalization']['energy_mean'] += 1
        with self.assertRaises(ValueError):
            load_spectral(bad, {}, 'fine', 'cpu')
        bad = copy.deepcopy(saved)
        bad['branch']['output.bias'][0] += 1
        with self.assertRaises(ValueError):
            load_spectral(bad, {}, 'fine', 'cpu')


if __name__ == '__main__':
    unittest.main()

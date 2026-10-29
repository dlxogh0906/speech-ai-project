import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_frozen_probe import sampled_frames, fit_projection, project_observed, projection_digest, validate_binding, READOUT


class FrozenProbeTest(unittest.TestCase):
    def test_sampler_ignores_labels_and_never_selects_padding(self):
        features = [{'uid': 'a', 'q_runtime': 2, 'actual': 'ㅅ'}, {'uid': 'b', 'q_runtime': 4}]
        first = sampled_frames(features, [2, 1], 20)
        features[0]['actual'] = 'ㅆ'
        self.assertTrue(np.array_equal(first, sampled_frames(features, [2, 1], 20)))
        self.assertEqual(set(map(tuple, first)), {(0, 0), (0, 1), (1, 0)})

    def test_constant_features_have_finite_projection(self):
        p = fit_projection(np.ones((8, 3)), 2, 'cpu')
        self.assertTrue(torch.isfinite(p['components']).all())
        self.assertTrue(torch.isfinite(p['scale']).all())

    def test_nan_padding_does_not_change_observed_projection(self):
        p = fit_projection(np.arange(24).reshape(8, 3), 2, 'cpu')
        h = torch.tensor([[[1., 2., 3.], [float('nan')] * 3]])
        result = project_observed(h, torch.tensor([[True, False]]), p)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[0, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(result[0, 0], project_observed(h[:, :1], torch.ones(1, 1, dtype=torch.bool), p)[0, 0]))

    def test_same_shape_head_rejects_other_backbone(self):
        p = fit_projection(np.arange(24).reshape(8, 3), 2, 'cpu')
        artifact = {'readout': READOUT, 'projection': p,
                    'binding': {'backbone_key': 'wave', 'projection_sha256': projection_digest(p)}}
        with self.assertRaises(ValueError):
            validate_binding(artifact, {'backbone_key': 'bert'})

    def test_changed_projection_rejected(self):
        p = fit_projection(np.arange(24).reshape(8, 3), 2, 'cpu')
        artifact = {'readout': READOUT, 'projection': p,
                    'binding': {'backbone_key': 'wave', 'projection_sha256': projection_digest(p)}}
        p['mean'][0] += 1
        with self.assertRaises(ValueError):
            validate_binding(artifact, {'backbone_key': 'wave'})


if __name__ == '__main__':
    unittest.main()

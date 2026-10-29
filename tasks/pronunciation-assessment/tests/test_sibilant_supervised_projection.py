import io
import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_supervised_projection import SupervisedProjectionProbe, supervised_artifact


class SupervisedProjectionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.projection = {'mean': torch.randn(6), 'scale': torch.ones(6),
                           'components': torch.linalg.qr(torch.randn(6, 3)).Q,
                           'explained_variance_ratio': .8}
        self.model = SupervisedProjectionProbe(self.projection).eval()
        self.hidden = torch.randn(2, 5, 6)
        self.mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
        self.relative = torch.zeros(2, 5)

    def update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.02)
        logits, _ = self.model(self.hidden, self.mask, self.relative)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        optimizer.step()

    def test_audio_loss_updates_projection_with_poisoned_padding_but_not_scaler(self):
        self.hidden[~self.mask] = float('nan')
        self.update()
        gradient = self.model.components.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertFalse(torch.equal(self.model.components, self.projection['components']))
        for key in ('mean', 'scale'):
            self.assertIsNone(getattr(self.model, key).grad)
            self.assertTrue(torch.equal(getattr(self.model, key), self.projection[key]))

    def test_saved_probe_uses_updated_matrix_and_preserves_its_outputs(self):
        initial = self.model(self.hidden, self.mask, self.relative)[0].detach().clone()
        self.update()
        trained = self.model(self.hidden, self.mask, self.relative)[0].detach()
        self.assertFalse(torch.allclose(initial, trained))
        binding = {'backbone_key': 'korean', 'coordinate_schema': 'observedV'}
        artifact = supervised_artifact(self.model, binding)
        self.assertNotIn('explained_variance_ratio', artifact['projection'])
        self.assertEqual(artifact['projection']['initial_pca_explained_variance_ratio'], .8)
        self.assertFalse(torch.equal(artifact['projection']['components'], self.projection['components']))
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        reloaded = load_probe(torch.load(buffer, map_location='cpu'), binding, 'cpu')
        self.assertTrue(torch.equal(reloaded(self.hidden, self.mask, self.relative)[0], trained))


if __name__ == '__main__':
    unittest.main()

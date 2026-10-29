"""Artifact coupling prevents pairing an adapted head with stale native features."""
import copy
import io
import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_frozen_probe import FrozenProbe, load_probe
from mdd.sibilant_hubert_adaptation import (binding_from_manifest, coupled_artifact,
    load_coupled, tensor_digest)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.binding = binding_from_manifest({'repo_id': 'unit', 'revision': 'pinned',
            'sha256': {'model.safetensors': 'base', 'config.json': 'config',
                       'preprocessor_config.json': 'processor'}})
        self.projection = {'mean': torch.zeros(1536), 'scale': torch.ones(1536),
                           'components': torch.randn(1536, 256) / 40}

    def encoder(self):
        encoder = torch.nn.Module()
        encoder.encoder = torch.nn.Module()
        encoder.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(12)])
        encoder._sibilant_native_binding = copy.deepcopy(self.binding)
        return encoder

    def artifact(self):
        encoder, head = self.encoder(), FrozenProbe(self.projection).eval()
        with torch.no_grad():
            encoder.encoder.layers[-1].weight.add_(.3)
            next(head.head.parameters()).add_(.01)
        return encoder, head, coupled_artifact(encoder, head, self.binding, 'tail2')

    def test_serialized_changed_tail_and_head_restore_together(self):
        encoder, head, artifact = self.artifact()
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        new = self.encoder()
        restored = load_coupled(torch.load(buffer), self.binding, new, 'cpu')
        self.assertEqual(tensor_digest(new.encoder.layers[-2:].state_dict()),
                         tensor_digest(encoder.encoder.layers[-2:].state_dict()))
        hidden = torch.randn(2, 10, 1536)
        mask, relative = torch.ones(2, 10, dtype=torch.bool), torch.zeros(2, 10)
        self.assertTrue(torch.equal(head(hidden, mask, relative)[0], restored(hidden, mask, relative)[0]))
        with self.assertRaises(ValueError):
            load_probe(artifact, {}, 'cpu')

    def test_wrong_source_classes_depth_and_corrupt_parts_reject_before_mutation(self):
        _, _, artifact = self.artifact()
        variants = []
        for key, value in [('revision', 'other'), ('classes', ['ss', 's']),
                           ('tail_depth', 6), ('precision', 'padded')]:
            bad = copy.deepcopy(artifact)
            bad['binding'][key] = value
            variants.append(bad)
        for part in ('head', 'tail'):
            bad = copy.deepcopy(artifact)
            next(iter(bad[part].values())).add_(1)
            variants.append(bad)
        bad = copy.deepcopy(artifact)
        key = next(iter(bad['tail']))
        bad['tail'][key] = bad['tail'][key].flatten()[:1]
        bad['binding']['tail_sha256'] = tensor_digest(bad['tail'])
        variants.append(bad)
        bad = copy.deepcopy(artifact)
        bad['projection']['scale'][0] = 0
        variants.append(bad)
        for bad in variants:
            encoder = self.encoder()
            before = tensor_digest(encoder.state_dict())
            with self.assertRaises(ValueError):
                load_coupled(bad, self.binding, encoder, 'cpu')
            self.assertEqual(before, tensor_digest(encoder.state_dict()))

    def test_encoder_must_be_attested_to_same_native_files(self):
        _, _, artifact = self.artifact()
        encoder = self.encoder()
        encoder._sibilant_native_binding['native_files_sha256']['model.safetensors'] = 'other'
        with self.assertRaises(ValueError):
            load_coupled(artifact, self.binding, encoder, 'cpu')


if __name__ == '__main__':
    unittest.main()

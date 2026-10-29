import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
from mdd.sibilant_local_ctc_artifact import make_artifact, load_artifact, validate_depth


class LocalCTCArtifactTest(unittest.TestCase):
    def model(self):
        return torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(8)]), torch.nn.Linear(3, 5)

    def test_legacy_tail_loads_only_original_two_layers(self):
        source, head = self.model()
        target, other_head = self.model()
        prefix = {k: v.clone() for k, v in target[:6].state_dict().items()}
        artifact = make_artifact(source[-2:], head, 2, 'vocab', 'local_CTC')
        self.assertEqual(artifact['readout'], 'local_phone_ctc_last2_v1')
        self.assertEqual(load_artifact(target, other_head, artifact, 'vocab'), 2)
        for k, v in target[:6].state_dict().items():
            self.assertTrue(torch.equal(v, prefix[k]))
        for k, v in target[-2:].state_dict().items():
            self.assertTrue(torch.equal(v, source[-2:].state_dict()[k]))

    def test_six_layer_tail_round_trip_and_frozen_prefix(self):
        source, head = self.model()
        target, other_head = self.model()
        prefix = {k: v.clone() for k, v in target[:2].state_dict().items()}
        artifact = make_artifact(source[-6:], head, 6, 'vocab', 'local_CTC')
        self.assertEqual(load_artifact(target, other_head, artifact, 'vocab'), 6)
        for k, v in target[:2].state_dict().items():
            self.assertTrue(torch.equal(v, prefix[k]))
        for k, v in target[-6:].state_dict().items():
            self.assertTrue(torch.equal(v, source[-6:].state_dict()[k]))
        for k, v in other_head.state_dict().items():
            self.assertTrue(torch.equal(v, head.state_dict()[k]))

    def test_invalid_schema_or_depth_rejected(self):
        layers, head = self.model()
        for depth in [0, -1, 9, True, 2.5, None]:
            with self.assertRaises(ValueError):
                validate_depth(depth, len(layers))
        artifact = make_artifact(layers[-6:], head, 6, 'vocab', 'local_CTC')
        with self.assertRaises(ValueError):
            load_artifact(layers, head, artifact, 'different_vocab')
        artifact['encoder_train_layers'] = 2
        with self.assertRaises(ValueError):
            load_artifact(layers, head, artifact, 'vocab')

    def test_bad_head_cannot_partially_mutate_encoder(self):
        source, head = self.model()
        target, other_head = self.model()
        before = {k: v.clone() for k, v in target.state_dict().items()}
        artifact = make_artifact(source[-6:], head, 6, 'vocab', 'local_CTC')
        artifact['lm_head']['weight'] = torch.ones(6, 3)
        with self.assertRaises(ValueError):
            load_artifact(target, other_head, artifact, 'vocab')
        for k, v in target.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))


if __name__ == '__main__':
    unittest.main()

import copy
import io
import sys
import unittest
from pathlib import Path
import torch
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/src'))
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/scripts/data'))
from prepare_sibilant_raw_cv_supervision import retained_policy
from mdd.sibilant_raw_cv import (RawCVResidual, coarse_role_loss, cv_terms,
    cv_artifact, load_cv)
from mdd.sibilant_hires_residual import load_residual
from mdd.sibilant_hires_acoustics import NAMES
from mdd.sibilant_frozen_probe import FrozenProbe, READOUT, projection_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class RawCVTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(103)

    def test_original_unknown_cannot_gain_class_or_role_supervision(self):
        def source(label, c, v, left, right, centers, adjacent, edge):
            return 1, np.ones(len(centers), np.int64), 'newly_supported'
        policy = retained_policy(np.array([-100, 1, 0]), source)
        args = (None, None, 0, 100, np.arange(4), True)
        cls, phase, state = policy({'feature_index': 0}, *args)
        self.assertEqual(cls, -100)
        np.testing.assert_array_equal(phase, [-100]*4)
        self.assertEqual(state, 'original_unknown_preserved')
        self.assertEqual(policy({'feature_index': 1}, *args)[0], 1)
        with self.assertRaises(ValueError):
            policy({'feature_index': 2}, *args)

    def test_CV_pooling_and_vowel_time_covariance_with_poisoned_padding(self):
        features = torch.tensor([[[1.], [1.], [1.], [2.], [3.], [4.], [float('nan')]]])
        role = torch.tensor([[[0., 12., 0.]]*3+[[0., 0., 12.]]*3+[[0., 0., 0.]]])
        mask = torch.tensor([[True]*6+[False]])
        relative = torch.tensor([[-.3, -.2, -.1, 0., .1, .2, float('nan')]])
        summary, weights = cv_terms(features, role, mask, relative)
        self.assertTrue(torch.isfinite(summary).all())
        self.assertAlmostEqual(float(summary[0, 0]), 1., places=3)
        self.assertAlmostEqual(float(summary[0, 1]), 2., places=3)
        self.assertGreater(float(summary[0, 2]), .7)
        self.assertEqual(float(weights[..., -1].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            cv_terms(features, torch.zeros(1, 7, 4), mask, relative)

    def test_unknown_role_targets_remain_unknown_and_loss_is_finite(self):
        logits = torch.randn(2, 4, 3, requires_grad=True)
        target = torch.full((2, 4), -100)
        loss = coarse_role_loss(logits, target)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.)
        target[0, 2] = 2
        self.assertTrue(torch.equal(coarse_role_loss(logits, target),
                                    torch.nn.functional.cross_entropy(logits[0, 2:3], torch.tensor([2]))))

    def test_zero_base_invariance_trained_role_reload_padding_and_typed_binding(self):
        projection = {'mean': torch.zeros(6), 'scale': torch.ones(6), 'components': torch.randn(6, 3)}
        base = FrozenProbe(projection).eval()
        artifact = {'readout': READOUT, 'projection': projection, 'head': copy.deepcopy(base.head.state_dict()),
            'binding': {'backbone_key': 'unit', 'classes': list(CLASSES),
                        'projection_sha256': projection_digest(projection)}}
        model = RawCVResidual(base, 'guided').eval()
        x, mask, rel = torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8)
        raw, rm, rr = torch.randn(2, 25, 2*len(NAMES)), torch.ones(2, 25, dtype=torch.bool), torch.linspace(-.5, .5, 25)[None].repeat(2, 1)
        logits, phase, raw_phase = model(x, mask, rel, raw, rm, rr)
        self.assertTrue(torch.equal(logits, base(x, mask, rel)[0]))
        before = tensor_digest(base.state_dict())
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits, _, role = model(x, mask, rel, raw, rm, rr)
            target = torch.tensor([[0]*5+[1]*10+[2]*10]*2)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))+.5*coarse_role_loss(role, target)
            loss.backward()
            optimizer.step()
        self.assertEqual(before, tensor_digest(base.state_dict()))
        self.assertGreater(float(model.role.weight.grad.abs().sum()), 0.)
        saved = cv_artifact(model, artifact, {'mean': [0.]*len(NAMES), 'scale': [1.]*len(NAMES)})
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = load_cv(torch.load(buffer), {'backbone_key': 'unit'}, 'guided', 'cpu')
        expected = model(x, mask, rel, raw, rm, rr)
        actual = restored(x, mask, rel, raw, rm, rr)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))
        expanded = torch.cat([raw, torch.full((2, 7, 2*len(NAMES)), float('nan'))], 1)
        em = torch.cat([rm, torch.zeros(2, 7, dtype=torch.bool)], 1)
        er = torch.cat([rr, torch.full((2, 7), float('nan'))], 1)
        torch.testing.assert_close(restored(x, mask, rel, expanded, em, er)[0], expected[0], atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            load_residual(saved, {}, 'fine', 'cpu')
        with self.assertRaises(ValueError):
            load_cv(saved, {}, 'unguided', 'cpu')
        broken = copy.deepcopy(saved)
        broken['binding']['role_schema'] = 'invented_early_late'
        with self.assertRaises(ValueError):
            load_cv(broken, {}, 'guided', 'cpu')


if __name__ == '__main__':
    unittest.main()

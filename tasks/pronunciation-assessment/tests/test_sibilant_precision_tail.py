import copy
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
from mdd.sibilant_precision_tail import PrecisionTailHead, rank_risk, opposite_score, tail_artifact, load_tail_head
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest, READOUT
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class TailTest(unittest.TestCase):
    def test_tail_selects_hard_confirmed_normals_and_unknown_gradient_zero(self):
        logits = torch.zeros(9, 5)
        logits[2, 0], logits[3, 0], logits[6, 1], logits[7, 1] = 3., 1., 4., 2.
        logits.requires_grad_()
        groups = [{'positive': [0], 'negative': [1, 2, 3]},
                  {'positive': [4], 'negative': [5, 6, 7]}]
        full, _ = rank_risk(logits, groups, 'fullrank', fraction=.01)
        tail, selected = rank_risk(logits, groups, 'tailrank', fraction=.01)
        self.assertEqual([z.tolist() for z in selected], [[2], [6]])
        self.assertGreater(float(tail), float(full))
        grad = torch.autograd.grad(tail, logits)[0]
        self.assertGreater(float(grad[2].abs().sum()), 0.)
        self.assertGreater(float(grad[6].abs().sum()), 0.)
        self.assertEqual(float(grad[[1, 3, 5, 7, 8]].abs().sum()), 0.)
        # Reducing the erroneous opposite evidence in a selected normal helps.
        better = logits.detach().clone()
        better[2, 0] -= 1
        self.assertLess(float(rank_risk(better, groups, 'tailrank')[0]), float(tail))
        shifted = logits.detach()+torch.arange(9)[:, None]*10
        torch.testing.assert_close(opposite_score(shifted, 0), opposite_score(logits.detach(), 0), atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            rank_risk(logits, [{'positive': [0], 'negative': [0]}, groups[1]], 'tailrank')
        with self.assertRaises(ValueError):
            rank_risk(logits, groups, 'tailrank', fraction=0.)

    def test_singleclip_summary_equivalence_trained_reload_and_frozen_support(self):
        torch.manual_seed(37)
        torch.set_num_threads(2)
        projection = {'mean': torch.zeros(6), 'scale': torch.ones(6), 'components': torch.randn(6, 3)}
        base = FrozenProbe(projection).eval()
        artifact = {'readout': READOUT, 'projection': projection, 'head': copy.deepcopy(base.head.state_dict()),
            'binding': {'backbone_key': 'unit', 'classes': list(CLASSES),
                        'projection_sha256': projection_digest(projection)}}
        model = PrecisionTailHead(base, 'tailrank').eval()
        x, mask, relative = torch.randn(8, 9, 6), torch.ones(8, 9, dtype=torch.bool), torch.zeros(8, 9)
        mask[0, 7:] = False
        x[0, 7:] = float('nan')
        summary, moments, phase = model.summarize(x, mask, relative)
        self.assertTrue(torch.equal(model.from_summary(summary, moments), base(x, mask, relative)[0]))
        before = tensor_digest(base.state_dict())
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        groups = [{'positive': [0], 'negative': [1, 2, 3]}, {'positive': [4], 'negative': [5, 6, 7]}]
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits = model.from_summary(summary, moments)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 1, 1, 1, 0, 2, 4]))+rank_risk(logits, groups, 'tailrank')[0]
            loss.backward()
            optimizer.step()
        self.assertEqual(before, tensor_digest(base.state_dict()))
        self.assertTrue(all(v.grad is None for v in model.head.support.parameters()))
        self.assertTrue(torch.equal(phase, model(x, mask, relative)[1]))
        saved = tail_artifact(model, artifact)
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        restored = load_tail_head(torch.load(stream), {'backbone_key': 'unit'}, 'tailrank', 'cpu')
        expected = model.from_summary(summary, moments)
        self.assertTrue(torch.equal(expected, restored(x, mask, relative)[0]))
        with self.assertRaises(ValueError):
            load_tail_head(saved, {}, 'fullrank', 'cpu')
        bad = copy.deepcopy(saved)
        bad['head']['actual.2.bias'][0] += 1
        with self.assertRaises(ValueError):
            load_tail_head(bad, {}, 'tailrank', 'cpu')


if __name__ == '__main__':
    unittest.main()

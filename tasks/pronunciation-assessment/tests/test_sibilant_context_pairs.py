import copy
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
from mdd.sibilant_context_pairs import make_plan, context, contrast_loss
from mdd.sibilant_context_pair_head import ContextPairHead, pair_artifact, load_pair_head
from mdd.sibilant_frozen_probe import FrozenProbe, projection_digest, READOUT
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES


class PairTest(unittest.TestCase):
    def test_exact_batch_exposure_rare_identity_and_changed_links(self):
        features, labels, y = [], {}, []
        for expected in ('ㅅ', 'ㅆ'):
            for cell in range(4):
                for kind in ('error', 'normal0', 'normal1'):
                    i = len(features)
                    actual = expected if kind.startswith('normal') else ('ㅆ' if expected == 'ㅅ' else 'ㅅ')
                    features.append({'canonical': expected, 'observed_V': 'ㅣ', 'feature_index': i})
                    labels[i] = {'uid': str(i), 'speaker': str(i), 'item': 'item'+str(cell), 'q': cell,
                        'expected': expected, 'actual': actual, 'position': 'initial', 'language': 'unit',
                        'label_kind': 'correct' if kind.startswith('normal') else 'pair_error'}
                    y.append(0 if actual == 'ㅅ' else 1)
        rare = []
        for cls in (2, 3, 4, 2, 3, 4, 2):
            rare.append(len(y))
            y.append(cls)
        unknown = len(y)
        y.append(-100)
        y = np.asarray(y)
        eligible = y >= 0
        original = np.asarray([[int(i) for i in np.flatnonzero(y == 0)[:4]]+
            [int(i) for i in np.flatnonzero(y == 1)[:4]]+rare+[unknown]]*4)
        a, m, s, records = make_plan(original, y, eligible, features, labels, 53)
        b = make_plan(original, y, eligible, features, labels, 53)
        for first, second in zip((a, m, s), b[:3]):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(a[:, 8:15], original[:, 8:15])
        self.assertTrue((a[:, 15] == -1).all())
        for step in range(len(a)):
            np.testing.assert_array_equal(np.sort(y[a[step, :8]]), np.sort(y[original[step, :8]]))
            for col in (0, 1):
                np.testing.assert_array_equal(np.sort(m[step, :, col]), np.sort(s[step, :, col]))
            for links, aligned in ((m[step], True), (s[step], False)):
                for ss, one_s in links:
                    i, j = int(a[step, ss]), int(a[step, one_s])
                    self.assertEqual(y[i], 1)
                    self.assertEqual(y[j], 0)
                    self.assertNotEqual(labels[i]['speaker'], labels[j]['speaker'])
                    self.assertEqual(context(features[i], labels[i]) == context(features[j], labels[j]), aligned)
        self.assertEqual(len(records), 4)

    def test_pair_gradient_uses_correct_actual_contrast_and_no_pad(self):
        z = torch.tensor([[0., 0., 1., 1., 1.], [0., 0., 1., 1., 1.]], requires_grad=True)
        loss = contrast_loss(z, [[0, 1]])
        loss.backward()
        self.assertLess(float(z.grad[0, 1]), 0)
        self.assertGreater(float(z.grad[0, 0]), 0)
        self.assertGreater(float(z.grad[1, 1]), 0)
        self.assertEqual(float(z.grad[:, 2:].abs().sum()), 0.)
        better = z.detach().clone()
        better[0, 1] += 2
        self.assertLess(float(contrast_loss(better, [[0, 1]])), float(loss))
        with self.assertRaises(ValueError):
            contrast_loss(z, [[0, -1]])

    def test_single_clip_initial_equivalence_frozen_role_and_trained_reload(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)
        projection = {'mean': torch.zeros(6), 'scale': torch.ones(6), 'components': torch.randn(6, 3)}
        base = FrozenProbe(projection).eval()
        artifact = {'readout': READOUT, 'projection': projection, 'head': copy.deepcopy(base.head.state_dict()),
            'binding': {'backbone_key': 'unit', 'classes': list(CLASSES),
                        'projection_sha256': projection_digest(projection)}}
        model = ContextPairHead(base, 'matched').eval()
        x, m, r = torch.randn(4, 8, 6), torch.ones(4, 8, dtype=torch.bool), torch.zeros(4, 8)
        self.assertTrue(torch.equal(model(x, m, r)[0], base(x, m, r)[0]))
        before = tensor_digest(base.state_dict())
        phase = base(x, m, r)[1].detach().clone()
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, m, r)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0, 2, 4]))+contrast_loss(logits, [[0, 1]])
            loss.backward()
            optimizer.step()
        self.assertEqual(before, tensor_digest(base.state_dict()))
        self.assertTrue(all(v.grad is None for v in model.head.support.parameters()))
        self.assertTrue(torch.equal(phase, model(x, m, r)[1]))
        saved = pair_artifact(model, artifact)
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = load_pair_head(torch.load(buffer), {'backbone_key': 'unit'}, 'matched', 'cpu')
        expected = model(x, m, r)[0]
        self.assertTrue(torch.equal(expected, restored(x, m, r)[0]))
        padded = torch.cat([x, torch.full((4, 5, 6), float('nan'))], 1)
        pm = torch.cat([m, torch.zeros(4, 5, dtype=torch.bool)], 1)
        pr = torch.cat([r, torch.zeros(4, 5)], 1)
        torch.testing.assert_close(restored(padded, pm, pr)[0], expected, atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            load_pair_head(saved, {}, 'shuffled', 'cpu')
        bad = copy.deepcopy(saved)
        bad['head']['actual.2.bias'][0] += 1
        with self.assertRaises(ValueError):
            load_pair_head(bad, {}, 'matched', 'cpu')
        changed = copy.deepcopy(saved)
        changed['head']['support.2.bias'][0] += 1
        changed['binding']['head_sha256'] = tensor_digest(changed['head'])
        with self.assertRaises(ValueError):
            load_pair_head(changed, {}, 'matched', 'cpu')


if __name__ == '__main__':
    unittest.main()

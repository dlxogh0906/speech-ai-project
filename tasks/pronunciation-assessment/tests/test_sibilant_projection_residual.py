import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
from mdd.sibilant_projection_residual import speaker_half, select_cells, full_observations, replication
from mdd.sibilant_frozen_probe import FrozenProbe
from mdd.sibilant_precision_tail import PrecisionTailHead


class ProjectionResidualTest(unittest.TestCase):
    def test_label_blind_dedup_global_speaker_halves_and_unknown_ledger(self):
        features, labels, y = [], {}, []
        halves = {h: [str(i) for i in range(100) if speaker_half(str(i), 37) == h][:4] for h in (0, 1)}
        for cell in range(2):
            for h in (0, 1):
                for j, speaker in enumerate(halves[h]):
                    i = len(y)
                    features.append({'uid': str(i), 'q_runtime': cell, 'canonical': 'ㅅ', 'observed_V': 'ㅣ'})
                    labels[i] = {'uid': str(i), 'speaker': speaker, 'item': str(cell), 'q': cell,
                        'expected': 'ㅅ', 'actual': ('ㅅ', 'ㅆ')[j % 2], 'position': 'initial', 'language': 'unit'}
                    y.append(j % 2)
        i = len(y)
        features.append(dict(features[0], uid='duplicate'))
        labels[i] = dict(labels[0], uid='duplicate')
        y.append(y[0])
        features.append(dict(features[0], uid='unknown'))
        labels[len(y)] = dict(labels[0], uid='unknown')
        y.append(-100)
        y = np.asarray(y)
        cells, ledger, flow = select_cells(features, labels, y, np.ones(len(y), bool), 37)
        self.assertEqual(len(cells), 2)
        self.assertEqual(flow['original_unknown'], 1)
        self.assertEqual(flow['same_speaker_context_repeat_removed'], 1)
        sets = [{labels[i]['speaker'] for c in cells for i in c['halves'][h]} for h in (0, 1)]
        self.assertFalse(sets[0] & sets[1])
        flipped = y.copy()
        flipped[y >= 0] = 1-y[y >= 0]
        other = copy.deepcopy(labels)
        for i in range(len(y)):
            if y[i] >= 0:
                other[i]['actual'] = ('ㅅ', 'ㅆ')[flipped[i]]
        _, ledger2, _ = select_cells(features, other, flipped, np.ones(len(y), bool), 37)
        self.assertEqual([c['indices'] for c in ledger], [c['indices'] for c in ledger2])

    def test_full_observations_commute_with_original_projection_and_ignore_pad(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)
        q, _ = torch.linalg.qr(torch.randn(10, 4))
        projection = {'mean': torch.randn(10), 'scale': torch.rand(10)+.5, 'components': q}
        base = FrozenProbe(projection).requires_grad_(False).eval()
        model = PrecisionTailHead(base, 'ce_full').eval()
        x, mask, relative = torch.randn(2, 9, 10), torch.ones(2, 9, dtype=torch.bool), torch.zeros(2, 9)
        mask[0, 7:] = False
        x[0, 7:] = float('nan')
        before = x.clone()
        full, phase = full_observations(base, x, mask, relative)
        kept, _, expected_phase = model.summarize(x, mask, relative)
        self.assertTrue(torch.equal(phase, expected_phase))
        torch.testing.assert_close((full@q).reshape(2, -1), kept, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(x.numpy(), before.numpy())

    def test_known_discarded_signal_vs_retained_only_and_small_i_count(self):
        rng = np.random.RandomState(17)
        cells, labels, rows = [], [], []
        for c in range(12):
            halves = []
            for _ in (0, 1):
                ix = []
                for actual in (0, 0, 1, 1):
                    ix.append(len(rows))
                    z = rng.normal(0, .01, (3, 4))
                    z[:2, 2] += actual*2
                    rows.append(z)
                    labels.append(actual)
                halves.append(ix)
            cells.append({'halves': halves, 'cell': ['unit', c, 'ㅅ', 'ㅣ']})
        full, y, q = np.asarray(rows), np.asarray(labels), np.eye(4)[:, :2]
        before = y.copy()
        result, null, delta = replication(cells, full, q, y, 100, 37)
        self.assertTrue(result['proceed_residual_candidate'])
        self.assertEqual(delta.shape, (12, 2, 3, 4))
        self.assertEqual(null['ㅣ'].shape, (100, 3, 3))
        np.testing.assert_array_equal(y, before)
        retained = full.copy()
        retained[:, :, :2] = full[:, :, 2:]
        retained[:, :, 2:] = 0
        result2, _, _ = replication(cells, retained, q, y, 100, 37)
        self.assertFalse(result2['proceed_residual_candidate'])
        result3, _, _ = replication(cells[:9], full, q, y, 100, 37)
        self.assertFalse(result3['proceed_residual_candidate'])


if __name__ == '__main__':
    unittest.main()

"""Training accounting and CTC objective invariants; no model download."""
import itertools
import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts/train'), str(ROOT / 'scripts/eval')]
from train_h0_backbone_control import ctc_loss, stream
from audit_h0_backbone_protocol import calibration_role, exposure
from eval_h0_backbone_control import (threshold_for_normal_budget, metrics, compare_trajectory,
                                    cluster_intervals, infer_utterance_sites)


class H0ControlTest(unittest.TestCase):
    def test_ctc_matches_enumerated_paths_and_has_gradient(self):
        logits = torch.tensor([[[.1, .8, -.3], [.5, -.2, .6], [.7, .2, -.1]],
                               [[.4, .2, .1], [.3, -.2, .8], [.1, .5, -.7]]], requires_grad=True)
        targets, lengths = [[1, 1], [2]], torch.tensor([3, 2])
        lp = logits.log_softmax(-1)
        expected = []
        for b, (y, n) in enumerate(zip(targets, lengths.tolist())):
            scores = []
            for p in itertools.product(range(3), repeat=n):
                collapsed = [v for v, _ in itertools.groupby(p) if v != 0]
                if collapsed == y:
                    scores.append(sum(lp[b, t, v] for t, v in enumerate(p)))
            expected.append(-torch.logsumexp(torch.stack(scores), 0) / len(y))
        loss = ctc_loss(logits, lengths, targets)
        torch.testing.assert_close(loss, torch.stack(expected).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_infeasible_label_not_silently_zeroed(self):
        with self.assertRaises(ValueError):
            ctc_loss(torch.zeros(1, 2, 3), torch.tensor([2]), [[1, 1]])

    def test_accounting_crosses_epoch_boundary_without_reset(self):
        import hashlib
        rows = [('a', 9), ('b', 8), ('c', 7)]
        gen = stream(rows, budget=10, seed=42)
        batches = [next(gen) for _ in range(8)]
        digest = hashlib.sha256()
        for batch in batches:
            digest.update((' '.join(map(str, batch)) + '\n').encode())
        report = exposure(rows, ['native'] * 3, [4], accum=2, budget=10)['4']
        self.assertEqual(report['microbatches'], 8)
        self.assertEqual(report['batch_order_sha256'], digest.hexdigest())
        self.assertEqual(report['row_exposures']['native'], 8)
        self.assertEqual(report['audio_samples']['native'], sum(rows[b[0]][1] for b in batches))

    def test_calibration_role_depends_only_on_speaker(self):
        seen = {str(i): calibration_role(str(i)) for i in range(100)}
        self.assertEqual(set(seen.values()), {'calibration', 'selection'})
        self.assertEqual(seen, {str(i): calibration_role(str(i)) for i in reversed(range(100))})

    def test_one_percent_threshold_keeps_ties_out_and_does_not_use_errors(self):
        sites = [dict(expected='ㅅ', actual='ㅅ', predicted='ㅆ', score=10. if i<2 else 0.)
                 for i in range(100)]
        threshold = threshold_for_normal_budget(sites)
        self.assertGreater(threshold, 10.)
        self.assertEqual(metrics(sites,threshold)['normal_false_alarms'], 0)
        error = dict(expected='ㅅ',actual='ㅆ',predicted='ㅆ',score=100.)
        self.assertEqual(threshold, threshold_for_normal_budget(sites+[error]))
        self.assertEqual(metrics(sites+[error],threshold)['exact'], 1)

    def test_wrong_actual_and_missing_score_remain_in_denominator(self):
        sites = [dict(expected='ㅅ',actual='ㅆ',predicted='ㅈ',score=4.),
                 dict(expected='ㅅ',actual='ㅆ',predicted='ㅆ',score=None)]
        got = metrics(sites, 1.)
        self.assertEqual(got['errors'], 2)
        self.assertEqual(got['exact'], 0)
        self.assertEqual(got['wrong_actual'], 1)
        self.assertEqual(got['unscorable_errors'], 1)

    def test_trajectory_is_not_extrapolated(self):
        def report(step,x,y):
            return dict(step=step,backbone='test',protocol_sha256='same',results={
                'valid':{'all':{'greedy':{'normal_fpr':x}}},
                'valid_unseen':{'all':{'greedy':{'normal_fpr':y}}}})
        baseline=[report(2000,.08,.14),report(4000,.04,.18)]
        got=compare_trajectory(baseline,[report(4000,.06,.15),report(8000,.01,.1)])
        self.assertAlmostEqual(got[0]['brackets'][0]['candidate_minus_baseline'],-.01)
        self.assertEqual(got[1]['status'],'outside_common_range')

    def test_cluster_bootstrap_keeps_repeated_sites_in_the_same_speaker(self):
        sites = [dict(speaker='a',expected='ㅅ',actual='ㅅ',predicted='ㅆ',score=2.)]*100
        sites += [dict(speaker='b',expected='ㅅ',actual='ㅅ',predicted='ㅅ',score=0.)]*100
        got = cluster_intervals(sites,1.,'speaker',repeats=1000)
        self.assertEqual(got['clusters'],2)
        # A site bootstrap would concentrate near .5, wrongly treating 200 sites as independent.
        self.assertEqual(got['normal_fpr']['ci95'],[0.,1.])
        self.assertIsNone(got['exact_recall']['ci95'])
        self.assertEqual(cluster_intervals(sites[:100],1.,'speaker')['status'],'insufficient_clusters')

    def test_actual_tag_cannot_change_inference_scores_or_prediction(self):
        class FixedModel:
            device = torch.device('cpu')
            def __call__(self,xs):
                logits = torch.full((1,4,7),-5.)
                for t,c in enumerate([5,0,6,0]): logits[0,t,c] = 5.
                return logits,torch.tensor([4])
        vocab={'ㅅ':4,'ㅆ':5,'ㅏ':6}
        meta=dict(stem='u',spk='p',item='i',task='RW',language='test')
        call=lambda actual:infer_utterance_sites(FixedModel(),[0.]*100,['ㅅ','ㅏ'],actual,
            vocab,meta,None,'valid',0)
        normal,error = call(['ㅅ','ㅏ']),call(['ㅆ','ㅏ'])
        self.assertEqual([(s['predicted'],s['score']) for s in normal],
                         [(s['predicted'],s['score']) for s in error])
        self.assertNotEqual(normal[0]['actual'],error[0]['actual'])


if __name__ == '__main__':
    unittest.main()

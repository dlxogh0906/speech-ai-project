"""Meaningful checks for label mapping, repeated features and CTC gradients."""
import itertools
import unittest
import torch
from mdd.phonological_aux import (VOWELS, COMPOSITES, schema, feature_targets,
                                  checked_ctc, minimum_frames, FeatureHeads, auxiliary_loss)


class PhonologicalAuxTests(unittest.TestCase):
    def setUp(self):
        tokens = sorted(VOWELS | COMPOSITES | set('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'))
        self.vocab = {t: i + 4 for i, t in enumerate(tokens)}
        self.spec = schema(self.vocab)

    def test_contrasts_and_unknowns(self):
        self.assertEqual(len(self.vocab), 46)
        for phones in ('ㅅㅆ', 'ㅈㅉㅊ'):
            for axis in ('manner', 'place'):
                self.assertEqual(len({self.spec[axis]['phone_to_id'][p] for p in phones}), 1)
            self.assertEqual(len({self.spec['series']['phone_to_id'][p] for p in phones}), len(phones))
        with self.assertRaises(ValueError):
            schema({'?': 4})
        self.assertNotEqual(self.spec['series']['phone_to_id']['ㅎ'],
                            self.spec['series']['phone_to_id']['ㅊ'])

    def test_actual_label_mapping_preserves_repeats(self):
        targets = feature_targets([[self.vocab[p] for p in 'ㅈㅉㅊ']], self.vocab, self.spec)
        self.assertEqual(len(targets['manner'][0]), 3)
        self.assertEqual(minimum_frames(targets['manner'][0]), 5)
        self.assertEqual(minimum_frames(targets['series'][0]), 3)

    def test_ctc_matches_enumeration_and_gradients(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 4, 3, requires_grad=True)
        target = [1, 1]
        total = logits.new_zeros(())
        probs = logits.softmax(-1)[0]
        for path in itertools.product(range(3), repeat=4):
            collapsed = [v for v, _ in itertools.groupby(path) if v != 0]
            if collapsed == target:
                total = total + torch.stack([probs[t, v] for t, v in enumerate(path)]).prod()
        expected = -total.log() / len(target)
        actual = checked_ctc(logits, torch.tensor([4]), [target])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(torch.allclose(ga, ge, atol=1e-6))
        with self.assertRaises(ValueError):
            checked_ctc(logits[:, :2], torch.tensor([2]), [target])

    def test_auxiliary_trains_shared_features(self):
        features = torch.randn(2, 12, 8, requires_grad=True)
        heads = FeatureHeads(8, self.spec)
        ys = [[self.vocab[p] for p in x] for x in ('ㅅㅏㅆ', 'ㅉㅣㅊ')]
        targets = feature_targets(ys, self.vocab, self.spec)
        loss, axes = auxiliary_loss(heads(features), torch.tensor([12, 12]), targets)
        loss.backward()
        self.assertEqual(set(axes), set(self.spec))
        self.assertGreater(float(features.grad.abs().sum()), 0)
        for head in heads.heads.values():
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()

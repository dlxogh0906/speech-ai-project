import itertools
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from mdd.h0_partial_reference import UNKNOWN, partial_mapping
from mdd.h0_partial_scoring import (mapping_indices, value_options, gap_value,
    partial_ctc_viterbi, annotation_mask, metric_record)


def accepts(ref, tokens):
    if not ref: return not tokens
    if ref[0] is None:
        return any(accepts(ref[1:], tokens[k:]) for k in range(len(tokens)+1))
    return bool(tokens) and tokens[0] == ref[0] and accepts(ref[1:], tokens[1:])


class PartialScoringTests(unittest.TestCase):
    def test_wildcard_ctc_against_all_raw_frame_paths(self):
        rng = np.random.RandomState(317)
        for T in (1, 2, 4, 5):
            raw = rng.randn(T, 3); lp = raw - np.log(np.exp(raw).sum(1))[:, None]
            for ref in ([], [1], [1,1], [1,2], [None], [1,None,1], [None,2,None], [1,None,None,2]):
                scores = []
                for path in itertools.product(range(3), repeat=T):
                    tokens = [x for x,_ in itertools.groupby(path) if x]
                    if accepts(ref, tokens): scores.append(sum(lp[t,c] for t,c in enumerate(path)))
                got = partial_ctc_viterbi(lp, ref, [1,2])
                if not scores: self.assertIsNone(got); continue
                self.assertAlmostEqual(got['score'], max(scores), places=10)
                self.assertTrue(accepts(ref, [x for x,_ in itertools.groupby(got['symbols']) if x]))
                for q, y in enumerate(ref):
                    if y is not None:
                        self.assertTrue(got['segments'][q])
                        self.assertTrue(all(got['symbols'][t] == y for t in got['segments'][q]))

    def test_occurrence_mapping_agrees_with_frozen_value_mapping(self):
        words = [list(x) for n in range(3) for x in itertools.product(['a','b',UNKNOWN],repeat=n)]
        for ref, tar in itertools.product(words, repeat=2):
            indices = mapping_indices(ref, tar)
            self.assertEqual([value_options(v,tar) for v in indices], partial_mapping(ref,tar)[1])

    def test_insertion_requires_two_unique_known_bounds(self):
        ref = list('ab'); tar = list('axb')
        self.assertEqual(gap_value(ref,tar,mapping_indices(ref,tar),1),('x',))
        self.assertIsNone(gap_value(ref,tar,mapping_indices(ref,tar),0))
        tar = ['a',UNKNOWN,'b']
        self.assertIsNone(gap_value(ref,tar,mapping_indices(ref,tar),1))
        self.assertIsNone(gap_value(list('aa'),list('a'),mapping_indices('aa','a'),1))

    def test_boundary_and_missing_prediction_cannot_improve_unconditional_fpr(self):
        record = dict(sites=[dict(status='normal'),dict(status='deletion'),dict(status='normal')])
        self.assertEqual(annotation_mask(record,mapping_indices('abc','ac'))[1],None)
        s = dict(expected='a',actual='a',annotation_exclusion=None,score=5.,
                 nearest_seam_samples=0,prediction_options=['b'],prediction_deletion_anchored=True)
        got = metric_record([s],4.,640)
        self.assertEqual(got['normal'],1)
        self.assertEqual(got['fp_upper'],1)
        self.assertEqual(got['unobserved_normal'],1)


if __name__ == '__main__': unittest.main()

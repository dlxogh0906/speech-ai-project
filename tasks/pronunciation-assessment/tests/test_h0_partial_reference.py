"""Partial references must not turn missing labels or chunk boundaries into truth."""
import itertools
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.h0_partial_reference import (
    UNKNOWN, UNOBSERVED, DELETED, partial_tokens, partial_mapping, classify_truth,
    known_site_bounds, input_windows, owned_ctc_frames, collapse_owned_frames)
from mdd.h0_phone_audit import optimal_mapping


class PartialReferenceTests(unittest.TestCase):
    def test_fully_known_matches_existing_all_path_audit(self):
        words = [''.join(x) for n in range(4) for x in itertools.product('ab', repeat=n)]
        for ref, target in itertools.product(words, repeat=2):
            distance, options = partial_mapping(ref, target)
            previous = optimal_mapping(ref, target)
            expected = [frozenset(DELETED if j is None else target[j] for j in js)
                        for js in previous]
            self.assertEqual(options, expected, (ref, target, distance))

    def test_unknown_and_explicit_deletion_are_different(self):
        intervals = [[0, .1, 'ㅜ'], [.1, .2, '-'], [.2, .3, 'ㅗ?'], [.3, .4, '']]
        tag, owners, ignored = partial_tokens(intervals, set('ㅜㅗ'))
        ref, _, _ = partial_tokens(intervals, set('ㅜㅗ'), reference=True)
        self.assertEqual(tag, ['ㅜ', UNKNOWN])
        self.assertEqual(ref, ['ㅜ', UNKNOWN, UNKNOWN])
        self.assertEqual(owners[1]['raw_label'], 'ㅗ?')
        self.assertEqual([x['reason'] for x in ignored], ['explicit_deletion', 'empty'])
        _, values = partial_mapping(['ㅅ', 'ㅜ', 'ㄹ'], ['ㅅ', UNKNOWN, 'ㄹ'])
        self.assertEqual(classify_truth('ㅜ', values[1]), ('unknown_actual_possible', None))
        _, values = partial_mapping(['ㅅ', 'ㅜ', 'ㄹ'], ['ㅅ', 'ㄹ'])
        self.assertEqual(classify_truth('ㅜ', values[1]), ('deletion', DELETED))

    def test_wildcard_zero_many_and_alignment_ambiguity(self):
        for fill in ([], ['x'], ['x', 'y', 'z']):
            distance, options = partial_mapping(['a', UNKNOWN, 'b'], ['a'] + fill + ['b'])
            self.assertEqual(distance, 0)
            self.assertEqual(options, [frozenset('a'), None, frozenset('b')])
        _, values = partial_mapping('aaa', [UNKNOWN, 'a', UNKNOWN])
        self.assertTrue(all(UNOBSERVED in x for x in values))
        _, values = partial_mapping('aa', 'a')
        self.assertEqual(values, [frozenset(['a', DELETED])] * 2)

    def test_prediction_uncertainty_never_drops_denominator(self):
        ref = ['a', 'b', UNKNOWN, 'c']
        truth = [frozenset('a'), frozenset('x'), None, frozenset([UNOBSERVED])]
        pred = [frozenset('az'), frozenset('xy'), None, frozenset('c')]
        got = known_site_bounds(ref, truth, pred)
        self.assertEqual((got['normal'], got['errors'], got['unidentifiable_entries']), (1, 1, 2))
        self.assertEqual((got['normal_false_lower'], got['normal_false_upper']), (0, 1))
        self.assertEqual((got['exact_lower'], got['exact_upper']), (0, 1))

    def test_audio_windows_keep_every_sample_and_halo(self):
        for n in (0, 1, 191999, 192000, 192001, 630001):
            windows = input_windows(n)
            self.assertEqual(sum(w['core_end'] - w['core_start'] for w in windows), n)
            for i, w in enumerate(windows):
                self.assertLessEqual(w['read_start'], w['core_start'])
                self.assertGreaterEqual(w['read_end'], w['core_end'])
                self.assertLessEqual(w['read_end'], n)
                if i:
                    self.assertEqual(windows[i-1]['core_end'], w['core_start'])

    def test_ctc_collapse_occurs_after_core_stitching(self):
        windows = input_windows(384000)
        pieces = []
        for w in windows:
            nframes = (w['read_end'] - w['read_start']) // 320
            pieces.extend(owned_ctc_frames(['a'] * nframes, w))
        self.assertEqual(len(pieces), 1200)
        self.assertEqual([t for t, _ in pieces], list(range(280, 384000, 320)))
        self.assertEqual(collapse_owned_frames(pieces, '_'), ['a'])
        pieces[600] = (pieces[600][0], '_')
        self.assertEqual(collapse_owned_frames(pieces, '_'), ['a', 'a'])
        with self.assertRaises(ValueError):
            collapse_owned_frames([(1, 'a'), (1, 'b')], '_')


if __name__ == '__main__':
    unittest.main()

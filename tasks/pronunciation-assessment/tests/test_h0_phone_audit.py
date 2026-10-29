"""Numerical/representation contracts for all-phone H0 diagnostics."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT/'tasks/pronunciation-assessment/src')]
import unittest
import itertools
from mdd.h0_phone_audit import (contexts, events, optimal_mapping, paired_cluster_ci,
                              project_gaps, sequence_audit, standardize, surface_position)


class AuditTests(unittest.TestCase):
    def test_error_partition_and_policy_suppression(self):
        for actual in ('ㄱ','ㅋ',None):
            for predicted in ('ㄱ','ㅋ','ㅌ',None):
                s=dict(expected='ㄱ',actual=actual,predicted=predicted,score=-1)
                e=events(s,0)
                self.assertEqual(e['normal']+e['errors'],1)
                self.assertEqual(e['error_revert']+e['error_exact']+e['error_wrong'],e['errors'])
                self.assertEqual(e['normal_fp_substitution']+e['normal_fp_deletion'],e['normal_fp'])
                self.assertEqual(e['policy_fp']+e['policy_exact']+e['policy_wrong'],0)
                self.assertEqual(e['error_exact'],e['exact_suppressed'])
        self.assertEqual(events(dict(expected='ㄱ',actual='ㄱ',predicted='ㄱ',score=100),0)['policy_fp'],0)

    def test_equal_sequences_and_gap_loss(self):
        canon=list('ㄱㅏ');actual=list('ㄱㅏㄴ')
        sites,gaps=project_gaps(canon,actual)
        self.assertEqual(sites,canon)
        self.assertEqual(gaps,[[],[],['ㄴ']])
        z=sequence_audit(canon,actual,actual)
        self.assertEqual(z['sequence_exact'],1)
        self.assertEqual(z.get('canonical_incorrect',0),0)
        self.assertEqual(z['insertion_exact_gaps'],1)
        missed=sequence_audit(canon,actual,canon)
        self.assertEqual(missed['insertion_missed_gaps'],1)
        # Identical projected sites can hide a real sequence error.
        self.assertEqual(project_gaps(canon,actual)[0],project_gaps(canon,canon)[0])

    def test_repeated_token_alignment_is_not_unique(self):
        opts=optimal_mapping(tuple('ㄱㄱ'),tuple('ㄱ'))
        self.assertEqual(opts,(frozenset([0,None]),frozenset([0,None])))
        self.assertEqual(optimal_mapping(tuple('ㄱㅏ'),tuple('ㄱㅏ')),
                         (frozenset([0]),frozenset([1])))

    def test_all_optimal_mapping_against_enumerated_edit_paths(self):
        strings = [tuple(x) for n in range(4) for x in itertools.product('ab', repeat=n)]
        for ref in strings:
            for target in strings:
                paths = []
                def enumerate_paths(i, j, cost, mapping):
                    if i == len(ref) and j == len(target):
                        paths.append((cost, mapping)); return
                    if i < len(ref): enumerate_paths(i+1,j,cost+1,mapping+[None])
                    if j < len(target): enumerate_paths(i,j+1,cost+1,mapping)
                    if i < len(ref) and j < len(target):
                        enumerate_paths(i+1,j+1,cost+(ref[i]!=target[j]),mapping+[j])
                enumerate_paths(0,0,0,[])
                best = min(cost for cost, _ in paths)
                expected = tuple(frozenset(path[i] for cost,path in paths if cost==best)
                                 for i in range(len(ref)))
                self.assertEqual(optimal_mapping(ref,target),expected)

    def test_surface_position_and_candidate_rules(self):
        self.assertEqual(surface_position(list('ㅂㅕㅇㅝㄴ'),2),'coda')
        self.assertEqual(surface_position(list('ㄱㅏㄴㅏ'),2),'onset')
        c=contexts(tuple('ㄱㅜㄱㅃㅏㅂ'),'국밥')
        self.assertIn('tensification_candidate',c[3]['rule_candidates'])
        self.assertTrue(c[3]['post_coda_onset'])
        c=contexts(tuple('ㅅㅏㄴㅔ'),'산에')
        self.assertIn('liaison_candidate',c[2]['rule_candidates'])

    def test_matching_removes_composition_only_shift(self):
        data={'a':{'valid':{'normal':90,'normal_fp':9},'valid_unseen':{'normal':10,'normal_fp':1}},
              'b':{'valid':{'normal':10,'normal_fp':5},'valid_unseen':{'normal':90,'normal_fp':45}},
              'unmatched':{'valid_unseen':{'normal':100,'normal_fp':100}}}
        r=standardize(data,'normal_fp','normal')
        self.assertEqual(r['unseen_denominator'],100)
        self.assertAlmostEqual(r['seen_raw_common'],.14)
        self.assertAlmostEqual(r['unseen_raw_common'],.46)
        self.assertAlmostEqual(r['within_context_gap'],0)
        self.assertAlmostEqual(r['composition_component'],.32)

    def test_paired_resampling_preserves_difference(self):
        r=paired_cluster_ci([{'normal':100,'d':0},{'normal':1,'d':0}],'d','normal')
        self.assertEqual(r['ci95'],[0,0])
        r=paired_cluster_ci([{'normal':100,'d':1}],'d','normal')
        self.assertIsNone(r['ci95'])


if __name__=='__main__':unittest.main()

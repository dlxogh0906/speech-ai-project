"""Unknown-span alignment against literal expansion and original-source controls."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import itertools
import unittest
from functools import lru_cache
from eval_sibilant_directional import all_optimal_mapping
from mdd.sibilant_partial_annotation import partial_mapping,UNKNOWN,recover_sites


def expansions(sequence):
    candidates = [()] + list(itertools.product('ab',repeat=1)) + list(itertools.product('ab',repeat=2))
    products = [candidates if x is None else [(x,)] for x in sequence]
    for choices in itertools.product(*products):
        yield [(c,i if sequence[i] is not None else UNKNOWN) for i,value in enumerate(choices) for c in value]


def expanded_oracle(reference,target):
    best,options = 10**9,[set() for _ in reference]
    for ref in expansions(reference):
        for actual in expansions(target):
            a,b = [x[0] for x in ref],[x[0] for x in actual]
            @lru_cache(None)
            def distance(i,j):
                if i == len(a):return len(b)-j
                if j == len(b):return len(a)-i
                return min(1+distance(i+1,j),1+distance(i,j+1),(a[i]!=b[j])+distance(i+1,j+1))
            score = distance(0,0)
            if score > best:continue
            if score < best:
                best,options = score,[set() for _ in reference]
            mapping = all_optimal_mapping(a,b)
            for i,(_,source_index) in enumerate(ref):
                if source_index == UNKNOWN:continue
                options[source_index].update(None if j is None else actual[j][1] for j in mapping[i])
    for i,x in enumerate(reference):
        if x is None:options[i] = {UNKNOWN}
    return options,best


class PartialAnnotationTest(unittest.TestCase):
    def test_no_unknown_matches_existing_full_alignment(self):
        sequences = [list(x) for n in range(4) for x in itertools.product('ab',repeat=n)]
        for a in sequences:
            for b in sequences:
                self.assertEqual(partial_mapping(a,b)[0],all_optimal_mapping(a,b))

    def test_unknown_graph_matches_independent_literal_expansion(self):
        cases = [([None,'a'],['b',None]),(['a',None,'b'],[None,'a','b']),
                 ([None,'a'],[None,'b']),(['a'],[None]),([None],['b']),
                 (['a','b'],['a',None,'b'])]
        for a,b in cases:
            self.assertEqual(partial_mapping(a,b),expanded_oracle(a,b),(a,b))

    def test_equal_partial_sequences_are_not_automatically_exact(self):
        options,score = partial_mapping([None,'ㅅ',None],[None,'ㅅ',None])
        self.assertEqual(score,0)
        self.assertEqual(options[1],{UNKNOWN,1})
        self.assertEqual(partial_mapping(['ㅅ','ㅏ'],[None])[0],[{UNKNOWN},{UNKNOWN}])

    def test_only_literal_singleton_pair_and_vowel_can_be_recovered(self):
        tag = [[0,.1,'spn'],[.1,.2,'ㄱ'],[.2,.3,'ㅆ'],[.3,.4,'ㅏ'],[.4,.5,'ㄴ']]
        lab = [[0,.1,'spn'],[.1,.2,'ㄱ'],[.2,.3,'ㅅ'],[.3,.4,'ㅏ'],[.4,.5,'ㄴ']]
        meta = {'tag_phones':tag,'lab_phones':lab,'ltr':list('ㄱㅆㅏㄴ'),'canon':list('ㄱㅅㅏㄴ')}
        allowed = set('ㄱㅅㅆㅏㄴ')
        site = recover_sites(meta,allowed,8000)['sites'][0]
        self.assertEqual(site['label_kind'],'pair_error')
        self.assertEqual(site['actual_raw_TAG_index'],2)
        self.assertEqual(site['C_TAG_interval'],[.2,.3,'ㅆ'])
        grouped = dict(meta,tag_phones=tag[:2]+[[.2,.4,'ㅆㅏ']]+tag[4:])
        self.assertEqual(recover_sites(grouped,allowed,8000)['sites'][0]['state'],'grouped_or_missing_actual_geometry')
        bad = dict(meta,tag_phones=tag[:3]+[[.25,.4,'ㅏ']]+tag[4:])
        self.assertEqual(recover_sites(bad,allowed,8000)['sites'][0]['state'],'invalid_or_outside_original_CV_geometry')


if __name__ == '__main__':unittest.main()

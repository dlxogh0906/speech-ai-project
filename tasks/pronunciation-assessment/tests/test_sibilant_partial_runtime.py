"""Runtime reference identity and unavailable inputs cannot manufacture labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import copy
import unittest
from mdd.sibilant_partial_runtime import runtime_labels


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.row = {'uid':'one','speaker':'s','item':'i','expected_partial':['ㄴ',None,'ㄱ','ㅅ','ㅏ','ㄴ'],
            'sites':[{'partial_q':3,'expected':'ㅅ','following':'ㅏ','source_time_witness':True,
                      'label_kind':'pair_error','actual':'ㅆ'}]}
        self.canonical = list('ㄴㅣㄱㅅㅏㄴ')
        self.phones = [{'alpha_jamo':c} for c in self.canonical]

    def test_exact_runtime_CV_after_unknown_prefix(self):
        z = runtime_labels(self.row,self.canonical,self.phones,{3:17},None)[0]
        self.assertEqual((z['q_runtime'],z['feature_index'],z['actual']),(3,17,'ㅆ'))

    def test_runtime_reference_substitution_is_not_a_label_match(self):
        canonical = self.canonical.copy();canonical[3] = 'ㅆ'
        z = runtime_labels(self.row,canonical,self.phones,{3:17},None)[0]
        self.assertIsNone(z['q_runtime']);self.assertIsNone(z['feature_index'])

    def test_rejected_record_retains_known_error_without_feature(self):
        z = runtime_labels(self.row,None,[],{},'low_speech')[0]
        self.assertEqual((z['label_kind'],z['actual']),('pair_error','ㅆ'))
        self.assertIsNone(z['feature_index']);self.assertIsNone(z['existing']['phone'])

    def test_unwitnessed_and_duplicate_source_sites_cannot_supervise_input(self):
        row = copy.deepcopy(self.row);row['sites'][0]['source_time_witness'] = False
        row['sites'][0]['label_kind'] = None
        z = runtime_labels(row,self.canonical,self.phones,{3:17},None)[0]
        self.assertEqual(z['label_kind'],'ambiguous');self.assertIsNone(z['actual']);self.assertIsNone(z['feature_index'])
        row = copy.deepcopy(self.row);row['sites'] *= 2
        self.assertTrue(all(z['feature_index'] is None for z in runtime_labels(row,self.canonical,self.phones,{3:17},None)))


if __name__ == '__main__':unittest.main()

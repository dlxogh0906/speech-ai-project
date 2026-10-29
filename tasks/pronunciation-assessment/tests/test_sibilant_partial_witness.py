"""Independent source time slots reject unsupported partial edit correspondences."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import copy
import unittest
from mdd.sibilant_partial_annotation import recover_sites
from mdd.sibilant_partial_witness import witness_sites


class WitnessTest(unittest.TestCase):
    def setUp(self):
        self.allowed = set('ㄱㅅㅆㅏㄴ')
        tag = [[0,.1,'spn'],[.1,.2,'ㄱ'],[.2,.3,'ㅆ'],[.3,.4,'ㅏ'],[.4,.5,'ㄴ']]
        lab = copy.deepcopy(tag);lab[2][2] = 'ㅅ'
        meta = {'tag_phones':tag,'lab_phones':lab,'ltr':list('ㄱㅆㅏㄴ'),'canon':list('ㄱㅅㅏㄴ')}
        self.row = dict(meta,frames=8000,sites=recover_sites(meta,self.allowed,8000)['sites'])

    def test_literal_labels_share_source_slots_without_becoming_runtime_labels(self):
        result = witness_sites(self.row,self.allowed)[0]
        self.assertTrue(result['source_time_witness'])
        self.assertFalse(result['usable_training_label'])

    def test_unique_edit_mapping_cannot_override_disagreeing_expected_time(self):
        row = copy.deepcopy(self.row);row['lab_phones'][2][:2] = [.21,.29]
        self.assertEqual(witness_sites(row,self.allowed)[0]['witness_state'],'TAG_LAB_CV_slots_disagree')

    def test_unknown_or_duplicate_annotation_inside_target_is_not_ignored(self):
        row = copy.deepcopy(self.row);row['tag_phones'].append([.25,.26,'spn'])
        self.assertEqual(witness_sites(row,self.allowed)[0]['witness_state'],'extra_annotation_inside_CV')
        row = copy.deepcopy(self.row);row['lab_phones'].append([.2,.3,'ㅅ'])
        self.assertEqual(witness_sites(row,self.allowed)[0]['witness_state'],'nonunique_source_time_slots')

    def test_unlocated_unknown_annotation_is_not_assumed_elsewhere(self):
        row = copy.deepcopy(self.row);row['tag_phones'][0][0] = None
        self.assertEqual(witness_sites(row,self.allowed)[0]['witness_state'],'unlocated_annotation_interval')


if __name__ == '__main__':unittest.main()

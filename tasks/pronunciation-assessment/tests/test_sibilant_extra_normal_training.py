"""Causal exposure matching and a shared, variable false-positive budget."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/data'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import unittest
import numpy as np
from train_sibilant_extra_normals import matched_plans, proxy


class ExtraNormalTrainingTest(unittest.TestCase):
    def test_only_normal_audio_changes_and_unknown_skip_survives(self):
        labels = {i:{'uid':str(i), 'q':0, 'expected':'ㅅ', 'following':'ㅣ',
            'label_kind':'pair_error' if i == 0 else 'correct'} for i in range(4)}
        rows = {str(i):{'item':'item', 'utterance_all_correct':i != 0,
            'sites':[{'label_kind':'pair_error' if i == 0 else 'correct'}]} for i in range(4)}
        features = [{'canonical':'ㅅ'} for _ in range(4)]
        plans, changed = matched_plans(np.array([[0,1,2],[0,1,2]]), np.array([1,0,0,0]),
            labels, features, rows, [{'target':True},{'target':True},{'target':None}], 3, 4)
        self.assertEqual(changed, 2)
        np.testing.assert_array_equal(plans[0], [[0,1,-1],[0,1,-1]])
        np.testing.assert_array_equal(plans[1], [[0,3,-1],[0,3,-1]])

    def test_global_budget_counts_both_directions_and_keeps_ties_together(self):
        ps = np.array([[.1,.9],[.05,.95],[.9,.1],[.8,.2]])
        labels = {i:{'label_kind':'pair_error' if i in (0,2) else 'correct',
            'expected':'ㅅ' if i < 2 else 'ㅆ', 'existing':{'phone':None}}
            for i in range(4)}
        features = [{'canonical':'ㅅ' if i < 2 else 'ㅆ'} for i in range(4)]
        y, av = np.array([1,0,0,1]), np.ones(4,bool)
        zero = proxy(ps,av,y,features,labels,0)
        one = proxy(ps,av,y,features,labels,1)
        self.assertEqual((zero['novel_TP'],zero['normal_FP']), (1,0))
        self.assertEqual((one['novel_TP'],one['normal_FP']), (2,1))
        tied = ps.copy()
        tied[1] = tied[0]
        self.assertEqual(proxy(tied,av,y,features,labels,0)['novel_TP'],1)


if __name__ == '__main__':
    unittest.main()

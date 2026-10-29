"""Speaker integrity, tied/unseen scores, and partial-source normal selection."""
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import numpy as np
from mdd.sibilant_source_shift import (auc,context_lookup,speaker_fold,
    matched_source_pairs,prototype_normal_targets,ridge_source_probe)


class ShiftTest(unittest.TestCase):
    def test_tied_auc_and_unseen_metadata_cells_remain_in_denominator(self):
        self.assertEqual(auc([0,1,0,1],[.5,.5,.5,.5]),.5)
        self.assertEqual(auc([0,0,1,1],[0.,0.,1.,1.]),1.)
        speakers = [next(str(i) for i in range(100) if speaker_fold(str(i))==fold) for fold in (0,1)]
        labels,features = {},[]
        for i,(fold,kind,item) in enumerate([(0,'correct','A'),(0,'pair_error','B'),(1,'correct','A'),(1,'pair_error','C')]):
            labels[i] = {'speaker':speakers[fold],'uid':str(i)+'-M-x','item':item,'position':'initial',
                         'language':'L','label_kind':kind}
            features.append({'q_runtime':0,'canonical':'ㅅ','expected_V':'ㅏ'})
        z = context_lookup(list(labels),labels,features,7)
        self.assertEqual(z['observations'],4);self.assertEqual(z['observed_contexts'],2)
        self.assertEqual(z['unseen_contexts_chance_kept'],2)

    def test_source_pairs_have_identical_context_actual_and_disjoint_speaker_folds(self):
        labels,features,y = {},[],[]
        speakers = [next(str(i) for i in range(100) if speaker_fold(str(i))==fold) for fold in (0,1)]
        for source in (0,1):
            for fold in (0,1):
                for cls in (0,1):
                    i = len(y);y.append(cls)
                    labels[i] = {'speaker':speakers[fold],'uid':str(i)+'-M-x','item':'A',
                        'position':'initial','language':'L','expected':'ㅅ','following':'ㅏ',
                        'label_kind':'correct' if cls==0 else 'pair_error'}
                    features.append({'q_runtime':0,'canonical':'ㅅ','expected_V':'ㅏ'})
        pairs,quad,used = matched_source_pairs(np.asarray(y),labels,features,4)
        self.assertEqual([len(z) for z in pairs],[2,2]);self.assertEqual([len(z) for z in quad],[1,1])
        self.assertEqual(len(used),8)

    def test_normal_requires_literal_normal_and_unique_partial_CV_mapping(self):
        row = {'uid':'u-M-x','item':'A','language':'L','expected_partial':['ㄴ','ㅏ','ㅆ','ㅓ'],
            'sites':[{'partial_q':2,'expected':'ㅆ','actual':'ㅆ','following':'ㅓ','position':'post_vowel',
                      'label_kind':'correct','source_time_witness':True}]}
        selected = prototype_normal_targets(row,['ㄴ','ㅏ','ㅆ','ㅓ'])
        self.assertEqual(len(selected),1);self.assertEqual(selected[0][2],2)
        row['sites'][0]['actual']='ㅅ';self.assertEqual(prototype_normal_targets(row,['ㄴ','ㅏ','ㅆ','ㅓ']),[])
        row['sites'][0]['actual']='ㅆ';row['expected_partial']=[None,'ㅆ','ㅓ',None];row['sites'][0]['partial_q']=1
        self.assertEqual(prototype_normal_targets(row,['ㅆ','ㅓ','ㅆ','ㅓ']),[])

    def test_probe_detects_a_source_axis_on_separated_groups_with_permutation_control(self):
        x = np.zeros((480,2));pairs=[]
        for fold in (0,1):
            group=[]
            for k in range(120):
                a,b=fold*240+2*k,fold*240+2*k+1;group.append((a,b))
                x[a]=[-1.,k%3];x[b]=[1.,k%3]
            pairs.append(np.asarray(group))
        results = ridge_source_probe(x,pairs,permutations=20)
        self.assertTrue(all(z['balanced_accuracy']==1. for z in results))
        # A tiny randomly signed coefficient can still classify a perfectly
        # separated one-dimensional fixture. Use its magnitude as the null
        # statistic rather than assuming null sign accuracy is always near .5.
        self.assertTrue(all(z['mean_new_minus_old_score_margin']>z['null_margin_95percentile'] for z in results))


if __name__=='__main__':unittest.main()

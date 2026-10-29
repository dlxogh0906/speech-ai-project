"""Checks against lost old errors, scope mixing, and FP-budget tie splitting."""
import copy
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import numpy as np
import torch
from mdd.sibilant_conditional_retained import (retained_plans,new_conditional,
    conditional_artifact,load_conditional,fixed_direction_proxy)
from mdd.sibilant_frozen_probe import FrozenProbe,projection_digest,READOUT
from mdd.sibilant_vowel_presence import CLASSES


class ConditionalTest(unittest.TestCase):
    def fixture(self):
        tokens = ['ㅅ']*4+['ㅆ']*4+['ㅅ','ㅆ','ㅅ','ㅆ']
        y = np.asarray([1,0,2,4,0,1,2,4,1,0,1,0])
        features = [{'canonical':t,'expected_V':'ㅏ','q_runtime':0} for t in tokens]
        labels = {i:{'uid':str(i),'expected':t,'actual':'ㅆ' if y[i]==1 else 'ㅅ',
            'following':'ㅏ','label_kind':(['pair_error','correct','third_phone','deletion'][i%4]
                if i<8 else 'pair_error'),'existing':{'phone':None}} for i,t in enumerate(tokens)}
        return np.arange(8).reshape(2,4),y,labels,features,[{'target':['x']} for _ in range(10)]

    def test_all_original_errors_retained_and_all_new_seen_without_nonpair_change(self):
        original,y,labels,features,eligible = self.fixture()
        plans,report = retained_plans(original,y,labels,features,eligible,10,3,steps=5,batch=4)
        for token,first,old_extra,new_extra in [('ㅅ',0,8,10),('ㅆ',4,9,11)]:
            old,new = plans[token]['old'].ravel(),plans[token]['new'].ravel()
            np.testing.assert_array_equal(old[:4],np.arange(first,first+4))
            np.testing.assert_array_equal(new[:4],np.arange(first,first+4))
            self.assertIn(first,new);self.assertIn(new_extra,new)
            self.assertEqual(int((old==old_extra).sum()),int((new==new_extra).sum()))
            for a,b in zip(old,new):
                if labels[int(a)]['label_kind']!='pair_error':self.assertEqual(a,b)
            self.assertEqual(report[token]['new_error_features'],1)
        with self.assertRaises(ValueError):
            retained_plans(original,y,labels,features,eligible,10,3,steps=1,batch=4)
        y[0] = -100
        with self.assertRaises(ValueError):
            retained_plans(original,y,labels,features,eligible,10,3,steps=5,batch=4)

    def test_runtime_route_reload_scope_and_frozen_support(self):
        torch.manual_seed(5);torch.set_num_threads(2)
        projection = {'mean':torch.zeros(6),'scale':torch.ones(6),'components':torch.randn(6,3)}
        probe = FrozenProbe(projection).eval()
        base = {'readout':READOUT,'projection':projection,'head':copy.deepcopy(probe.head.state_dict()),
            'binding':{'backbone_key':'unit','classes':list(CLASSES),
                       'projection_sha256':projection_digest(projection)}}
        model = new_conditional(base,base['binding'],'cpu')
        x,t,route = torch.randn(4,9),torch.randn(4,9),torch.tensor([0,1,1,0])
        with torch.no_grad():model.heads[1].head.actual[2].bias.add_(torch.arange(5).float())
        expected = torch.stack([model.heads[int(route[i])].from_summary(x[i:i+1],t[i:i+1])[0] for i in range(4)])
        torch.testing.assert_close(model.from_summary(x,t,route),expected)
        artifact = conditional_artifact(model,base)
        loaded = load_conditional(artifact,base['binding'],'cpu')
        self.assertTrue(torch.equal(model.from_summary(x,t,route),loaded.from_summary(x,t,route)))
        with self.assertRaises(ValueError):model.from_summary(x,t,torch.tensor([0,1,2,0]))
        swapped = copy.deepcopy(artifact);swapped['heads']['ㅅ'],swapped['heads']['ㅆ'] = swapped['heads']['ㅆ'],swapped['heads']['ㅅ']
        with self.assertRaises(ValueError):load_conditional(swapped,base['binding'],'cpu')
        damaged = copy.deepcopy(artifact)
        key = next(k for k in damaged['heads']['ㅅ']['head'] if k.startswith('support.'))
        damaged['heads']['ㅅ']['head'][key].add_(1.)
        with self.assertRaises(ValueError):load_conditional(damaged,base['binding'],'cpu')

    def test_tied_normal_scores_and_unknowns_cannot_buy_error_recall(self):
        features = [{'canonical':'ㅅ'}]*5+[{'canonical':'ㅆ'}]*2
        labels = {i:{'label_kind':'correct' if i in (0,1,5) else 'pair_error',
            'expected':features[i]['canonical'],'actual':'ㅆ' if i<5 else 'ㅅ',
            'existing':{'phone':None}} for i in range(7)}
        ps = np.asarray([[.1,.9,0,0,0],[.1,.9,0,0,0],[.1,.9,0,0,0],
            [.05,.95,0,0,0],[0,1,0,0,0],[.8,.2,0,0,0],[.95,.05,0,0,0]])
        y = np.asarray([0,0,1,1,-100,1,0])
        result = fixed_direction_proxy(ps,np.ones(7,bool),y,features,labels,(1,0))
        self.assertEqual(result['S_to_SS']['novel_TP'],1)
        self.assertEqual(result['S_to_SS']['normal_FP'],0)
        self.assertEqual(result['SS_to_S']['novel_TP'],1)
        self.assertEqual(result['SS_to_S']['normal_FP'],0)
        self.assertEqual(result['novel_TP'],2)


if __name__=='__main__':unittest.main()

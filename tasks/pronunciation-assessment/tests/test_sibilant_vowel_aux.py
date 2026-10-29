import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_vowel_aux import IDS,VowelAuxReadout,intended_vowel_target,vowel_loss


class VowelAuxTest(unittest.TestCase):
    def test_actual_vowel_not_expected_identity(self):
        row={'actual':['ㅅ','ㅜ'],'canonical':['ㅅ','ㅣ']};label={'q':0,'a':0,'alignment_ambiguous':False}
        y,state=intended_vowel_target(label,row,[{0},{1}],np.array([1,2,3]))
        self.assertEqual(y,IDS['ㅜ']);self.assertEqual(state,'known_actual_intended_V')

    def test_does_not_borrow_inserted_or_later_vowel(self):
        row={'actual':['ㅅ','ㅣ','ㅜ']};label={'q':0,'a':0,'alignment_ambiguous':False}
        for mapping in ([{0},{2}],[{0},{None}],[{0},{1,2}]):
            self.assertEqual(intended_vowel_target(label,row,mapping)[0],-100)
        self.assertEqual(intended_vowel_target(label,row,[{0},{1}],np.array([0,1,-100]))[0],-100)

    def test_initial_actual_logits_and_roles_unchanged(self):
        torch.manual_seed(4);old=SupportReadout(5,True);new=VowelAuxReadout(5);new.load_base(old.state_dict())
        h=torch.randn(2,7,5);mask=torch.arange(7)[None,:]<torch.tensor([7,5])[:,None];r=torch.randn(2,7)
        a,p=old(h,mask,r);b,q=new(h,mask,r)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_unknown_targets_do_not_imply_deletion_or_supply_gradient(self):
        logits=torch.randn(3,len(IDS),requires_grad=True);loss=vowel_loss(logits,torch.full((3,),-100))
        loss.backward();self.assertEqual(float(loss),0);self.assertEqual(int(logits.grad.count_nonzero()),0)


if __name__=='__main__':unittest.main()

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_vowel_presence import PresenceReadout,training_target,actual_class,observed_vowel_support


class PresenceTest(unittest.TestCase):
    def test_absence_split_preserves_coarse_probabilities_initially(self):
        torch.manual_seed(7);old=SupportReadout(5,True);new=PresenceReadout(5,True);new.load_base(old.state_dict())
        h=torch.randn(2,9,5);mask=torch.ones(2,9,dtype=torch.bool);r=torch.randn(2,9)
        a,p=old(h,mask,r);b,q=new(h,mask,r);b=b.softmax(-1)
        coarse=torch.cat([b[:,:3],b[:,3:].sum(-1,keepdim=True)],-1)
        torch.testing.assert_close(coarse,a.softmax(-1),rtol=1e-6,atol=1e-7);torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_deletion_requires_TAG_label_and_intended_V_support(self):
        label={'actual':None,'label_kind':'deletion','alignment_ambiguous':False};t=np.arange(10)*.02
        y,phase,state=training_target(label,None,(.04,.15,'ㅣ'),0,3200,t,False)
        self.assertEqual(y,4);self.assertNotIn(1,phase);self.assertIn(2,phase)
        self.assertEqual(training_target(label,None,(.5,.6,'ㅣ'),0,3200,t,False)[0],-100)
        self.assertEqual(actual_class(dict(label,label_kind='ambiguous')),-100)

    def test_no_C_peak_is_not_a_measurement_failure_for_absence(self):
        logits=torch.full((1,8,4),-10.);logits[:,:,2]=10.
        self.assertTrue(bool(observed_vowel_support(logits,torch.ones(1,8,dtype=torch.bool))[0]))


if __name__=='__main__':unittest.main()

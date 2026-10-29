import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT/'tasks/pronunciation-assessment/scripts/eval'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import unittest
import numpy as np
import torch
from diagnose_h0_p4 import pool,apply_probe,score_ctc
from fit_h0_representation_probes import threshold,counts
from audit_h0_p4_audio_labels import acoustic_qc


class P4Tests(unittest.TestCase):
    def test_fixed_windows_do_not_follow_changed_predictions(self):
        h=np.arange(24,dtype=np.float32).reshape(6,4)/7
        bounds=np.array([[1,3],[-1,-1],[4,6]])
        x,ok=pool(h,bounds)
        np.testing.assert_equal(ok,[True,False,True])
        np.testing.assert_equal(x[0],h[1:3].mean(0).astype(np.float16).astype(np.float32))
        np.testing.assert_equal(x[1],np.zeros(4))
        with self.assertRaises(ValueError):pool(h,np.array([[4,7]]))

    def test_frozen_normalization_is_not_refit(self):
        state=dict(mean=np.array([10.,20.],np.float32),std=np.array([2.,4.],np.float32),
                   weight=torch.tensor([[1.,-1.],[-1.,1.]]),bias=torch.zeros(2))
        x=np.array([[12.,20.],[10.,24.]],np.float32)
        np.testing.assert_equal(apply_probe(x,state),[[1.,-1.],[-1.,1.]])
        np.testing.assert_equal(state['mean'],[10.,20.])

    def test_ctc_sequence_and_projection_distinct(self):
        vocab={'ㅅ':4,'ㅜ':5,'ㄹ':6,'ㅣ':7,'ㅗ':8}
        tokens=[0,4,4,0,8,0,6,0,7,0]
        logits=torch.full((1,len(tokens),9),-10.)
        for i,t in enumerate(tokens):logits[0,i,t]=10.
        seq,proj,scores,segments=score_ctc(logits,len(tokens),list('ㅅㅜㄹㅣ'),vocab)
        self.assertEqual(seq,list('ㅅㅗㄹㅣ'));self.assertEqual(proj,list('ㅅㅗㄹㅣ'))
        self.assertGreater(scores[1],0);self.assertTrue(all(segments))

    def test_calibration_ties_do_not_exceed_budget(self):
        score=np.array([5.]*3+[0.]*97,np.float32);normal=np.ones(100,bool)
        tau=threshold(score,normal,normal,normal)
        m=counts(np.arange(100),np.zeros(100,int),np.zeros(100,int),np.ones(100,int),score,normal,tau)
        self.assertEqual(m['normal_fp'],0)
        self.assertGreater(tau,5.)

    def test_missing_tag_interval_not_silence_or_normal(self):
        audio=np.sin(np.arange(1600)*2*np.pi*200/16000).astype(np.float32)*.1
        self.assertEqual(acoustic_qc(audio,None)['status'],'no_confirmed_TAG_window')
        r=acoustic_qc(audio,[0.,.2])
        self.assertEqual(r['samples'],1600);self.assertEqual(r['span_end_exceeds_audio_ms'],100.)
        self.assertEqual(r['clipped_fraction'],0.)


if __name__=='__main__':unittest.main()

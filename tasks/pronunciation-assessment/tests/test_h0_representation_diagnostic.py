"""Evidence-window contracts, including unavailable and oracle-only timing."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/scripts/eval'))
import unittest
import numpy as np
from diagnose_h0_representations import timing_windows, pool_indices, frame_centers
from fit_h0_representation_probes import threshold, counts


class WindowTests(unittest.TestCase):
    def test_real_timestamps_not_tag_forced_alignment(self):
        meta = dict(tag_phones=[[.1,.2,'ㄱ'],[.2,.4,'ㅗ']], lab_phones=[[.1,.2,'ㄱ'],[.2,.4,'ㅜ']])
        windows, reasons = timing_windows(list('ㄱㅜ'), list('ㄱㅗ'), meta)
        self.assertEqual(windows,[(.1,.2),(.2,.4)])
        self.assertEqual(reasons,['tag_phone','tag_phone'])

    def test_ambiguous_and_deletion_are_not_guessed(self):
        m = dict(tag_phones=[[.1,.2,'ㄱ']],lab_phones=[[.1,.2,'ㄱ'],[.2,.3,'ㄱ']])
        self.assertEqual(timing_windows(list('ㄱㄱ'),list('ㄱ'),m)[0],[None,None])
        m = dict(tag_phones=[[.1,.2,'-'],[.2,.3,'ㅏ']],lab_phones=[[.1,.2,'ㄱ'],[.2,.3,'ㅏ']])
        self.assertEqual(timing_windows(list('ㄱㅏ'),list('ㅏ'),m)[0],[(.1,.2),(.2,.3)])
        m['tag_phones'] = [[.2,.3,'ㅏ']]
        self.assertIsNone(timing_windows(list('ㄱㅏ'),list('ㅏ'),m)[0][0])

    def test_multi_token_interval_has_no_individual_boundary(self):
        m = dict(tag_phones=[[.1,.3,'ㄱㅏ']],lab_phones=[[.1,.2,'ㄱ'],[.2,.3,'ㅏ']])
        self.assertEqual(timing_windows(list('ㄱㅏ'),list('ㄱㅏ'),m)[0],[None,None])

    def test_auto_pooling_independent_of_tag(self):
        c = frame_centers('w2vbert', 20)
        a,b,t = pool_indices([5,6], [.10,.15], c)
        aa,bb,tt = pool_indices([5,6], [.2,.3], c)
        np.testing.assert_array_equal(a,aa)
        for x,y in zip(b,bb): np.testing.assert_array_equal(x,y)
        self.assertNotEqual(t.tolist(),tt.tolist())
        self.assertEqual(len(b),3)
        self.assertEqual(pool_indices([],None,c)[0].size,0)
        self.assertEqual(pool_indices([1], [10,11],c)[2].size,0)

    def test_frontend_clock(self):
        self.assertAlmostEqual(frame_centers('w2vbert',2)[0],.0175)
        self.assertAlmostEqual(frame_centers('xlsr300m',2)[0],.0125)
        self.assertAlmostEqual(np.diff(frame_centers('xlsr300m',2))[0],.02)

    def test_threshold_ties_and_unavailable_denominator(self):
        normal=np.ones(200,dtype=bool);mismatch=normal.copy();avail=normal.copy()
        scores=np.zeros(200,dtype=np.float32);scores[:3]=1
        tau=threshold(scores,normal,mismatch,avail)
        self.assertGreater(tau,1.)
        # Compare in float64: otherwise a scalar cast can re-include ties.
        self.assertEqual(int((scores.astype(float)>=tau).sum()),0)
        e=np.array([0,0,0]);t=np.array([0,1,1]);p=np.array([1,1,2])
        m=counts(np.arange(3),e,t,p,np.ones(3),np.array([False,True,True]),0.)
        self.assertEqual((m['normal'],m['errors'],m['available']),(1,2,2))
        self.assertEqual((m['normal_fp'],m['exact'],m['wrong_actual']),(0,1,1))


if __name__ == '__main__': unittest.main()

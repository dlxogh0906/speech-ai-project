"""Meaningful guards against oracle cropping and unknown-label normal inflation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from mdd.h0_evaluation_readiness import audio_windows, window_annotation, check_intervals, label_kind


class ReadinessTests(unittest.TestCase):
    def test_all_samples_partitioned_without_annotation_input(self):
        sr=1000;x=np.ones(37001,np.float32)*.1;x[11000:11500]=0
        for mode in ('fixed','energy'):
            windows=audio_windows(x,sr,mode)
            self.assertEqual(windows[0][0],0);self.assertEqual(windows[-1][1],len(x))
            self.assertEqual(sum(b-a for a,b in windows),len(x))
            self.assertTrue(all(windows[i][1]==windows[i+1][0] for i in range(len(windows)-1)))
        self.assertNotEqual(audio_windows(x,sr,'fixed'),audio_windows(x,sr,'energy'))

    def test_unknown_is_never_normal_or_deleted_phone(self):
        v=set('ㅅㅜㄹㅣ')
        for label in ('spn','<unk>','?','ㅜ?','.'):
            self.assertEqual(label_kind(label,v),'unknown')
        r=window_annotation([[0,.5,'ㅜ'],[.5,1,'spn']],0,1,v)
        self.assertFalse(r['complete']);self.assertEqual(r['tokens'],['ㅜ'])
        self.assertEqual(label_kind('-',v),'explicit_deletion')
        self.assertEqual(label_kind('-',v,True),'unknown')

    def test_crossing_phone_does_not_become_full_reference(self):
        r=window_annotation([[.9,1.2,'ㅜ']],0,1,set('ㅜ'))
        self.assertEqual(r['boundary_crossings'],1);self.assertEqual(r['tokens'],[])
        self.assertFalse(r['complete'])

    def test_invalid_times_and_no_missing_tail(self):
        c=check_intervals([[0,.5,'ㅜ'],[.4,2,'ㅗ'],[2,2,'-']],1,set('ㅜㅗ'))
        self.assertEqual(c['issues']['out_of_order'],1)
        self.assertEqual(c['issues']['past_audio_10ms'],2)
        self.assertEqual(c['issues']['nonpositive'],1)
        self.assertEqual(audio_windows(np.zeros(12001),1000),[(0,12000),(12000,12001)])


if __name__=='__main__':unittest.main()

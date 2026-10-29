import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdd.sibilant_vowel_reference import reference_for_site,original_window,annotated_intended_vowel_interval


class VowelReferenceTest(unittest.TestCase):
    def test_unlinked_c_can_use_uniquely_linked_v(self):
        sites=[{'pred':None,'j':None},{'pred':'ㅣ','j':1}]
        value,state=reference_for_site(['ㅏ','ㅣ'],[(3,4),(8,10)],['ㅅ','ㅣ'],sites,0,[{None},{1}])
        self.assertEqual(state,'observed_next_V');self.assertEqual(value['token'],'ㅣ')
        self.assertIsNone(reference_for_site(['ㅏ','ㅣ'],[(3,4),(8,10)],['ㅅ','ㅣ'],sites,0,[{None},{0,1}])[0])

    def test_observed_v_is_not_replaced_by_expected_v(self):
        sites=[{'pred':None,'j':None},{'pred':'ㅜ','j':0}]
        value,_=reference_for_site(['ㅜ'],[(8,9)],['ㅆ','ㅣ'],sites,0,[{None},{0}])
        self.assertEqual(value['token'],'ㅜ')

    def test_window_preserves_continuous_original_samples(self):
        idx=np.concatenate([np.arange(10000),np.arange(18000,40000)])
        left,right,shift=original_window(idx,40000,34,.02)
        self.assertEqual(shift,8000)
        clip=np.arange(40000)[left:right]
        self.assertIn(17900,clip);self.assertTrue((np.diff(clip)==1).all())

    def test_diagnostic_vowel_slot_must_actually_be_a_vowel(self):
        intervals=[(0.,.1,'ㄱ'),(.1,.2,'ㅣ')]
        self.assertIsNone(annotated_intended_vowel_interval(intervals,{0}))
        self.assertIsNone(annotated_intended_vowel_interval(intervals,{0,1}))
        self.assertEqual(annotated_intended_vowel_interval(intervals,{1}),intervals[1])


if __name__=='__main__':unittest.main()

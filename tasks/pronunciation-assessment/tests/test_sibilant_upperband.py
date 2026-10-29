"""Energy-preserving ablation, immutable ROI, and retained failed observations."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import hashlib
import unittest
import numpy as np
from mdd.sibilant_upperband import power_frames,flatten_upper,mean_pair,UPPER,missing_balanced_abx
from mdd.sibilant_trajectory_abx import class_balanced_abx


def feature(wave,left=13,right=1800):
    return dict(left=left,right=right,clip_sha256=hashlib.sha256(wave[left:right].tobytes()).hexdigest())


class UpperBandTest(unittest.TestCase):
    def test_ablation_preserves_low_bins_and_upper_energy(self):
        t = np.arange(2000)/16000.
        wave = np.sin(2*np.pi*1000*t)+.7*np.sin(2*np.pi*6200*t)
        power,starts = power_frames(wave,feature(wave))
        flat = flatten_upper(power)
        np.testing.assert_array_equal(power[:,~UPPER],flat[:,~UPPER])
        np.testing.assert_allclose(power[:,UPPER].sum(-1),flat[:,UPPER].sum(-1),rtol=1e-14)
        self.assertTrue((np.ptp(flat[:,UPPER],axis=-1) == 0).all())
        self.assertTrue((np.ptp(power[:,UPPER],axis=-1) > 1).all())
        self.assertGreaterEqual(starts[0],13);self.assertLessEqual(starts[-1]+400,1800)

    def test_gain_dc_and_outside_roi_cannot_supply_class_information(self):
        wave = np.random.RandomState(4).normal(size=2000)
        original = wave.copy();f = feature(wave)
        expected,_ = mean_pair(wave,f)
        changed = 3*wave+2
        actual,_ = mean_pair(changed,feature(changed))
        np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
        outside = wave.copy();outside[:13] += 100;outside[1800:] -= 100
        np.testing.assert_array_equal(expected,mean_pair(outside,f)[0])
        np.testing.assert_array_equal(wave,original)
        corrupt = wave.copy();corrupt[100] += 1
        with self.assertRaises(ValueError):mean_pair(corrupt,f)

    def test_failed_frames_and_queries_remain_at_chance(self):
        wave = np.zeros(2000)
        value,flow = mean_pair(wave,feature(wave))
        self.assertTrue(np.isnan(value).all())
        self.assertEqual(flow['failed_frames'],flow['frames'])
        self.assertEqual(flow['valid_frames'],0)
        labels = np.asarray([0,1])
        matrix = np.asarray([[0.,1.],[1.,0.]])
        self.assertEqual(missing_balanced_abx(matrix,labels,labels),class_balanced_abx(matrix,labels,labels))
        matrix[0] = np.nan
        self.assertEqual(missing_balanced_abx(matrix,labels,labels)[0],.625)
        matrix[:] = np.nan
        self.assertEqual(missing_balanced_abx(matrix,labels,labels)[0],.5)


if __name__ == '__main__':unittest.main()

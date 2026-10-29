import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd.sibilant_source_view import cached_anchor, source_bounds, source_relative


class SourceViewTest(unittest.TestCase):
    def test_continuous_input_preserves_relative_bits(self):
        n = 32000
        frames = (1+(n-400)//160)//2
        dt = n/(16000.*frames)
        frame = 45
        left, right = int((frame-15)*dt*16000), int((frame+16)*dt*16000)
        length = (1+(right-left-400)//160)//2
        relative = source_relative(left, right, length, 0, frame, dt)
        self.assertEqual(cached_anchor(left, right, length, relative, n), (frame, dt))
        a, b, shift = source_bounds(np.arange(n)+1000, frame, dt, n+2000)
        self.assertEqual((a,b,shift), (left+1000,right+1000,1000))
        np.testing.assert_array_equal(relative, source_relative(a,b,length,shift,frame,dt))

    def test_jump_is_not_stitched_into_observation(self):
        idx = np.concatenate([np.arange(10000), np.arange(18000,40000)])
        frame, dt = 34, .02
        left,right,shift = source_bounds(idx,frame,dt,40000)
        self.assertEqual(shift,8000)
        original = np.arange(40000)
        clip = original[left:right]
        self.assertTrue((np.diff(clip)==1).all())
        self.assertIn(17900,clip)
        self.assertNotIn(17900,idx)

    def test_original_prefix_can_be_recovered(self):
        idx = np.arange(4000,24000)
        left,right,shift = source_bounds(idx,2,.02,24000)
        self.assertLess(left,idx[0])
        self.assertGreater(right,left)
        self.assertEqual(shift,4000)


if __name__ == '__main__':
    unittest.main()

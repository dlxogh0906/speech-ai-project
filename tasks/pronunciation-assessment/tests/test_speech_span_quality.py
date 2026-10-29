import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.speech_span_quality import assess_span


class SpanQualityTests(unittest.TestCase):
    def test_frame_aligned_outer_silence_does_not_change_span_decision(self):
        # A deterministic energy fixture, not evidence that a tone is speech.
        x = (.1 * np.sin(2 * np.pi * 200 * np.arange(9600) / 16000)).astype(np.float32)
        short = assess_span(np.pad(x, (4800, 4800)), 16000, .2)
        padded = assess_span(np.pad(x, (48000, 48000)), 16000, .2)
        self.assertTrue(short["accepted"] and padded["accepted"])
        self.assertEqual(short["span_ratio"], padded["span_ratio"])
        self.assertLess(padded["recording_ratio"], .3)

    def test_minimum_active_duration_preserved(self):
        x = np.zeros(16000, np.float32)
        x[4800:5280] = .1
        self.assertFalse(assess_span(x, 16000, .2)["accepted"])

    def test_silence_stays_rejected(self):
        self.assertFalse(assess_span(np.zeros(16000), 16000, .2)["accepted"])
        self.assertFalse(assess_span(np.zeros(100), 16000, .2)["accepted"])

    def test_internal_sparse_activity_is_not_compressed(self):
        x = np.zeros(160000)
        x[16000:20800] = .1
        x[144000:148800] = .1
        result = assess_span(x, 16000, .2)
        self.assertGreater(result["active_seconds"], .2)
        self.assertFalse(result["accepted"])

    def test_nonfinite_is_not_audio(self):
        with self.assertRaises(ValueError):
            assess_span(np.array([float("nan")]), 16000, .2)


if __name__ == "__main__":
    unittest.main()

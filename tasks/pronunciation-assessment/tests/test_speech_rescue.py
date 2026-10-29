import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.speech_rescue import SpeechRescue


class RescueTests(unittest.TestCase):
    def gate(self, seconds, no_speech=False):
        gate = object.__new__(SpeechRescue)
        self.calls = 0

        def detector(wav):
            self.calls += 1
            return wav[100:-100], {"speech_sec": seconds, "no_speech": no_speech}

        gate.detector = detector
        return gate

    def signal(self):
        # Energy fixture only; the injected detector is not acoustic evidence.
        return np.pad(np.full(9600, .1, np.float32), (32000, 32000))

    def test_accepted_input_and_other_rejections_do_not_run_detector(self):
        g = self.gate(1.)
        self.assertTrue(g.assess(self.signal(), None, .2)["accepted"])
        self.assertFalse(g.assess(self.signal(), "too_short", .2)["accepted"])
        self.assertEqual(self.calls, 0)

    def test_energy_alone_cannot_rescue(self):
        g = self.gate(0., True)
        self.assertFalse(g.assess(self.signal(), "low_speech", .2)["accepted"])
        self.assertEqual(self.calls, 1)

    def test_minimum_duration_and_sparse_span_are_preserved(self):
        g = self.gate(.19)
        self.assertFalse(g.assess(self.signal(), "low_speech", .2)["accepted"])
        x = np.zeros(160000, np.float32)
        x[16000:20800] = .1
        x[144000:148800] = .1
        self.assertFalse(g.assess(x, "low_speech", .2)["accepted"])
        self.assertEqual(self.calls, 1)

    def test_supported_span_does_not_change_input(self):
        g = self.gate(.6)
        x = self.signal()
        before = x.tobytes()
        self.assertTrue(g.assess(x, "low_speech", .2)["rescued"])
        self.assertEqual(x.tobytes(), before)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            self.gate(1.).assess(np.array([np.nan]), None, .2)


if __name__ == "__main__":
    unittest.main()

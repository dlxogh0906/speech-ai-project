"""Signal and provenance checks for offline observation candidates."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import importlib.util
import hashlib
import tempfile
import unittest
import wave

import numpy as np
from scipy import signal
from mdd.sibilant_observations import finite_low_band, noise_observations, periodicity_observations, interval_observations, interval_relation


class ObservationSignals(unittest.TestCase):
    def test_finite_filter_support_and_original_noise_spectrum(self):
        sr = 16000
        impulse = np.zeros(sr)
        impulse[8000] = 1.
        low, info = finite_low_band(impulse, sr)
        half = round(info["half_support_s"] * sr)
        self.assertTrue(np.all(low[:8000-half] == 0))
        self.assertTrue(np.all(low[8001+half:] == 0))
        self.assertEqual(impulse[8000], 1.)
        t = np.arange(sr) / sr
        x = .1 * np.sin(2*np.pi*5000*t)
        original = x.copy()
        finite_low_band(x, sr)
        obs = noise_observations(x, sr)
        self.assertAlmostEqual(np.median([f["centroid_1000_to_top_hz"] for f in obs["frames"]]), 5000, delta=2)
        np.testing.assert_array_equal(x, original)

    def test_long_high_rate_windows_are_not_silently_truncated_by_fft(self):
        sr = 48000
        # Energy is only in the last half of a 50ms frame. A fixed 1024 FFT
        # would discard it and invent a missing spectrum.
        x = np.zeros(2400)
        x[1200:] = .1*np.sin(2*np.pi*6000*np.arange(1200)/sr)
        obs = noise_observations(x, sr, window_seconds=.05)
        self.assertGreaterEqual(obs["fft_samples"], 2400)
        self.assertAlmostEqual(obs["frames"][0]["centroid_1000_to_top_hz"], 6000, delta=25)
        self.assertEqual(obs["frames"][0]["support_s"], [0., .05])

    def test_invalid_pcm_and_silence_stay_explicit(self):
        for bad in (np.zeros((2, 20)), np.array([np.nan]), np.empty(0)):
            with self.assertRaises(ValueError):
                noise_observations(bad, 16000)
        rows = noise_observations(np.zeros(1600), 16000)["frames"]
        self.assertTrue(all(r["centroid_1000_to_top_hz"] is None and r["upper_lower_3000_db"] is None for r in rows))

    def test_octave_disagreement_is_retained_without_a_pronunciation_decision(self):
        obs = {"clip_duration_s": 1., "time_step_s": .005, "settings": []}
        for floor, f0 in [(70., 75.), (100., 150.)]:
            obs["settings"].append({"floor_hz": floor, "frames": [{"center_s": .5,
                "paired_support_s": [.48, .52], "raw": {"candidate_hz": f0}, "low_band": {"candidate_hz": f0}}]})
        out = interval_observations(obs, .4, .6, "synthetic_known_octave_disagreement")
        self.assertTrue(all(r["near_octave_review_flag"] for r in out["cross_floor_comparisons"]))
        self.assertIsNone(out["pronunciation_decision"])
        self.assertFalse(out["glottal_source_validated"])
        with self.assertRaises(ValueError):
            interval_observations(obs, .9, 1.1, "out_of_clip")

    def test_interval_overlap_does_not_certify_or_move_phone_boundaries(self):
        relation = interval_relation(.663, .703, .64, .68, "synthetic_TAG_vowel")
        self.assertAlmostEqual(relation["overlap_s"], .017)
        self.assertAlmostEqual(relation["after_reference_s"], .023)
        self.assertFalse(relation["within_reference_interval"])
        self.assertFalse(relation["acoustic_phone_containment_verified"])
        inside = interval_relation(.65, .67, .64, .68, "synthetic_TAG_vowel")
        self.assertTrue(inside["within_reference_interval"])
        self.assertFalse(inside["acoustic_phone_containment_verified"])
        before = interval_relation(.1, .2, .64, .68, "synthetic_TAG_vowel")
        self.assertEqual(before["overlap_s"], 0.)
        self.assertAlmostEqual(before["before_reference_s"], .1)


class CandidatePitchSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import parselmouth
        except ImportError:
            raise unittest.SkipTest("Parselmouth required for candidate pitch checks")
        if parselmouth.PRAAT_VERSION != "6.1.38":
            raise unittest.SkipTest("support audit is pinned to Praat 6.1.38")

    def test_known_periodicity_under_highband_noise_and_pcm_immutability(self):
        sr = 16000
        t = np.arange(sr)/sr
        rng = np.random.RandomState(101)
        noise = signal.sosfiltfilt(signal.butter(4, 3000, fs=sr, btype="highpass", output="sos"), rng.normal(size=sr))
        noise /= np.sqrt(np.mean(noise**2))
        x = .03*(np.sin(2*np.pi*200*t)*np.sqrt(2) + 10*noise)
        original = x.copy()
        obs = periodicity_observations(x, sr)
        summary = interval_observations(obs, .45, .53, "synthetic_200Hz_plus_noise")
        for setting in summary["settings"]:
            self.assertAlmostEqual(setting["branches"]["low_band"]["median_candidate_hz"], 200., delta=2)
        np.testing.assert_array_equal(x, original)
        short = interval_observations(obs, .5, .508, "synthetic_short_span")
        self.assertTrue(all(s["supported_frames"] == 0 for s in short["settings"]))

    def test_resonant_noise_can_return_pitch_without_certifying_glottal_source(self):
        sr = 16000
        x = signal.sosfiltfilt(signal.butter(4, [200, 400], fs=sr, btype="bandpass", output="sos"), np.random.RandomState(0).normal(size=sr))
        obs = periodicity_observations(x, sr)
        summary = interval_observations(obs, .45, .53, "synthetic_noise_no_glottal_source")
        self.assertTrue(any(s["branches"]["low_band"]["returned_frames"] > 0 for s in summary["settings"]))
        self.assertFalse(summary["glottal_source_validated"])
        self.assertIsNone(summary["pronunciation_decision"])

    def test_known_tone_keeps_unselected_subharmonic_candidate(self):
        sr = 16000
        t = np.arange(int(.65*sr))/sr
        # A known 240Hz middle plateau between 120Hz plateaus. Its CC also
        # offers a 120Hz subharmonic; retain it without treating it as gold.
        f0 = np.where((t >= .2) & (t < .45), 240., 120.)
        x = .1*np.sin(2*np.pi*np.cumsum(f0)/sr)
        obs = periodicity_observations(x, sr, floors=(100.,))
        frame = min(obs["settings"][0]["frames"], key=lambda r: abs(r["center_s"] - .32))
        raw = frame["raw"]
        self.assertAlmostEqual(raw["candidate_hz"], 240., delta=1.)
        self.assertAlmostEqual(raw["path_comparison"]["no_transition_cost_candidate_hz"], 240., delta=1.)
        self.assertTrue(raw["path_comparison"]["local_candidates_identical"])
        self.assertFalse(raw["path_comparison"]["selected_candidate_changed"])
        self.assertTrue(any(c["frequency_hz"] and abs(c["frequency_hz"] - 120.) < 1
                            and not c["selected_by_default_path"] for c in raw["local_candidates"]))
        self.assertFalse(obs["glottal_source_validated"])

    def test_export_retains_next_vowel_overlap_and_rejects_changed_TAG(self):
        spec = importlib.util.spec_from_file_location("periodicity_export_context_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/audit_sibilant_periodicity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            path, tag = Path(tmp)/"synthetic.wav", Path(tmp)/"synthetic.TextGrid"
            with wave.open(str(path), "wb") as w:
                w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes(np.zeros(16000, dtype="<i2").tobytes())
            text = 'File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 1\n        intervals: size = 4\n'
            for k, (a, b, label) in enumerate([(0, .64, ""), (.64, .68, "ㅜ"), (.68, .9, "ㅣ"), (.9, 1, "")], 1):
                text += '        intervals [%d]:\n            xmin = %s\n            xmax = %s\n            text = "%s"\n' % (k, a, b, label)
            tag.write_text(text)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            item = {"id": "synthetic", "source_wav": str(path), "source_sha256": source_hash,
                    "tag_file": str(tag), "tag_sha256": hashlib.sha256(tag.read_bytes()).hexdigest(),
                    "analysis_interval": [.5, 1.], "tag_vowel": [.64, .68], "tag_vowel_label": "ㅜ"}
            review = {"source_sha256": source_hash, "actor_type": "synthetic_fixture", "review_status": "not_human_gold",
                      "voicing_first": .663, "first_regular_run_start": None, "first_regular_run_end": None}
            record = mod.inspect_item(item, review)
            view = record["intervals"]["reviewed_first_voicing_first40ms"]
            np.testing.assert_allclose(view["original_interval_s"], [.663, .703])
            self.assertAlmostEqual(view["TAG_vowel_relation"]["after_reference_s"], .023)
            self.assertEqual([r["label"] for r in view["overlapping_TAG_phone_intervals"]], ["ㅜ", "ㅣ"])
            self.assertIsNone(record["pronunciation_decision"])
            tag.write_text(text.replace('text = "ㅣ"', 'text = "ㅏ"'))
            with self.assertRaisesRegex(ValueError, "TAG fingerprint differs"):
                mod.inspect_item(item, review)


class PacketProvenance(unittest.TestCase):
    def test_expected_vowel_does_not_overwrite_transcribed_or_missing_context(self):
        spec = importlib.util.spec_from_file_location("periodicity_context_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/audit_sibilant_periodicity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mismatch = mod.vowel_context_observation({"following_vowel": "ㅡ", "actual_vowel": "ㅢ", "tag_vowel_label": "ㅢ"})
        self.assertEqual(mismatch["expected_following_vowel"], "ㅡ")
        self.assertEqual(mismatch["existing_TAG_following_vowel"], "ㅢ")
        self.assertEqual(mismatch["status"], "context_label_disagreement")
        self.assertFalse(mismatch["pronunciation_error_inferred_from_label_disagreement"])
        missing = mod.vowel_context_observation({})
        self.assertIsNone(missing["labels_agree"])
        self.assertEqual(missing["status"], "context_labels_incomplete")
        agreed = mod.vowel_context_observation({"following_vowel": "ㅣ", "actual_vowel": "ㅣ"})
        self.assertTrue(agreed["labels_agree"])
        self.assertFalse(agreed["acoustic_vowel_identity_verified"])

    def test_changed_source_is_rejected_before_measurement(self):
        spec = importlib.util.spec_from_file_location("periodicity_export_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/audit_sibilant_periodicity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic.wav"
            with wave.open(str(path), "wb") as w:
                w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes(np.zeros(1600, dtype="<i2").tobytes())
            with self.assertRaisesRegex(ValueError, "hash differs"):
                mod.inspect_item({"id": "synthetic", "source_wav": str(path), "source_sha256": "wrong"})


if __name__ == "__main__":
    unittest.main()

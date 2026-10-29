"""Decision safety and runtime-input invariants for the transition pilot."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import numpy as np
from mdd.sibilant_transition import (acoustic_tracks, observation_window,
                                    annotate_probabilities, fit_acoustic_normalization,
                                    normalize_acoustics)


class TransitionTests(unittest.TestCase):
    def test_expected_sibilant_does_not_change_classifier_inputs(self):
        vocab = {"_": 0, "ㅅ": 1, "ㅆ": 2, "ㅏ": 3}
        lp = np.full((40, 4), -8., np.float32)
        lp[:, 0] = -.1
        lp[10, 1], lp[13, 2], lp[25, 3] = -.01, -.02, -.01
        lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
        rng = np.random.RandomState(1)
        wav = rng.normal(0, .02, 12800).astype(np.float32)
        original = wav.copy()
        left = observation_window(wav, lp, ["ㅅ", "ㅏ"], 0, vocab.__getitem__, 0)
        right = observation_window(wav, lp, ["ㅆ", "ㅏ"], 0, vocab.__getitem__, 0)
        for name in ("posterior", "acoustic", "mask", "context"):
            np.testing.assert_allclose(left[name], right[name], equal_nan=True)
        np.testing.assert_array_equal(wav, original)

    def test_silence_is_not_a_measured_spectrum_or_periodicity(self):
        _, values = acoustic_tracks(np.zeros(1600, np.float32))
        self.assertTrue(np.isnan(values[:, 1:]).all())
        norm = fit_acoustic_normalization([values])
        transformed = normalize_acoustics(values, norm)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue((transformed[:, values.shape[1] + 1:] == 0).all())

    def test_normalizer_does_not_refit_on_prediction(self):
        train = np.array([[1., np.nan], [3., np.nan]])
        norm = fit_acoustic_normalization([train])
        self.assertEqual(norm["mean"], [2., 0.])
        before = repr(norm)
        values = normalize_acoustics(np.array([[100., np.nan]]), norm)
        self.assertEqual(repr(norm), before)
        np.testing.assert_array_equal(values, [[8., 0., 1., 0.]])

    def test_adapter_keeps_third_phones_and_deletions(self):
        canon = list("ㅅㅏㅆㅏㅅㅏ")
        pred = list(canon)
        pred[0], pred[2], pred[4] = "ㅈ", None, "ㅅ"
        sites = [dict(i=i, canon=c, pred=p) for i, (c, p) in enumerate(zip(canon, pred))]
        actual = annotate_probabilities(canon, sites, {0: .99, 2: .01, 4: .99}, {"ㅅ": .9, "ㅆ": .9})
        self.assertNotIn("sz", actual[0])
        self.assertNotIn("sz", actual[2])
        self.assertEqual(actual[4]["sz"]["alt"], "ㅆ")
        self.assertTrue(all("sz" not in s for s in sites))

    def test_none_threshold_disables_direction(self):
        sites = [dict(i=0, canon="ㅆ", pred="ㅆ"), dict(i=1, canon="ㅏ", pred="ㅏ")]
        actual = annotate_probabilities(list("ㅆㅏ"), sites, {0: 0.}, {"ㅆ": None})
        self.assertEqual(actual, sites)

    def test_nonfinite_probability_rejected(self):
        sites = [dict(i=0, canon="ㅆ", pred="ㅆ"), dict(i=1, canon="ㅏ", pred="ㅏ")]
        with self.assertRaises(ValueError):
            annotate_probabilities(list("ㅆㅏ"), sites, {0: float("nan")}, {"ㅆ": .9})

    def test_calibration_ties_are_excluded_at_zero_additional_budget(self):
        sys.path[:0] = [str(ROOT / "tasks/pronunciation-assessment/scripts/train"),
                       str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
        from train_sibilant_transition import calibrate
        rows = [dict(expected="ㅅ", actual="ㅅ", state="available", feature_index=i,
                     baseline_phone={}) for i in range(10)]
        features = [dict(baseline_pred="ㅅ") for _ in rows]
        thresholds, audit = calibrate(rows, features, np.full(10, .9))
        self.assertGreater(thresholds["ㅅ"], .9)
        self.assertEqual(audit["ㅅ"]["additional_fp"], 0)
        self.assertIsNone(thresholds["ㅆ"])

    def test_existing_over_budget_alerts_disable_additions(self):
        sys.path[:0] = [str(ROOT / "tasks/pronunciation-assessment/scripts/train"),
                       str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
        from train_sibilant_transition import calibrate
        rows = [dict(expected="ㅆ", actual="ㅆ", state="available", feature_index=0,
                     baseline_phone={"errorType": "wrong"})]
        thresholds, _ = calibrate(rows, [dict(baseline_pred="ㅆ")], np.array([.01]))
        self.assertIsNone(thresholds["ㅆ"])

    def test_rejections_do_not_inflate_calibration_budget(self):
        sys.path[:0] = [str(ROOT / "tasks/pronunciation-assessment/scripts/train"),
                       str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
        from train_sibilant_transition import calibrate
        rows = [dict(expected="ㅅ", actual="ㅅ", state="input_rejected", feature_index=None,
                     baseline_phone=None) for _ in range(1000)]
        rows.append(dict(expected="ㅅ", actual="ㅅ", state="available", feature_index=0, baseline_phone={}))
        thresholds, audit = calibrate(rows, [dict(baseline_pred="ㅅ")], np.array([.8]))
        self.assertGreater(thresholds["ㅅ"], .8)
        self.assertEqual(audit["ㅅ"]["normal_matched_accepted"], 1)

    def test_margin_candidate_has_no_expected_phone_bias(self):
        from mdd.sibilant_margin import margin_observations
        vocab = {"_": 0, "ㅅ": 1, "ㅆ": 2, "ㅏ": 3}
        lp = np.full((40, 4), -8., np.float32)
        lp[:, 0] = -.1
        lp[10, 1], lp[13, 2], lp[25, 3] = -.01, -.02, -.01
        lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
        a = margin_observations(lp, list("ㅅㅏ"), vocab.__getitem__, 0)
        b = margin_observations(lp, list("ㅆㅏ"), vocab.__getitem__, 0)
        self.assertEqual(a, b)

    def test_margin_uniform_evidence_does_not_force_an_error(self):
        from mdd.sibilant_margin import annotate_margin
        vocab = {"_": 0, "ㅅ": 1, "ㅆ": 2, "ㅏ": 3}
        sites = [dict(i=0, canon="ㅆ", pred="ㅆ"), dict(i=1, canon="ㅏ", pred="ㅏ")]
        after, audit = annotate_margin(np.full((40, 4), -np.log(4)), list("ㅆㅏ"), sites,
                                       vocab.__getitem__, 0, {"ㅅ": None, "ㅆ": .82})
        self.assertEqual(after, sites)
        self.assertEqual(audit[0]["score_ss"], .5)


if __name__ == "__main__":
    unittest.main()

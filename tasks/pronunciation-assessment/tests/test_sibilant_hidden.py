import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import numpy as np
from mdd.sibilant_hidden import hidden_window, fit_projection, project_sequences, zero_extra_thresholds


class HiddenObservationTests(unittest.TestCase):
    def test_boundary_padding_cannot_wrap_to_end_of_utterance(self):
        h = np.arange(24, dtype=np.float32).reshape(8, 3)
        window, mask = hidden_window(h, 0, radius=2)
        np.testing.assert_array_equal(mask, [False, False, True, True, True])
        np.testing.assert_array_equal(window[:2], np.zeros((2, 3)))
        np.testing.assert_array_equal(window[2:], h[:3])

    def test_invalid_anchor_or_nonfinite_capture_is_rejected(self):
        for h, anchor in [(np.zeros((8, 3)), -1), (np.full((8, 3), np.nan), 2)]:
            with self.assertRaises(ValueError):
                hidden_window(h, anchor)

    def test_projection_fit_excludes_padding(self):
        x = np.array([[[1., 2], [3, 5], [900, 800]], [[2, 4], [8, 9], [700, 600]]])
        mask = np.array([[True, True, False], [True, True, False]])
        projection = fit_projection(x, mask, components=2)
        np.testing.assert_allclose(projection["mean"], x[mask].mean(0))
        self.assertEqual(projection["training_frames"], 4)

    def test_pooled_control_loses_order_but_preserves_valid_values(self):
        x = np.array([[[1., 3], [4, 2], [99, 99]]])
        mask = np.array([[True, True, False]])
        p = {"mean": np.zeros(2), "components": np.eye(2), "scale": np.ones(2)}
        a = project_sequences(x, mask, p, pooled=True)
        b = project_sequences(x[:, [1, 0, 2]], mask, p, pooled=True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[0, 0], [2.5, 2.5])
        np.testing.assert_array_equal(a[0, 2], [0, 0])

    def test_zero_extra_calibration_in_both_directions_ignores_rejections(self):
        labels, features, scores = [], [], []
        for expected, values in [("ㅅ", [.2, .7, .9]), ("ㅆ", [.8, .3, .1])]:
            for j, p in enumerate(values):
                ix = len(scores)
                labels.append(dict(expected=expected, actual=expected, state="available", uid=str(ix),
                                   q_runtime=0, feature_index=ix,
                                   baseline_phone={"errorType": "existing"} if j == 2 else {}))
                features.append(dict(uid=str(ix), q_runtime=0, baseline_pred=expected))
                scores.append(p)
        labels += [dict(expected="ㅅ", actual="ㅅ", state="input_rejected", feature_index=None)] * 100
        thresholds, audit = zero_extra_thresholds(labels, features, scores, minimum_normals=3)
        for expected in ("ㅅ", "ㅆ"):
            self.assertGreater(thresholds[expected], .7)
            self.assertLess(thresholds[expected], .9)
            self.assertEqual(audit[expected]["normal_matched_accepted"], 3)
            self.assertEqual(audit[expected]["additional_calibration_fp"], 0)

    def test_saturated_unflagged_normal_disables_additions(self):
        row = dict(expected="ㅅ", actual="ㅅ", state="available", uid="x", q_runtime=0,
                   feature_index=0, baseline_phone={})
        feature = dict(uid="x", q_runtime=0, baseline_pred="ㅅ")
        thresholds, _ = zero_extra_thresholds([row], [feature], [1.], minimum_normals=1)
        self.assertIsNone(thresholds["ㅅ"])


if __name__ == "__main__":
    unittest.main()

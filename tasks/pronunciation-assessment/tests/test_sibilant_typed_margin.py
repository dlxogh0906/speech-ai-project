import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.sibilant_typed_margin import observations, annotate, calibrate


class TypedMarginTests(unittest.TestCase):
    def test_actual_phone_preserved_and_mismatch_refused(self):
        sites = [{"i": 0, "canon": "ㅅ", "pred": "ㅊ"}, {"i": 1, "canon": "ㅏ", "pred": "ㅏ"}]
        value = {0: {"state": "available", "alt": "ㅊ", "score": .95}}
        result = annotate(list("ㅅㅏ"), sites, value, {"ㅅ": .9, "ㅆ": None})
        self.assertEqual(result[0]["pred"], "ㅊ")
        self.assertEqual(result[0]["sz"]["alt"], "ㅊ")
        self.assertNotIn("sz", sites[0])
        value[0]["alt"] = "ㅈ"
        with self.assertRaises(ValueError):
            annotate(list("ㅅㅏ"), sites, value, {"ㅅ": .9, "ㅆ": None})

    def test_wrong_actual_is_negative_even_at_a_real_error(self):
        def row(kind, actual, score):
            return {"expected": "ㅅ", "actual": actual, "label_kind": kind,
                    "view": {"phone": {"alpha_jamo": "ㅅ"}},
                    "observation": {"state": "available", "alt": "ㅊ", "score": score}}
        rows = [row("correct", "ㅅ", .6), row("third_phone", "ㅈ", .98), row("third_phone", "ㅊ", .95)]
        thresholds, audit = calibrate(rows, minimum_normals=1)
        self.assertGreater(thresholds["ㅅ"], .98)
        self.assertEqual(audit["ㅅ"]["cal_potential_exact_gains"], 0)
        self.assertIsNone(thresholds["ㅆ"])

    def test_no_negative_evidence_disables_direction(self):
        thresholds, _ = calibrate([], minimum_normals=1)
        self.assertEqual(thresholds, {"ㅅ": None, "ㅆ": None})

    def test_symmetric_paths_support_third_consonant_not_deletion(self):
        vocab = list("_ㅅㅆㅊㅏ")
        lp = np.full((5, len(vocab)), -12.)
        for i, token in enumerate("_ㅊ_ㅏ_"):
            lp[i, vocab.index(token)] = 0.
        sites = [{"pred": "ㅊ"}, {"pred": "ㅏ"}]
        values = observations(lp, list("ㅅㅏ"), sites, vocab.index, 0)
        self.assertEqual(values[0]["alt"], "ㅊ")
        self.assertGreater(values[0]["score"], .99)
        sites[0]["pred"] = None
        self.assertEqual(observations(lp, list("ㅅㅏ"), sites, vocab.index, 0)[0]["state"], "not_third_consonant")


if __name__ == "__main__":
    unittest.main()

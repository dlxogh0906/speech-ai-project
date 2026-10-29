"""Meaningful scope and formatter-contract tests for the isolated adapter."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import copy
import unittest
from mdd.sibilant_directional import annotate_directional


def sites(canon):
    return [{"i": i, "canon": t, "pred": t, "j": i, "verdict": "일치", "prob": 0.9,
             "pred_ph": t} for i, t in enumerate(canon)]


class DirectionalTests(unittest.TestCase):
    def test_scope_and_no_mutation(self):
        canon = list("ㅆㅏㅅㅏㄹㅆㅜㅆ")
        b, c = sites(canon), sites(canon)
        for i in (0, 5, 7):
            c[i]["pred"] = "ㅅ"
        c[2]["pred"] = "ㅆ"
        before = copy.deepcopy(b)
        narrow, audit = annotate_directional(canon, b, c, set("ㅏㅜ"))
        self.assertEqual(audit["annotated_indices"], [0])
        self.assertEqual(b, before)
        self.assertEqual(narrow[1:], b[1:])
        broad, audit = annotate_directional(canon, b, c, set("ㅏㅜ"), allow_post_coda=True)
        self.assertEqual(audit["annotated_indices"], [0, 5])
        self.assertEqual(broad[7], b[7])  # coda cannot become a target

    def test_bad_evidence_and_baseline_errors_preserved(self):
        canon = list("ㅆㅏ")
        for pred in (None, "ㅈ", "ㄹ"):
            b, c = sites(canon), sites(canon)
            b[0]["pred"] = pred
            c[0]["pred"] = "ㅅ"
            result, audit = annotate_directional(canon, b, c, {"ㅏ"})
            self.assertEqual(result, b)
            self.assertEqual(audit["annotated_indices"], [])
        for prob in (None, float("nan"), float("inf"), -0.1, 1.1, 0.59):
            b, c = sites(canon), sites(canon)
            c[0].update(pred="ㅅ", prob=prob)
            result, audit = annotate_directional(canon, b, c, {"ㅏ"})
            self.assertEqual(result, b)
            self.assertEqual(audit["annotated_indices"], [])

    def test_canonical_identity_required(self):
        b, c = sites(list("ㅆㅏ")), sites(list("ㅅㅏ"))
        with self.assertRaises(ValueError):
            annotate_directional(list("ㅆㅏ"), b, c, {"ㅏ"})

    def test_real_formatter_changes_only_target_phone_and_owner(self):
        from mdd import speechsuper as ss
        from mdd.session import phone_context
        from mdd.jamo_to_kog2p import jamo_to_phones, VOWEL
        # No heavyweight decoder/G2P import is needed for the formatter check.
        old = (ss.SZ_RULE, ss.SS_RULE)
        try:
            ss.SZ_RULE = ss.SS_RULE = False
            canon = list("ㅆㅏㅅㅏ")
            b, c = sites(canon), sites(canon)
            c[0].update(pred="ㅅ", prob=0.8)
            f = ss.SpeechSuperFormatter(jamo_to_phones=jamo_to_phones,
                phone_context=phone_context, ctype=lambda *args: "음운대치")
            f.cal = {"_global": {"a": -1.0, "b": -3.0}}  # high score hides ordinary substitutions
            margins = [2.0] * len(canon)
            before = f.build("싸사", canon, b, margins, vowels=set(VOWEL))
            updated, audit = annotate_directional(canon, b, c, set(VOWEL))
            ss.SZ_RULE = True
            after = f.build("싸사", canon, updated, margins, vowels=set(VOWEL))
            bp = [p for w in before["result"]["words"] for p in w["phonemes"]]
            ap = [p for w in after["result"]["words"] for p in w["phonemes"]]
            self.assertNotIn("errorType", bp[0])
            self.assertEqual(ap[0]["actual_jamo"], "ㅅ")
            self.assertTrue(ap[0]["errorType"])
            self.assertEqual(ap[0]["pronunciation"], bp[0]["pronunciation"])
            self.assertEqual(ap[1:], bp[1:])
            self.assertEqual(before["result"]["words"][1], after["result"]["words"][1])
            for key in set(before["result"]) - {"words"}:
                self.assertEqual(before["result"][key], after["result"][key])
        finally:
            ss.SZ_RULE, ss.SS_RULE = old

    def test_low_owner_score_can_change_formatter_aggregate(self):
        from mdd import speechsuper as ss
        from mdd.session import phone_context
        from mdd.jamo_to_kog2p import jamo_to_phones, VOWEL
        old = (ss.SZ_RULE, ss.SS_RULE)
        try:
            ss.SZ_RULE = ss.SS_RULE = False
            canon = list("ㅆㅏㅅㅏ")
            b, c = sites(canon), sites(canon)
            c[0].update(pred="ㅅ", prob=0.8)
            f = ss.SpeechSuperFormatter(jamo_to_phones=jamo_to_phones,
                phone_context=phone_context, ctype=lambda *args: "음운대치")
            f.cal = {"_global": {"a": 0.0, "b": 2.1972245773362196}}
            before = f.build("싸사", canon, b, [0.0] * 4, vowels=set(VOWEL))
            updated, _ = annotate_directional(canon, b, c, set(VOWEL))
            ss.SZ_RULE = True
            after = f.build("싸사", canon, updated, [0.0] * 4, vowels=set(VOWEL))
            bw, aw = before["result"]["words"], after["result"]["words"]
            self.assertEqual(bw[1], aw[1])
            self.assertEqual([p["pronunciation"] for w in bw for p in w["phonemes"]],
                             [p["pronunciation"] for w in aw for p in w["phonemes"]])
            self.assertEqual(bw[0]["score"], 10)
            self.assertIsNone(aw[0]["score"])
            self.assertEqual(aw[0]["readType"], ss.READ_BAD)
            self.assertNotEqual(before["result"]["overall"], after["result"]["overall"])
        finally:
            ss.SZ_RULE, ss.SS_RULE = old


if __name__ == "__main__":
    unittest.main()

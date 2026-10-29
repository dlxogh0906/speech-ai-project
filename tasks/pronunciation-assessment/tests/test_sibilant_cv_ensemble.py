import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.sibilant_cv_ensemble import annotate_heads


class EnsembleTest(unittest.TestCase):
    def test_complementary_heads_preserve_prior_alert_and_input(self):
        canon = list("ㅆㅓㅅㅣㅆㅜ")
        sites = [{"pred": p} for p in list("ㅅㅓㅆㅣㅅㅜ")]
        sites[4]["sz"] = {"alt": "ㅅ", "p": .99, "th": .8}
        phones = [{}, {}, {}, {}, {"errorType": "경음화"}, {}]
        threshold = {"ㅆ/pair": .8, "ㅅ/pair": .8}
        first = {"available": True, "probabilities": [.9, .05, .03, .02]}
        second = {"available": True, "probabilities": [.05, .9, .03, .02]}
        result, _ = annotate_heads(canon, sites, phones, [
            {"name": "first", "observations": {0: first, 4: first}, "thresholds": threshold},
            {"name": "second", "observations": {2: second}, "thresholds": threshold}])
        self.assertEqual(result[0]["sz"]["alt"], "ㅅ")
        self.assertEqual(result[2]["sz"]["alt"], "ㅆ")
        self.assertEqual(result[4], sites[4])
        self.assertNotIn("sz", sites[0])
        self.assertNotIn("sz", sites[2])

    def test_later_score_cannot_replace_first_confirmation(self):
        heads = [{"name": name, "observations": {0: {"available": True, "probabilities": [p, 1 - p, 0., 0.]}},
                  "thresholds": {"ㅆ/pair": .8}} for name, p in (("first", .9), ("second", .99))]
        result, traces = annotate_heads(list("ㅆㅓ"), [{"pred": "ㅅ"}, {"pred": "ㅓ"}], [{}, {}], heads)
        self.assertEqual(result[0]["sz"]["p"], .9)
        self.assertEqual(traces["second"], {})


if __name__ == "__main__":
    unittest.main()

"""Behavioral protections for offline C/V confirmation of sibilant errors."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.sibilant_cv_policy import annotate


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = {"ㅅ/pair": .9, "ㅆ/pair": .9}
        self.canonical = ["ㅆ", "ㅓ"]
        self.phones = [{"alpha_jamo": "ㅆ"}, {"alpha_jamo": "ㅓ"}]
        self.observations = {0: {"available": True, "probabilities": [.95, .02, .02, .01]}}

    def test_model_cannot_invent_opposite_against_decoder(self):
        sites = [{"pred": "ㅆ"}, {"pred": "ㅓ"}]
        guarded, state = annotate(self.canonical, sites, self.phones, self.observations, self.thresholds)
        unguarded, _ = annotate(self.canonical, sites, self.phones, self.observations, self.thresholds, False)
        self.assertEqual(guarded, sites)
        self.assertEqual(state[0], "decoder_disagrees")
        self.assertEqual(unguarded[0]["sz"]["alt"], "ㅅ")

    def test_unreliable_support_cannot_confirm_recognized_error(self):
        sites = [{"pred": "ㅅ"}, {"pred": "ㅓ"}]
        self.observations[0]["available"] = False
        unchanged, state = annotate(self.canonical, sites, self.phones, self.observations, self.thresholds)
        self.assertEqual(unchanged, sites)
        self.assertEqual(state[0], "support_unavailable")
        self.observations[0]["available"] = True
        confirmed, state = annotate(self.canonical, sites, self.phones, self.observations, self.thresholds)
        self.assertEqual(confirmed[0]["sz"]["alt"], "ㅅ")
        self.assertEqual(confirmed[1], sites[1])
        self.assertNotIn("sz", sites[0])

    def test_existing_alert_is_preserved_even_when_model_disagrees(self):
        sites = [{"pred": "ㅅ", "sz": {"alt": "ㅅ", "p": .97, "th": .8}}, {"pred": "ㅓ"}]
        self.phones[0]["errorType"] = "경음화"
        confirmed, state = annotate(self.canonical, sites, self.phones, self.observations, self.thresholds)
        self.assertEqual(confirmed, sites)
        self.assertEqual(state[0], "existing_alert_preserved")


if __name__ == "__main__":
    unittest.main()

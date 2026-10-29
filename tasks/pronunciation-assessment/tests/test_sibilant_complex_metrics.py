import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
from eval_sibilant_complex import metrics, METHODS


class ComplexMetricsTests(unittest.TestCase):
    def row(self, kind, actual, phone):
        value = {"label_kind": kind, "expected": "ㅆ", "actual": actual}
        value.update({name: {"state": "input_rejected" if phone is None else "available", "phone": phone} for name in METHODS})
        return value

    def test_rejected_errors_remain_in_denominator(self):
        result = metrics([self.row("pair_error", "ㅅ", None)])["baseline"]
        self.assertEqual(result["known_error_all"], 1)
        self.assertEqual(result["exact_recall_all"], 0.)

    def test_wrong_phone_alert_is_not_exact_detection(self):
        phone = {"alpha_jamo": "ㅆ", "actual_jamo": "ㅅ", "errorType": "경음화"}
        result = metrics([self.row("third_phone", "ㅈ", phone)])["baseline"]
        self.assertEqual(result["any_alert_on_known_error"], 1)
        self.assertEqual(result["exact_error_alerts"], 0)
        self.assertEqual(result["wrong_actual_alert_on_known_error"], 1)

    def test_deletion_needs_correct_expected_phone_and_alert(self):
        phone = {"alpha_jamo": "ㅆ", "actual_jamo": None, "errorType": "음운탈락"}
        self.assertEqual(metrics([self.row("deletion", None, phone)])["baseline"]["exact_error_alerts"], 1)
        phone["alpha_jamo"] = "ㅅ"
        self.assertEqual(metrics([self.row("deletion", None, phone)])["baseline"]["exact_error_alerts"], 0)

    def test_human_ambiguity_is_visible_and_not_called_deletion(self):
        result = metrics([self.row("ambiguous", None, None)])
        self.assertEqual(result["all_sites"], 1)
        self.assertEqual(result["baseline"]["human_ambiguous_sites"], 1)
        self.assertEqual(result["baseline"]["known_error_all"], 0)


if __name__ == "__main__":
    unittest.main()

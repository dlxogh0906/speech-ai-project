import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tasks/pronunciation-assessment/scripts/eval"))
from eval_speech_rescue import metrics


def observation(state="input_rejected", expected=None, actual=None, error=None):
    return {"state": state, "phone": None if expected is None else
            {"alpha_jamo": expected, "actual_jamo": actual, "errorType": error}}


class RescueMetricTests(unittest.TestCase):
    def test_rejected_errors_stay_in_recall_denominator(self):
        missing = observation()
        detected = observation("available", "ㅆ", "ㅅ", "평음화")
        rows = [{"expected": "ㅆ", "actual": "ㅅ", "baseline": missing, "candidate": detected},
                {"expected": "ㅆ", "actual": "ㅅ", "baseline": missing, "candidate": missing}]
        result = metrics(rows)["ㅆ→ㅅ"]
        self.assertEqual(result["errors_all"], 2)
        self.assertEqual(result["candidate"]["recall_all"], .5)
        self.assertEqual(result["additional_tp"], 1)

    def test_canonical_disagreement_alert_is_not_correct_error_direction(self):
        wrong = observation("runtime_canonical_disagrees", "ㅅ", "ㅆ", "경음화")
        rows = [{"expected": "ㅆ", "actual": "ㅅ", "baseline": observation(), "candidate": wrong}]
        self.assertEqual(metrics(rows)["ㅆ→ㅅ"]["candidate"]["tp"], 0)

    def test_new_normal_alert_count_and_coverage_are_both_reported(self):
        wrong = observation("runtime_canonical_disagrees", "ㅅ", "ㅆ", "경음화")
        rows = [{"expected": "ㅆ", "actual": "ㅆ", "baseline": observation(), "candidate": wrong},
                {"expected": "ㅆ", "actual": "ㅆ", "baseline": observation(), "candidate": observation()}]
        result = metrics(rows)["ㅆ→ㅅ"]
        self.assertEqual(result["normal_all"], 2)
        self.assertEqual(result["baseline"]["normal_position_known_accepted"], 0)
        self.assertEqual(result["candidate"]["normal_position_known_accepted"], 1)
        self.assertEqual(result["additional_normal_alerts"], 1)
        self.assertEqual(result["candidate"]["tag_normal_alert_rate_known"], 1.)


if __name__ == "__main__":
    unittest.main()

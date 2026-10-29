import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tasks/pronunciation-assessment/scripts/eval"))
from audit_sibilant_service_coverage import audit


class ServiceCoverageTests(unittest.TestCase):
    def test_wrong_reference_cannot_hide_visible_alert_on_tag_normal(self):
        rows = []
        for state, q, expected in [("available", 0, "ㅆ"), ("runtime_canonical_disagrees", 0, "ㅅ"),
                                   ("input_rejected", None, "ㅆ"), ("runtime_mapping_ambiguous", None, "ㅆ")]:
            phone = {"alpha_jamo": expected, "actual_jamo": "ㅆ", "errorType": "경음화"} if q is not None else None
            rows.append(dict(expected="ㅆ", actual="ㅆ", state=state, q_runtime=q,
                             baseline_phone=phone, candidate_phone=phone))
        result = audit(rows)["ㅆ→ㅅ"]
        self.assertEqual(result["normal_all"], 4)
        self.assertEqual(result["normal_position_known_accepted"], 2)
        self.assertEqual(result["baseline"]["tag_normal_visible_alerts"], 2)
        self.assertEqual(result["baseline"]["alerts_on_canonical_disagreements"], 1)

    def test_reverse_direction_alert_is_not_a_correct_detection(self):
        phone = dict(alpha_jamo="ㅅ", actual_jamo="ㅆ", errorType="경음화")
        row = dict(expected="ㅆ", actual="ㅅ", state="runtime_canonical_disagrees", q_runtime=0,
                   baseline_phone=phone, candidate_phone=phone)
        result = audit([row])["ㅆ→ㅅ"]
        self.assertEqual(result["error_all"], 1)
        self.assertEqual(result["baseline"]["correct_direction_error_detections"], 0)


if __name__ == "__main__":
    unittest.main()

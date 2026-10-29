# -*- coding: utf-8 -*-
"""인식 실패의 null 점수와 실제 0점을 구분하는 API 응답 회귀 검사."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks/pronunciation-assessment/src"))

from mdd.speechsuper import SpeechSuperFormatter


class NullScoreTests(unittest.TestCase):
    def setUp(self):
        self.formatter = SpeechSuperFormatter()
        self.formatter.cal = {"_global": {"a": -1.0, "b": 0.0}}

    def build(self, predictions, margins):
        body = self.formatter.build(
            "아이", ["ㅏ", "ㅣ"],
            [{"pred": p, "pred_ph": p} for p in predictions], margins,
            utt_scores={"pron": 4, "flu": 4, "comp": 4})
        # 실제 JSON 응답에서도 null 로 유지되는지 확인한다.
        return json.loads(json.dumps(body, allow_nan=False))["result"]

    def test_no_recognition_has_no_scores(self):
        result = self.build([None, None], [-100, -100])
        for key in ("overall", "pronunciation", "fluency", "rhythm", "comprehensibility"):
            self.assertIsNone(result[key])
        for word in result["words"]:
            self.assertIsNone(word["score"])
            self.assertIsNone(word["phonemes"][0]["pronunciation"])

    def test_missing_sound_is_excluded_from_average(self):
        result = self.build([None, "ㅣ"], [-100, 100])
        self.assertIsNone(result["words"][0]["score"])
        self.assertEqual(result["words"][1]["score"], 100)
        self.assertEqual(result["pronunciation"], 100)
        self.assertEqual(result["overall"], 92)

    def test_real_zero_is_preserved(self):
        result = self.build(["ㅓ", "ㅣ"], [-100, 100])
        self.assertEqual(result["words"][0]["score"], 0)
        self.assertEqual(result["words"][0]["phonemes"][0]["pronunciation"], 0)
        self.assertEqual(result["pronunciation"], 50)

    def test_missing_margin_stays_null(self):
        result = self.build(["ㅏ", "ㅣ"], [None, None])
        self.assertIsNone(result["pronunciation"])
        self.assertTrue(all(w["score"] is None for w in result["words"]))


if __name__ == "__main__":
    unittest.main()

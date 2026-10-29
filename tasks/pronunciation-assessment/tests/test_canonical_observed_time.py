import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.canonical_fix import CanonicalBuilder
from mdd.word_judge import KoCanonical
from mdd.canonical_observed_time import observed_time_reference, FixedReference


class ObservedTimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = CanonicalBuilder(KoCanonical())

    def test_attested_reading_is_selected_from_observed_phone(self):
        base = self.builder.fixed("여덟 시")
        q = base.index("ㅅ")
        sites = [dict(i=i, canon=t, pred=t) for i, t in enumerate(base)]
        sites[q]["pred"] = "ㅆ"
        result, accepted, _ = observed_time_reference(self.builder, "여덟 시", base, sites)
        self.assertEqual(accepted, [q])
        self.assertEqual(result[q], "ㅆ")
        self.assertEqual([i for i, (a, b) in enumerate(zip(base, result)) if a != b], [q])
        self.assertEqual(base[q], "ㅅ")

    def test_deletion_third_phone_and_unattested_context_are_not_accepted(self):
        for text in ("여덟 시", "여덟 시간", "열여덟 시", "여덟 시작", "사다 싸다"):
            base = self.builder.fixed(text)
            for actual in (None, "ㅊ", "ㅅ"):
                sites = [dict(i=i, canon=t, pred=actual) for i, t in enumerate(base)]
                result, accepted, _ = observed_time_reference(self.builder, text, base, sites)
                self.assertEqual(result, base)
                self.assertEqual(accepted, [])

    def test_reference_cannot_leak_to_another_request(self):
        fixed = FixedReference("여덟 시", list("ㅕㄷㅓㄹㅆㅣ"))
        self.assertEqual(fixed.variants("여덟 시")[1], [])
        with self.assertRaises(ValueError):
            fixed.fixed("시장에")


if __name__ == "__main__":
    unittest.main()

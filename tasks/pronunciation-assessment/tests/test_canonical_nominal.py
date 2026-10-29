import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

from mdd.canonical_fix import CanonicalBuilder
from mdd.canonical_nominal import NominalCorrectionBuilder, rewrite_nominal_suffixes
from mdd.word_judge import KoCanonical


class NominalCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = CanonicalBuilder(KoCanonical())
        cls.candidate = NominalCorrectionBuilder(cls.base)

    def test_documented_pronunciations_through_actual_g2p(self):
        expected = {"여덟도": "ㅕㄷㅓㄹㄷㅗ", "여덟과": "ㅕㄷㅓㄹㄱㅘ"}
        for text, phones in expected.items():
            self.assertNotEqual("".join(self.base.fixed(text)), phones)
            self.assertEqual("".join(self.candidate.fixed(text)), phones)
            self.assertEqual("".join(self.candidate.variants(text)[0]), phones)

    def test_vowel_particles_and_verbal_clusters_are_untouched(self):
        for text in ["여덟이", "여덟을", "여덟에", "여덟은", "밟고", "넓다", "짧지", "여덟과학", "열여덟도"]:
            self.assertEqual(rewrite_nominal_suffixes(text), text)
            self.assertEqual(self.candidate.variants(text), self.base.variants(text))

    def test_undefined_time_phrase_is_not_reclassified(self):
        for text in ["여덟 시", "여덟시", "여덟 시에 만나요", "여덟 시간", "여덟 시작"]:
            self.assertEqual(self.candidate.variants(text), self.base.variants(text))

    def test_sentence_boundaries_and_particle_combinations(self):
        self.assertEqual(rewrite_nominal_suffixes("여덟도, 여덟과는 달라요."), "여덜도, 여덜과는 달라요.")
        self.assertEqual(rewrite_nominal_suffixes("여덟과학 여덟도시"), "여덟과학 여덟도시")


if __name__ == "__main__":
    unittest.main()

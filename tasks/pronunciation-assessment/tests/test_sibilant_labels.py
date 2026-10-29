import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
from mdd.sibilant_labels import label_sites
from eval_sibilant_directional import all_optimal_mapping


class LabelsTests(unittest.TestCase):
    def labels(self, c, a):
        return label_sites(list(c), list(a), all_optimal_mapping(list(c), list(a)))

    def test_third_phone_and_deletion_retained(self):
        self.assertEqual(self.labels("ㅆㅏ", "ㅈㅏ")[0]["label_kind"], "third_phone")
        self.assertEqual(self.labels("ㅆㅏ", "ㅏ")[0]["label_kind"], "deletion")

    def test_ambiguity_is_not_a_deletion_label(self):
        site = self.labels("ㅆㅏㅆㅏ", "ㅆㅏ")[0]
        self.assertEqual(site["label_kind"], "ambiguous")
        self.assertGreater(len(site["alignment_options"]), 1)

    def test_indices_survive_unrelated_insertions(self):
        site = self.labels("ㅏㅆㅓ", "ㄱㅏㅅㅓ")[0]
        self.assertEqual((site["q"], site["a"], site["actual"]), (1, 2, "ㅅ"))
        self.assertEqual(site["label_kind"], "pair_error")

    def test_changed_following_vowel_not_filtered(self):
        site = self.labels("ㅆㅏ", "ㅆㄱ")[0]
        self.assertEqual(site["label_kind"], "correct")
        self.assertFalse(site["actual_onset"])


if __name__ == "__main__":
    unittest.main()

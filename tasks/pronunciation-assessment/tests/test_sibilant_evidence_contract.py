"""Meaningful failure-mode checks for the offline acoustic evidence join."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import json
import unittest

from mdd.sibilant_evidence import assemble_evidence


def fixture():
    item = {"id": "case", "source_sha256": "a" * 64, "tag_sha256": "b" * 64,
            "following_vowel": "ㅜ", "actual_vowel": "ㅜ", "tag_vowel": [1.64, 1.68], "tag_vowel_label": "ㅜ"}
    branches = {name: {"returned_frames": 1, "state": "candidate_returned", "path_choice_disagreement_frames": 0,
                      "path_choice_octave_review_frames": 0} for name in ("raw", "low_band")}
    setting = {"floor_hz": 100., "supported_frames": 1, "branches": branches, "branch_return_disagreements": 0}
    view = {"status": "observed_candidates_only", "interval_provenance": "synthetic_first_voicing_not_gold",
            "original_interval_s": [1.663, 1.703], "interval_s": [.663, .703], "glottal_source_validated": False,
            "pronunciation_decision": None, "cross_floor_comparisons": [], "settings": [setting]}
    observation = {"schema_version": 3, "id": "case", "source_sha256": "a" * 64, "TAG_sha256": "b" * 64,
        "sample_rate": 16000, "offset_s": 1., "pronunciation_decision": None, "new_human_gold": False,
        "vowel_context": {"expected_following_vowel": "ㅜ", "existing_TAG_following_vowel": "ㅜ", "labels_agree": True,
                          "status": "labels_agree_not_acoustically_verified"},
        "target_association": "not_independently_verified",
        "noise": {"waveform": "original_PCM", "frames": [], "spectral_top_hz": 7900.},
        "periodicity": {"settings": [{"floor_hz": 100., "frames": [{"paired_support_s": [.665, .695], "raw_cc_support_s": [.6675, .6925],
            "raw": {"candidate_hz": 150.}, "low_band": {"candidate_hz": 150.}}]}]},
        "intervals": {"reviewed_first_voicing_first40ms": view}}
    slope = {"status": "observation_not_phonation_class", "ols_db_per_s": 42., "endpoint_db_per_s": 43.,
             "support_s": [1.663, 1.693], "frame_support_original_s": [[1.663, 1.683], [1.673, 1.693]],
             "within_labeled_vowel": False, "containment_basis": "same_nearest_sample_quantization_as_requested_anchor"}
    contrast = {"status": "observation_not_phone_class", "vowel_minus_noise_db": -3., "within_labeled_vowel": False,
                "vowel_support_s": [1.663, 1.683], "noise_support_s": [1.61, 1.63]}
    intensity = {"id": "case", "source_sha256": "a" * 64, "expected_vowel": "ㅜ", "TAG_vowel": "ㅜ",
                 "TAG_vowel_span": [1.64, 1.68], "TAG_anchor": {"status": "anchor_missing", "variants": []},
                 "provisional_voicing_anchor": {"status": "observed_with_unverified_anchor", "variants": [
                     {"offset_ms": 0, "metrics": {"rise30_window20": slope}, "cv_contrast": contrast}]}}
    phones = [[1.60, 1.64, "ㅆ"], [1.64, 1.68, "ㅜ"], [1.68, 1.90, "ㅣ"]]
    return item, observation, intensity, phones


class EvidenceContract(unittest.TestCase):
    def test_contained_candidate_is_retained_but_does_not_become_verified_f0(self):
        item, obs, intensity, phones = fixture()
        view = obs["intervals"]["reviewed_first_voicing_first40ms"]
        view.update(original_interval_s=[1.64, 1.68], interval_s=[.64, .68])
        obs["periodicity"]["settings"][0]["frames"][0]["paired_support_s"] = [.645, .675]
        obs["periodicity"]["settings"][0]["frames"][0]["raw_cc_support_s"] = [.6475, .6725]
        result = assemble_evidence(item, obs, intensity, phones)
        evidence = result["periodicity_views"]["reviewed_first_voicing_first40ms"]
        self.assertTrue(evidence["any_common_support_candidate_contained_in_existing_TAG_vowel"])
        self.assertEqual(evidence["settings"][0]["common_support"]["raw"]["candidate_frame_indices_contained_in_existing_TAG_vowel"], [0])
        self.assertIsNone(evidence["validated_glottal_frequency_hz"])
        self.assertFalse(result["acoustic_target_identity_verified"])

    def test_crossing_into_next_vowel_keeps_values_without_glottal_or_error_claims(self):
        args = fixture()
        before = json.dumps(args, ensure_ascii=False, sort_keys=True)
        result = assemble_evidence(*args)
        view = result["periodicity_views"]["reviewed_first_voicing_first40ms"]
        self.assertIn("support_overlaps_other_existing_TAG_interval", view["review_flags"])
        self.assertIn("no_common_support_candidate_fully_contained_in_existing_TAG_vowel", view["review_flags"])
        self.assertFalse(view["any_common_support_candidate_contained_in_existing_TAG_vowel"])
        raw = view["settings"][0]["common_support"]["raw"]
        self.assertEqual(raw["candidate_frame_indices"], [0])
        self.assertEqual(raw["candidate_frame_indices_contained_in_existing_TAG_vowel"], [])
        self.assertFalse(raw["glottal_source_verified"])
        self.assertIsNone(view["validated_glottal_frequency_hz"])
        metric = result["intensity_views"]["provisional_voicing_anchor"]["variants"][0]["metrics"]["rise30_window20"]
        self.assertEqual(metric["value_db_per_s"], 42.)
        self.assertFalse(metric["support"]["acoustic_phone_containment_verified"])
        self.assertIsNone(result["pronunciation_decision"])
        self.assertFalse(result["pronunciation_error_inferred_from_missingness"])
        self.assertEqual(before, json.dumps(args, ensure_ascii=False, sort_keys=True))

    def test_missing_support_and_missing_candidate_remain_distinct(self):
        for no_support in (True, False):
            with self.subTest(no_support=no_support):
                item, obs, intensity, phones = fixture()
                setting = obs["intervals"]["reviewed_first_voicing_first40ms"]["settings"][0]
                if no_support:
                    obs["periodicity"]["settings"][0]["frames"] = []
                    setting["supported_frames"] = 0
                else:
                    for branch in ("raw", "low_band"):
                        obs["periodicity"]["settings"][0]["frames"][0][branch]["candidate_hz"] = None
                for branch in setting["branches"].values():
                    branch["returned_frames"] = 0
                    branch["state"] = "no_locally_supported_frame" if no_support else "no_candidate_returned"
                result = assemble_evidence(item, obs, intensity, phones)
                flags = result["periodicity_views"]["reviewed_first_voicing_first40ms"]["review_flags"]
                self.assertIn(("no_locally_supported_periodicity_frame" if no_support else "no_periodicity_candidate_returned") + "_on_common_support", flags)
                self.assertIsNone(result["periodicity_views"]["reviewed_first_voicing_first40ms"]["any_common_support_candidate_contained_in_existing_TAG_vowel"])
                self.assertIsNone(result["pronunciation_decision"])

    def test_raw_observation_is_not_discarded_when_filtered_comparison_needs_more_context(self):
        item, obs, intensity, phones = fixture()
        obs["intervals"]["reviewed_first_voicing_first40ms"].update(original_interval_s=[1.65, 1.69], interval_s=[.65, .69])
        frame = obs["periodicity"]["settings"][0]["frames"][0]
        frame.update(paired_support_s=[.656, .68125], raw_cc_support_s=[.6585, .67875])
        result = assemble_evidence(item, obs, intensity, phones)
        view = result["periodicity_views"]["reviewed_first_voicing_first40ms"]
        self.assertFalse(view["any_common_support_candidate_contained_in_existing_TAG_vowel"])
        self.assertTrue(view["any_branch_specific_candidate_contained_in_existing_TAG_vowel"])
        self.assertEqual(view["settings"][0]["branch_specific_support"]["raw"]["candidate_frame_indices_contained_in_existing_TAG_vowel"], [0])
        self.assertIsNone(view["validated_glottal_frequency_hz"])

    def test_missing_anchor_is_not_zero_or_normal(self):
        item, obs, intensity, phones = fixture()
        obs["intervals"] = {"reviewed_cycle_group": {"status": "interval_not_annotated", "interval_provenance": "not_gold"}}
        result = assemble_evidence(item, obs, intensity, phones)
        self.assertEqual(result["periodicity_views"]["reviewed_cycle_group"]["settings"], [])
        self.assertIn("interval_not_annotated", result["periodicity_views"]["reviewed_cycle_group"]["review_flags"])
        self.assertFalse(result["pronunciation_error_inferred_from_missingness"])

    def test_inconsistent_identity_or_reference_is_rejected(self):
        for field in ("id", "source_sha256", "TAG_vowel_span"):
            with self.subTest(field=field):
                args = list(fixture())
                args[2][field] = [1.60, 1.64] if field == "TAG_vowel_span" else "different"
                with self.assertRaises(ValueError):
                    assemble_evidence(*args)

    def test_stale_containment_and_forged_support_are_rejected(self):
        for change in ("within_labeled_vowel", "support_s"):
            with self.subTest(change=change):
                args = fixture()
                metric = args[2]["provisional_voicing_anchor"]["variants"][0]["metrics"]["rise30_window20"]
                metric[change] = True if change == "within_labeled_vowel" else [1.663, 1.68]
                with self.assertRaises(ValueError):
                    assemble_evidence(*args)

    def test_agreement_does_not_validate_phonation_and_disagreement_does_not_infer_error(self):
        args = fixture()
        args[0]["following_vowel"] = "ㅡ"
        args[1]["vowel_context"].update(expected_following_vowel="ㅡ", labels_agree=False, status="context_label_disagreement")
        args[2]["expected_vowel"] = "ㅡ"
        result = assemble_evidence(*args)
        self.assertEqual(result["context_review_flags"], ["vowel_context_label_disagreement"])
        self.assertFalse(result["new_human_gold"])
        args[1]["vowel_context"]["labels_agree"] = True
        with self.assertRaisesRegex(ValueError, "agreement flag"):
            assemble_evidence(*args)
        args[1]["vowel_context"]["labels_agree"] = False
        args[1]["intervals"]["reviewed_first_voicing_first40ms"]["glottal_source_validated"] = True
        with self.assertRaisesRegex(ValueError, "unsupported phonetic validation"):
            assemble_evidence(*args)

    def test_nonfinite_observation_and_wrong_trace_counts_are_rejected(self):
        args = fixture()
        args[2]["provisional_voicing_anchor"]["variants"][0]["metrics"]["rise30_window20"]["ols_db_per_s"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite intensity"):
            assemble_evidence(*args)
        args = fixture()
        args[1]["intervals"]["reviewed_first_voicing_first40ms"]["settings"][0]["supported_frames"] = 2
        with self.assertRaisesRegex(ValueError, "support count"):
            assemble_evidence(*args)
        args = fixture()
        args[1]["periodicity"]["settings"][0]["frames"][0]["raw"]["candidate_hz"] = float("inf")
        with self.assertRaisesRegex(ValueError, "selected periodicity"):
            assemble_evidence(*args)


if __name__ == "__main__":
    unittest.main()

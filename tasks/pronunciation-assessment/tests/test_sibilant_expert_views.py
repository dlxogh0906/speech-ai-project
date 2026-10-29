"""Synthetic-only checks of expert-event handoff; these are not human labels."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import copy
import importlib.util
import json
import unittest

path = ROOT / "tasks/pronunciation-assessment/scripts/eval/export_sibilant_expert_views.py"
spec = importlib.util.spec_from_file_location("expert_handoff_test", str(path))
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def fixture():
    item = {"id": "synthetic_only", "source_sha256": "a" * 64, "tag_sha256": "b" * 64,
            "analysis_interval": [10., 10.2], "tag_vowel": [10.01, 10.11]}
    frames = [{"center_s": t, "raw_cc_support_s": [t - .01, t + .01],
               "paired_support_s": [t - .0125, t + .0125],
               "raw": {"candidate_hz": f}, "low_band": {"candidate_hz": f}}
              for t, f in ((.025, 150.), (.08, 180.))]
    observation = {"schema_version": 3, "id": item["id"], "source_sha256": item["source_sha256"],
                   "TAG_sha256": item["tag_sha256"], "sample_rate": 16000, "offset_s": 10.,
                   "analysis_interval_samples": [160000, 163200], "pronunciation_decision": None,
                   "new_human_gold": False, "vowel_context": {"synthetic_test": True},
                   "periodicity": {"clip_duration_s": .2, "time_step_s": .005,
                                   "settings": [{"floor_hz": 100., "frames": frames}]}}
    points = {"frication_start": 10., "frication_to_diffuse_noise": None,
              "first_vowel_voicing": 10.01, "first_regular_run_start": 10.014,
              "first_regular_run_end": 10.036}
    review = {"status": "complete_review", "actor_type": "human_self_declared_expert",
              "target_association": "confirmed", "transition_kind": "overlap",
              "landmark_status": {k: "observed" if v is not None else "absent" for k, v in points.items()},
              "points_original_seconds": points, "uncertain_ranges_original_seconds": {},
              "note": "SYNTHETIC FIXTURE, NOT A HUMAN RESPONSE"}
    annotation = {"id": item["id"], "source_sha256": item["source_sha256"],
                  "reviewers": {"synthetic_A": review, "synthetic_B": copy.deepcopy(review)}}
    annotation["reviewers"]["synthetic_B"]["points_original_seconds"]["first_vowel_voicing"] = 10.06
    annotation["reviewers"]["synthetic_B"]["points_original_seconds"]["first_regular_run_start"] = 10.07
    annotation["reviewers"]["synthetic_B"]["points_original_seconds"]["first_regular_run_end"] = 10.09
    phones = [[10., 10.01, "ㅆ"], [10.01, 10.11, "ㅜ"], [10.11, 10.2, "ㅣ"]]
    return item, observation, annotation, phones


class ExpertHandoff(unittest.TestCase):
    def test_reviewers_keep_distinct_times_and_candidates_without_mutation(self):
        args = fixture()
        before = json.dumps(args, sort_keys=True)
        result = handoff.expert_views(*args)
        for who, index in (("synthetic_A", 0), ("synthetic_B", 1)):
            view = result["reviewers"][who]["periodicity_views"]["first_voicing_40ms"]
            self.assertEqual(view["settings"][0]["common_support"]["raw"]["candidate_frame_indices"], [index])
            self.assertIsNone(view["validated_glottal_frequency_hz"])
        self.assertFalse(result["reviewer_measurements_averaged"])
        self.assertFalse(result["new_adjudicated_gold"])
        self.assertFalse(result["independence_verified"])
        self.assertEqual(json.dumps(args, sort_keys=True), before)

    def test_raw_support_survives_when_common_support_does_not_fit(self):
        result = handoff.expert_views(*fixture())
        view = result["reviewers"]["synthetic_A"]["periodicity_views"]["regular_run"]
        setting = view["settings"][0]
        self.assertEqual(setting["branch_specific_support"]["raw"]["candidate_frame_indices"], [0])
        self.assertEqual(setting["common_support"]["raw"]["candidate_frame_indices"], [])

    def test_absent_uncertain_unreviewed_do_not_become_points_or_zeroes(self):
        for state in ("absent", "uncertain", "unreviewed"):
            with self.subTest(state=state):
                args = fixture()
                review = args[2]["reviewers"]["synthetic_A"]
                review["landmark_status"]["first_vowel_voicing"] = state
                review["points_original_seconds"]["first_vowel_voicing"] = None
                if state == "uncertain":
                    review["uncertain_ranges_original_seconds"]["first_vowel_voicing"] = [10.01, 10.04]
                result = handoff.expert_views(*args)["reviewers"]["synthetic_A"]
                view = result["periodicity_views"]["first_voicing_40ms"]
                self.assertEqual(view["observation_status"], "required_landmark_not_observed")
                self.assertNotIn("requested_support", view)
                self.assertFalse(view["settings"])
                self.assertEqual(result["annotation"], review)

    def test_unattested_points_and_missing_file_remain_unmeasured(self):
        for missing in (False, True):
            args = fixture()
            review = args[2]["reviewers"]["synthetic_A"]
            if missing:
                review.clear()
                review["status"] = "missing_file"
            else:
                review.update(actor_type="unattested", status="unattested_points")
            view = handoff.expert_views(*args)["reviewers"]["synthetic_A"]["periodicity_views"]["first_voicing_40ms"]
            self.assertEqual(view["observation_status"], "reviewer_input_not_attested")
            self.assertEqual(view["settings"], [])

    def test_unconfirmed_target_cannot_borrow_neighboring_voicing(self):
        for association in ("uncertain", "outside_clip", "unreviewed"):
            args = fixture()
            args[2]["reviewers"]["synthetic_A"]["target_association"] = association
            view = handoff.expert_views(*args)["reviewers"]["synthetic_A"]["periodicity_views"]["first_voicing_40ms"]
            self.assertEqual(view["observation_status"], "target_association_not_confirmed")
            self.assertEqual(view["settings"], [])

    def test_context_annotation_outside_cached_clip_is_not_clipped_or_replaced(self):
        args = fixture()
        args[2]["reviewers"]["synthetic_A"]["points_original_seconds"]["first_vowel_voicing"] = 10.19
        view = handoff.expert_views(*args)["reviewers"]["synthetic_A"]["periodicity_views"]["first_voicing_40ms"]
        self.assertEqual(view["observation_status"], "interval_outside_analysis_clip")
        self.assertAlmostEqual(view["requested_support"]["support_original_s"][1], 10.23)
        self.assertEqual(view["settings"], [])

    def test_window_crossing_next_label_is_retained_as_unverified_observation(self):
        args = fixture()
        args[2]["reviewers"]["synthetic_A"]["points_original_seconds"]["first_vowel_voicing"] = 10.10
        view = handoff.expert_views(*args)["reviewers"]["synthetic_A"]["periodicity_views"]["first_voicing_40ms"]
        self.assertEqual(view["observation_status"], "observed_candidates_only")
        self.assertIn("support_overlaps_other_existing_TAG_interval", view["review_flags"])
        self.assertFalse(view["requested_support"]["acoustic_phone_containment_verified"])

    def test_mismatched_identity_hash_or_timebase_is_rejected(self):
        for field, value in (("id", "wrong"), ("source_sha256", "c" * 64), ("offset_s", 9.),
                             ("analysis_interval_samples", [160001, 163200]), ("TAG_sha256", "d" * 64)):
            args = fixture()
            args[1][field] = value
            with self.assertRaises(ValueError):
                handoff.expert_views(*args)

    def test_invalid_points_are_rejected(self):
        for value in (float("nan"), True, "10.01", None):
            args = fixture()
            args[2]["reviewers"]["synthetic_A"]["points_original_seconds"]["first_vowel_voicing"] = value
            with self.assertRaises(ValueError):
                handoff.expert_views(*args)


if __name__ == "__main__":
    unittest.main()

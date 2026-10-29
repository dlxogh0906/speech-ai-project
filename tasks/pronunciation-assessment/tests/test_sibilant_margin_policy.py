import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

from mdd.sibilant_margin_policy import fit_zero_extra_policies, apply_margin_policy


def fixture(vowels):
    labels, features = [], []
    for i, v in enumerate(vowels):
        labels.append(dict(expected="ㅆ", actual="ㅆ", state="available", uid=str(i), q_runtime=0,
                           feature_index=i, baseline_phone={}, runtime_following=v, following="ㅓ"))
        features.append(dict(uid=str(i), q_runtime=0, canonical="ㅆ", following=v,
                             baseline_pred="ㅆ", token_margin=-1.))
    return labels, features


class MarginPolicyTests(unittest.TestCase):
    def test_runtime_vowel_not_tag_vowel_controls_calibration(self):
        labels, features = fixture(["ㅣ"] * 100)
        policies, audit = fit_zero_extra_policies(labels, features)
        th = policies["vowel_zero_extra_calibration_fp"]["thresholds"]
        self.assertIsNotNone(th["ㅣ"])
        self.assertIsNone(th["ㅓ"])
        self.assertEqual(audit["following_vowel"]["ㅣ"]["additional_calibration_fp"], 0)

    def test_undersampled_vowel_does_not_borrow_global_threshold(self):
        labels, features = fixture(["ㅣ"] * 100 + ["ㅜ"] * 99)
        policies, _ = fit_zero_extra_policies(labels, features)
        canon = list("ㅆㅜㅆㅣ")
        sites = [dict(i=i, canon=c, pred=c) for i, c in enumerate(canon)]
        result = apply_margin_policy(canon, sites, {0: .001, 2: .001}, policies["vowel_zero_extra_calibration_fp"])
        self.assertNotIn("sz", result[0])
        self.assertIn("sz", result[2])
        self.assertTrue(all("sz" not in s for s in sites))

    def test_errors_and_rejections_do_not_set_normal_threshold(self):
        labels, features = fixture(["ㅣ"] * 100)
        before, _ = fit_zero_extra_policies(labels, features)
        labels += [dict(expected="ㅆ", actual="ㅅ", state="available", feature_index=None),
                   dict(expected="ㅆ", actual="ㅆ", state="input_rejected", feature_index=None)]
        after, _ = fit_zero_extra_policies(labels, features)
        self.assertEqual(before, after)

    def test_already_alerted_normals_do_not_consume_additional_tail(self):
        labels, features = fixture(["ㅣ"] * 101)
        features[-1]["token_margin"] = -100
        labels[-1]["baseline_phone"] = {"errorType": "already_wrong"}
        _, audit = fit_zero_extra_policies(labels, features)
        self.assertLess(audit["global"]["threshold"], .9)
        self.assertEqual(audit["global"]["existing_alerts"], 1)

    def test_saturation_disables_instead_of_admitting_a_tie(self):
        labels, features = fixture(["ㅣ"] * 100)
        features[-1]["token_margin"] = -100
        policies, audit = fit_zero_extra_policies(labels, features)
        self.assertIsNone(policies["global_zero_extra_calibration_fp"]["thresholds"]["ㅆ"])
        self.assertEqual(audit["global"]["disabled_reason"], "saturated_normal_score")

    def test_normal_feature_identity_must_match(self):
        labels, features = fixture(["ㅣ"] * 100)
        features[-1]["uid"] = "wrong"
        with self.assertRaises(ValueError):
            fit_zero_extra_policies(labels, features)

    def test_other_phone_substitutions_are_preserved(self):
        canon = list("ㅆㅣㅅㅣ")
        sites = [dict(i=i, canon=c, pred=c) for i, c in enumerate(canon)]
        sites[0]["pred"] = "ㅈ"
        policy = dict(mode="following_vowel", expected="ㅆ", thresholds={"ㅣ": .7})
        self.assertEqual(apply_margin_policy(canon, sites, {0: 0., 2: 1.}, policy), sites)


if __name__ == "__main__":
    unittest.main()

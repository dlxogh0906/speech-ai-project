"""Calibration-only margin policies for the frozen expanded corpus comparison.

Conditioning uses the following vowel from runtime canonical, not a TAG vowel
or a phonetic boundary. A margin transformation is not an error probability.
No service imports or activates this module by default.
"""
import math

import numpy as np

from mdd.sibilant_transition import PAIR, VOWELS, annotate_probabilities


def score_ss(margin):
    value = float(margin)
    if not math.isfinite(value):
        raise ValueError("finite paired margin required")
    return float(1 / (1 + np.exp(-np.clip(value, -700, 700))))


def fit_zero_extra_policies(calibration_rows, features, min_normals=100):
    """Use only normal calibration labels; never select on error/test scores.

    The sample count is mapped accepted normals, including unmeasured cases.
    Tail maxima use measurable, eligible normals with no existing alert.
    The caller attaches runtime_following from cached runtime canonical.
    """
    if min_normals < 1:
        raise ValueError("positive normal sample count required")
    normals = [r for r in calibration_rows if r["expected"] == r["actual"] == "ㅆ"
               and r["state"] in ("available", "features_unavailable")]

    def fit(rows):
        scores = []
        for row in rows:
            ix = row.get("feature_index")
            if ix is None or (row.get("baseline_phone") or {}).get("errorType"):
                continue
            f = features[ix]
            if f["uid"] != row["uid"] or f["q_runtime"] != row["q_runtime"]:
                raise ValueError("feature/label runtime key mismatch")
            if f["canonical"] != "ㅆ" or f["following"] != row.get("runtime_following"):
                raise ValueError("runtime context mismatch")
            if f["baseline_pred"] in PAIR:
                scores.append(1 - score_ss(f["token_margin"]))
        threshold, reason = None, None
        if len(rows) < min_normals:
            reason = "insufficient_normal_sites"
        elif not scores:
            reason = "no_eligible_unflagged_normals"
        else:
            threshold = max(.5, float(np.nextafter(max(scores), np.inf)))
            if threshold > 1:
                threshold, reason = None, "saturated_normal_score"
        added = sum(s >= threshold for s in scores) if threshold is not None else 0
        if added:
            raise AssertionError("normal calibration tail was not excluded")
        return {"threshold": threshold, "normal_sites": len(rows), "eligible_unflagged_normals": len(scores),
                "normal_speakers": len({r.get("speaker", r["uid"]) for r in rows}),
                "normal_items": len({r.get("item") for r in rows}),
                "normal_utterances": len({r["uid"] for r in rows}),
                "existing_alerts": sum(bool((r.get("baseline_phone") or {}).get("errorType")) for r in rows),
                "maximum_unflagged_score": max(scores) if scores else None,
                "additional_calibration_fp": added, "disabled_reason": reason}

    global_audit = fit(normals)
    by_vowel = {v: fit([r for r in normals if r.get("runtime_following") == v]) for v in VOWELS}
    policies = {
        "global_zero_extra_calibration_fp": {"mode": "global", "thresholds": {"ㅅ": None, "ㅆ": global_audit["threshold"]}},
        "vowel_zero_extra_calibration_fp": {"mode": "following_vowel", "expected": "ㅆ",
                                           "thresholds": {v: a["threshold"] for v, a in by_vowel.items()}}}
    return policies, {"min_normals": min_normals, "global": global_audit, "following_vowel": by_vowel,
                      "unknown_runtime_vowel_normals": sum(r.get("runtime_following") not in VOWELS for r in normals)}


def apply_margin_policy(canon, sites, scores_ss, policy):
    if policy["mode"] == "global":
        return annotate_probabilities(canon, sites, scores_ss, policy["thresholds"])
    if policy["mode"] != "following_vowel" or policy.get("expected") != "ㅆ":
        raise ValueError("unsupported policy")
    result = annotate_probabilities(canon, sites, {}, {"ㅅ": None, "ㅆ": None})
    groups = {}
    for q, score in scores_ss.items():
        if not isinstance(q, int) or not 0 <= q < len(canon):
            raise ValueError("invalid runtime target index")
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            raise ValueError("invalid margin score")
        if canon[q] == "ㅆ" and q + 1 < len(canon) and canon[q + 1] in VOWELS:
            groups.setdefault(canon[q + 1], {})[q] = score
    for vowel, values in groups.items():
        # Missing/undersampled vowels stay off; there is no global fallback.
        threshold = policy["thresholds"].get(vowel)
        changed = annotate_probabilities(canon, sites, values, {"ㅅ": None, "ㅆ": threshold})
        for q in values:
            result[q] = changed[q]
    return result

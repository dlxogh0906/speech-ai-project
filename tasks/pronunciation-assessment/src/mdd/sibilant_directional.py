"""Offline-only annotation adapter for a narrowly scoped ㅆ→ㅅ candidate.

This module does not load models or change formatter globals. The caller must
freeze canonical with the original model, align both recognizers to it, and
explicitly enable SpeechSuperFormatter's SZ_RULE hook for the candidate body.
The annotation's p is token confidence, NOT a calibrated error probability.
Nothing imports this adapter in the serving application.
"""

import copy
import math


def annotate_directional(canon, base_sites, candidate_sites, vowels,
                         min_prob=0.6, allow_post_coda=False):
    """Return copied baseline sites and an audit; change only `sz` at target sites.

Default policy is first onset / after a vowel, matching the scope of the old
native-speech safety observation. Broad post-coda activation is diagnostic.
Existing deletions or substitutions to any third phone are preserved.
No actual-phone labels are accepted or consulted by this function.
"""
    if not math.isfinite(float(min_prob)) or not 0 <= min_prob <= 1:
        raise ValueError("min_prob must be finite and in [0,1]")
    if len(base_sites) != len(canon) or len(candidate_sites) != len(canon):
        raise ValueError("Both recognizers must be aligned to the same frozen canonical")
    for i, (b, c) in enumerate(zip(base_sites, candidate_sites)):
        if b.get("i") != i or c.get("i") != i or b.get("canon") != canon[i] or c.get("canon") != canon[i]:
            raise ValueError("Site index/canonical mismatch at %d" % i)
        if "sz" in b:
            raise ValueError("Baseline already has detector annotations")
    result = copy.deepcopy(base_sites)
    rows = []
    for i, token in enumerate(canon):
        if token != "ㅆ":
            continue
        b, c = base_sites[i], candidate_sites[i]
        value = c.get("prob")
        try:
            prob = float(value) if value is not None else None
        except (TypeError, ValueError):
            prob = None
        row = {"i": i, "base_pred": b.get("pred"), "candidate_pred": c.get("pred"),
               "candidate_prob": prob if prob is not None and math.isfinite(prob) else None,
               "preceding": canon[i - 1] if i else None,
               "following": canon[i + 1] if i + 1 < len(canon) else None}
        if i + 1 >= len(canon) or canon[i + 1] not in vowels:
            reason = "not_onset"
        elif not allow_post_coda and i > 0 and canon[i - 1] not in vowels:
            reason = "post_coda_excluded"
        elif b.get("pred") not in ("ㅆ", "ㅅ"):
            reason = "baseline_other_or_deleted"
        elif c.get("pred") != "ㅅ":
            reason = "candidate_not_s"
        elif prob is None or not math.isfinite(prob) or not 0 <= prob <= 1:
            reason = "invalid_confidence"
        elif prob < min_prob:
            reason = "below_confidence"
        else:
            reason = "annotated"
            result[i]["sz"] = {"alt": "ㅅ", "p": prob, "th": float(min_prob)}
        row.update(reason=reason, annotated=reason == "annotated")
        rows.append(row)
    audit = {"min_prob": float(min_prob), "allow_post_coda": bool(allow_post_coda),
             "annotated_indices": [r["i"] for r in rows if r["annotated"]], "sites": rows,
             "confidence_kind": "greedy token posterior, uncalibrated"}
    return result, audit

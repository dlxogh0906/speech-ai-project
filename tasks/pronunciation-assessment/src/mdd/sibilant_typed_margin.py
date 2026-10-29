"""Inactive confidence candidate for an already recognized third consonant.

Preserves the recognized actual phone; no binary replacement of third phones,
deletion inference or interpretation of CTC frames as acoustic boundaries.
"""
import copy
import math
import numpy as np
from mdd.canonical_fix import CONSONANTS
from mdd.sibilant_transition import PAIR, VOWELS, paired_anchor


def observations(lp, canonical, sites, index, blank):
    lp = np.asarray(lp)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all() or len(sites) != len(canonical):
        raise ValueError("finite logits and aligned sites required")
    ids = [index(t) for t in canonical]
    output = {}
    for q, expected in enumerate(canonical[:-1]):
        if expected not in PAIR or canonical[q + 1] not in VOWELS:
            continue
        alt = sites[q].get("pred")
        if alt not in CONSONANTS or alt in PAIR:
            output[q] = {"state": "not_third_consonant", "alt": alt}
            continue
        anchor = paired_anchor(lp, ids, q, [index(expected), index(alt)], blank)
        if anchor is None:
            output[q] = {"state": "paired_alignment_unavailable", "alt": alt}
            continue
        frames = sorted(set(sum(anchor["segments"], [])))
        margin = float(lp[frames, index(alt)].max() - lp[frames, index(expected)].max())
        score = float(1 / (1 + np.exp(-np.clip(margin, -700, 700))))
        output[q] = {"state": "available", "alt": alt, "score": score, "margin": margin,
                     "frames": frames, "hypothesis_frames": anchor["hypothesis_frames"]}
    return output


def annotate(canonical, sites, values, thresholds):
    if set(thresholds) != set(PAIR) or len(sites) != len(canonical):
        raise ValueError("aligned sites and both thresholds required")
    for threshold in thresholds.values():
        if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("invalid threshold")
    out = copy.deepcopy(sites)
    for q, value in values.items():
        q = int(q)
        if value["state"] != "available":
            continue
        if not 0 <= q < len(canonical) - 1 or canonical[q] not in PAIR or canonical[q + 1] not in VOWELS:
            raise ValueError("invalid runtime onset")
        alt, score = value["alt"], float(value["score"])
        if alt not in CONSONANTS or alt in PAIR or alt != sites[q].get("pred"):
            raise ValueError("typed candidate must preserve a recognized third consonant")
        if not math.isfinite(score) or not 0 <= score <= 1 or "sz" in sites[q]:
            raise ValueError("invalid score or overlapping annotation")
        threshold = thresholds[canonical[q]]
        if threshold is not None and score >= threshold:
            out[q]["sz"] = {"alt": alt, "p": score, "th": threshold}
    return out


def calibrate(rows, minimum_normals=100):
    """Cal-only strict negative tail: wrong actual types count as negatives too."""
    thresholds, audit = {}, {}
    for expected in PAIR:
        group = [r for r in rows if r["expected"] == expected]
        normals = [r for r in group if r["label_kind"] == "correct" and r["view"]["phone"] is not None]
        negatives, positives = [], []
        for row in group:
            value, phone = row.get("observation"), row["view"]["phone"]
            if value is None or value["state"] != "available" or phone is None or phone.get("errorType"):
                continue
            positive = (row["label_kind"] not in ("correct", "ambiguous") and
                        phone.get("alpha_jamo") == expected and row["actual"] == value["alt"])
            (positives if positive else negatives).append(float(value["score"]))
        threshold = None
        if len(normals) >= minimum_normals and negatives:
            threshold = max(.5, float(np.nextafter(max(negatives), np.inf)))
            if threshold > 1:
                threshold = None
        thresholds[expected] = threshold
        audit[expected] = {"known_normals": len(normals), "negative_positions": len(negatives),
                           "positive_positions": len(positives), "threshold": threshold,
                           "cal_new_wrong_or_normal_alerts": sum(v >= threshold for v in negatives) if threshold is not None else 0,
                           "cal_potential_exact_gains": sum(v >= threshold for v in positives) if threshold is not None else 0}
    return thresholds, audit

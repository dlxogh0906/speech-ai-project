"""Unregistered runtime candidate using symmetric s/ss CTC alignments.

No extra acoustic model or expert boundary is required. Scores are transformed
token margins, NOT calibrated pronunciation-error probabilities. Adoption must
use fresh response-level evidence, not the pilot that selected this candidate.
"""
import math

import numpy as np

from mdd.sibilant_transition import PAIR, VOWELS, paired_anchor, annotate_probabilities


def margin_observations(lp, canon, token_index, blank):
    lp = np.asarray(lp)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all():
        raise ValueError("finite frame log probabilities required")
    pairs = [token_index("ㅅ"), token_index("ㅆ")]
    ids = [token_index(t) for t in canon]
    observations = {}
    for q, token in enumerate(canon):
        if token not in PAIR or q + 1 >= len(canon) or canon[q + 1] not in VOWELS:
            continue
        anchor = paired_anchor(lp, ids, q, pairs, blank)
        if anchor is None:
            observations[q] = {"state": "paired_alignment_unavailable"}
            continue
        frames = sorted(set(sum(anchor["segments"], [])))
        margin = float(lp[frames, pairs[1]].max() - lp[frames, pairs[0]].max())
        # Use the same float64 transformation as frozen calibration.
        score_ss = float(1 / (1 + np.exp(-np.clip(margin, -700, 700))))
        observations[q] = {"state": "available", "score_ss": score_ss, "margin_ss_minus_s": margin,
                           "frames": frames, "hypothesis_frames": anchor["hypothesis_frames"]}
    return observations


def annotate_margin(lp, canon, base_sites, token_index, blank, thresholds):
    if set(thresholds) != set(PAIR):
        raise ValueError("both directional thresholds, including disabled None, are required")
    for threshold in thresholds.values():
        if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("invalid fixed threshold")
    observations = margin_observations(lp, canon, token_index, blank)
    scores = {q: r["score_ss"] for q, r in observations.items() if r["state"] == "available"}
    return annotate_probabilities(canon, base_sites, scores, thresholds), observations

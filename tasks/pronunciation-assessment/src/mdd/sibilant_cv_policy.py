"""Offline confirmation policy for a frozen four-class C/V readout.

Only runtime canonical phones, decoder predictions, model probabilities and
visible API flags enter this policy. TAG outcomes and boundaries are not inputs.
The agreement arm confirms the decoder's opposite sibilant; it cannot replace a
decoder prediction of the expected phone with an opposite phone.
"""
import copy
import math

from mdd.sibilant_transition import PAIR, VOWELS


def annotate(canonical, sites, phones, observations, thresholds, agreement=True):
    result = copy.deepcopy(sites)
    states = {}
    for q, value in observations.items():
        if (q >= len(canonical) - 1 or canonical[q] not in PAIR
                or canonical[q + 1] not in VOWELS):
            states[q] = "ineligible_context"
            continue
        if phones[q].get("errorType"):
            states[q] = "existing_alert_preserved"
            continue
        expected, alternative = canonical[q], PAIR[canonical[q]]
        actual = sites[q].get("pred")
        if actual not in PAIR or agreement and actual != alternative:
            states[q] = "decoder_disagrees"
            continue
        if not value["available"]:
            states[q] = "support_unavailable"
            continue
        probabilities = value["probabilities"]
        if (len(probabilities) != 4 or any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in probabilities)
                or abs(sum(probabilities) - 1.) > .001):
            raise ValueError("invalid four-class probabilities")
        target = int(alternative == "ㅆ")
        if max(range(4), key=lambda i: probabilities[i]) != target:
            states[q] = "head_disagrees"
            continue
        threshold = thresholds[expected + "/pair"]
        if threshold is None:
            states[q] = "disabled_channel"
            continue
        confidence = float(probabilities[target])
        if confidence < threshold:
            states[q] = "below_threshold"
            continue
        result[q]["sz"] = {"alt": alternative, "p": confidence, "th": threshold}
        states[q] = "confirmed"
    return result, states

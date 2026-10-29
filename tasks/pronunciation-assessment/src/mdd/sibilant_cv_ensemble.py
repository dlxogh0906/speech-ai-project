"""Combine frozen runtime confirmations in a fixed order without score mixing."""
import copy
from mdd.sibilant_cv_policy import annotate


def annotate_heads(canonical, sites, phones, heads):
    result, claimed, traces = copy.deepcopy(sites), set(), {}
    for head in heads:
        if head["name"] in traces:
            raise ValueError("duplicate head name")
        observations = {q: v for q, v in head["observations"].items() if q not in claimed}
        result, states = annotate(canonical, result, phones, observations, head["thresholds"], agreement=True)
        claimed.update(q for q, state in states.items() if state == "confirmed")
        traces[head["name"]] = states
    return result, traces

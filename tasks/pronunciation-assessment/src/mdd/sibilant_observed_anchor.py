"""Observed greedy CTC token centers; these are not acoustic boundaries."""
import itertools
import numpy as np


def observed_tokens(log_probs, dictionary, blank):
    if log_probs.ndim != 2 or not len(log_probs) or not np.isfinite(log_probs).all():
        raise ValueError("finite nonempty frame-by-vocabulary log probabilities required")
    tokens, spans, start = [], [], 0
    for value, group in itertools.groupby(log_probs.argmax(-1)):
        end = start + len(list(group))
        token = dictionary[int(value)]
        if value != blank and len(token) == 1 and 0x3131 <= ord(token) <= 0x3163:
            tokens.append(token)
            spans.append((start, end))
        start = end
    return tokens, spans


def linked_anchor(tokens, spans, site):
    j = site.get("j")
    if j is None:
        return None
    if not 0 <= j < len(tokens) or tokens[j] != site.get("pred"):
        raise ValueError("stored decoder site does not match observed token sequence")
    left, right = spans[j]
    return {"frame": int(round((left + right - 1) / 2.)), "token_index": j,
            "token": tokens[j], "greedy_span": [left, right]}

"""Resolve TAG token indices without inventing internal grouped boundaries."""
import numpy as np


def single_phone_intervals(tag_phones, actual):
    flat, intervals = [], []
    for start, end, token in tag_phones:
        # Match the original TAG-to-transcript parser: internal whitespace is
        # formatting, while a grouped phone region still has no inner bounds.
        token = ''.join(c for c in token if not c.isspace())
        if token in ("", ".", "-", "spn", "<unk>"):
            continue
        flat.extend(token)
        intervals.extend([(float(start), float(end), token)] if len(token) == 1 else [None] * len(token))
    return intervals if flat == list(actual) else None


def coarse_roles(times, intervals, actual_index, vowels, prefix=.12, edge=.01):
    unknown = np.full(len(times), -100, np.int64)
    if intervals is None or actual_index is None or actual_index + 1 >= len(intervals):
        return unknown
    c, v = intervals[actual_index:actual_index + 2]
    if c is None or v is None or v[2] not in vowels:
        return unknown
    if not (np.isfinite(c[:2]).all() and np.isfinite(v[:2]).all() and c[0] < c[1] <= v[0] < v[1]):
        return unknown
    end = min(v[1], v[0] + prefix)
    mid = (v[0] + end) / 2
    roles = np.zeros(len(times), np.int64)
    roles[(times >= c[0]) & (times < c[1])] = 1
    roles[(times >= v[0]) & (times < mid)] = 2
    roles[(times >= mid) & (times < end)] = 3
    for boundary in (c[0], c[1], v[0], mid, end):
        roles[abs(times - boundary) <= edge] = -100
    return roles

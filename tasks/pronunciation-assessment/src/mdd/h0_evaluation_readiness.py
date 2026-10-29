"""Label-independent windows and conservative annotation checks for RP audits.

This module does not infer phones, repair unknowns, or select easy recordings.
"""
import math
import re
import numpy as np


def normal_text(value):
    return re.sub('[^가-힣0-9A-Za-z]', '', value or '')


def label_kind(value, vocabulary, reference=False):
    value = str(value).strip()
    if not value:
        return 'empty'
    if value == '-' and not reference:
        return 'explicit_deletion'
    if all(c in vocabulary or c.isspace() for c in value):
        return 'phones'
    return 'unknown'


def audio_windows(audio, sr, mode='fixed', seconds=12.):
    """Partition every sample, including silence/tails, without reading labels.

    Energy variant searches +/- 2 s around 12 s for >= 100 ms quiet runs.
    It uses 20 ms RMS and max(0.0008, 0.06 * recording peak RMS).
    Without a quiet candidate it keeps the fixed cut. No chunk is discarded.
    """
    n = len(audio)
    if not n:
        return []
    hop = max(1, int(sr * .02))
    starts = np.arange(0, n, hop)
    square = np.asarray(audio, dtype=np.float64) ** 2
    sums = np.add.reduceat(square, starts)
    rms = np.sqrt(sums / np.minimum(hop, n - starts))
    quiet = rms < max(.0008, .06 * float(rms.max()))
    runs = []
    begin = None
    for i, q in enumerate(list(quiet) + [False]):
        if q and begin is None:
            begin = i
        if not q and begin is not None:
            if i - begin >= 5:
                runs.append((begin * hop + min(n, i * hop)) // 2)
            begin = None
    result, left = [], 0
    while left < n:
        right = min(n, left + int(seconds * sr))
        if mode == 'energy' and right < n:
            candidates = [t for t in runs if abs(t - right) <= 2 * sr
                          and t - left >= 6 * sr and n - t >= .3 * sr]
            if candidates:
                right = min(candidates, key=lambda t: (abs(t - right), t))
        elif mode != 'fixed':
            if mode != 'energy':
                raise ValueError('unknown window mode')
        result.append((left, right))
        left = right
    return result


def check_intervals(intervals, duration, vocabulary, reference=False):
    kinds, issues = {}, {}
    previous = -math.inf
    unknown_seconds = 0.
    for a, b, label in intervals:
        kind = label_kind(label, vocabulary, reference)
        kinds[kind] = kinds.get(kind, 0) + 1
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            issues['non_numeric'] = issues.get('non_numeric', 0) + 1
            continue
        for name, bad in [('non_finite', not math.isfinite(a + b)),
                          ('nonpositive', b <= a), ('out_of_order', a < previous - .002),
                          ('negative_start', a < -.002), ('past_audio_10ms', b > duration + .01)]:
            if bad:
                issues[name] = issues.get(name, 0) + 1
        previous = max(previous, b)
        if kind == 'unknown':
            unknown_seconds += max(0., min(duration, b) - max(0., a))
    return dict(kinds=kinds, issues=issues, unknown_seconds=unknown_seconds)


def window_annotation(intervals, left, right, vocabulary, reference=False):
    """Describe a pre-existing window. Never change its boundaries or labels."""
    tokens, counts, crossing = [], {}, 0
    for a, b, value in intervals:
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            counts['invalid_time'] = counts.get('invalid_time', 0) + 1
            continue
        if not math.isfinite(a + b) or b <= a:
            counts['invalid_time'] = counts.get('invalid_time', 0) + 1
            continue
        if b <= left or a >= right:
            continue
        kind = label_kind(value, vocabulary, reference)
        counts[kind] = counts.get(kind, 0) + 1
        # 2 ms accounts for the original LAB/TAG time rounding, not a crop search.
        if kind != 'empty' and (a < left - .002 or b > right + .002):
            crossing += 1
        if kind == 'phones' and a >= left - .002 and b <= right + .002:
            tokens.extend(c for c in str(value) if c in vocabulary)
    return dict(tokens=tokens, interval_counts=counts, boundary_crossings=crossing,
                complete=not crossing and not counts.get('unknown') and not counts.get('invalid_time'))

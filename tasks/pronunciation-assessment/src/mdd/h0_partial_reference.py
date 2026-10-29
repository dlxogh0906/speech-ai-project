"""Conservative sequence evaluation with explicitly unknown annotation spans.

Unknown spans match zero or more tokens. All minimum-cost alignments are retained;
this is an annotation-sensitivity analysis, not inferred or repaired ground truth.
"""
import math

UNKNOWN = '<UNKNOWN_SPAN>'
UNOBSERVED = '<UNOBSERVED_ACTUAL>'
DELETED = '<DEL>'


def partial_tokens(intervals, vocabulary, reference=False):
    tokens, owners, ignored = [], [], []
    for index, (start, end, label) in enumerate(intervals):
        s = str(label).strip()
        if not s or (s == '-' and not reference):
            ignored.append(dict(interval=index, label=s,
                                reason='explicit_deletion' if s else 'empty'))
            continue
        known = all(c in vocabulary or c.isspace() for c in s)
        emitted = [c for c in s if c in vocabulary] if known else [UNKNOWN]
        for token in emitted:
            tokens.append(token)
            owners.append(dict(interval=index, start=start, end=end, raw_label=s))
    return tokens, owners, ignored


def partial_mapping(reference, target):
    """Edit distance and possible corresponding token values at every ref site.

    A target wildcard can consume a known reference token, marking it unobserved.
    A reference wildcard absorbs arbitrary target content without declaring a
    canonical phone. Adjacent known phones are not automatically declared known.
    Inserted target phones remain in the distance but insertion-gap diagnosis is
    deliberately not produced by this site-only function.
    """
    n, m = len(reference), len(target)
    limit = n + m + 1
    f = [[limit] * (m + 1) for _ in range(n + 1)]
    f[0][0] = 0
    for i in range(n + 1):
        row = f[i]; rw = i < n and reference[i] == UNKNOWN
        below = f[i + 1] if i < n else None
        for j in range(m + 1):
            value = row[j]; tw = j < m and target[j] == UNKNOWN
            if i < n:
                cost = 0 if rw or tw else 1
                below[j] = min(below[j], value + cost)
            if j < m:
                cost = 0 if rw or tw else 1
                row[j + 1] = min(row[j + 1], value + cost)
            if i < n and j < m and not rw and not tw:
                cost = int(reference[i] != target[j])
                below[j + 1] = min(below[j + 1], value + cost)
    b = [[limit] * (m + 1) for _ in range(n + 1)]
    b[n][m] = 0
    for i in range(n, -1, -1):
        row = b[i]; rw = i < n and reference[i] == UNKNOWN
        below = b[i + 1] if i < n else None
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            tw = j < m and target[j] == UNKNOWN
            cost = 0 if rw or tw else 1
            best = below[j] + cost if i < n else limit
            if j < m:
                best = min(best, row[j + 1] + cost)
            if i < n and j < m and not rw and not tw:
                best = min(best, below[j + 1] + int(reference[i] != target[j]))
            row[j] = best
    distance = f[n][m]; options = []
    for i, token in enumerate(reference):
        if token == UNKNOWN:
            options.append(None)
            continue
        values = set()
        for j in range(m + 1):
            tw = j < m and target[j] == UNKNOWN
            cost = 0 if tw else 1
            if f[i][j] + cost + b[i + 1][j] == distance:
                values.add(UNOBSERVED if tw else DELETED)
            if j < m and not tw:
                cost = int(token != target[j])
                if f[i][j] + cost + b[i + 1][j + 1] == distance:
                    values.add(target[j])
        if not values:
            raise AssertionError('reference site lost from optimal paths')
        options.append(frozenset(values))
    return distance, options


def classify_truth(expected, values):
    if expected == UNKNOWN:
        return 'unknown_reference', None
    if UNOBSERVED in values:
        return 'unknown_actual_possible', None
    if len(values) != 1:
        return 'alignment_ambiguous', None
    actual = next(iter(values))
    if actual == expected:
        return 'normal', actual
    return ('deletion' if actual == DELETED else 'substitution'), actual


def known_site_bounds(reference, truth_options, prediction_options):
    """Conditional greedy diagnosis bounds; never calibrated @ FPR<=1%.

    Truth denominators depend only on the fixed annotations. Prediction ambiguity
    gives separate lower/upper counts, retaining the full identified denominator.
    Bounds sum per-site possibilities and need not be attainable by one joint path.
    """
    if not (len(reference) == len(truth_options) == len(prediction_options)):
        raise ValueError('unpaired site arrays')
    result = dict(normal=0, errors=0, unidentifiable_entries=0,
                  normal_false_lower=0, normal_false_upper=0,
                  exact_lower=0, exact_upper=0, prediction_ambiguous=0)
    for expected, truth, pred in zip(reference, truth_options, prediction_options):
        kind, actual = classify_truth(expected, truth)
        if kind not in ('normal', 'substitution', 'deletion'):
            result['unidentifiable_entries'] += 1
            continue
        if not pred or UNOBSERVED in pred:
            raise ValueError('model predictions must be observed tokens, not wildcards')
        result['prediction_ambiguous'] += int(len(pred) > 1)
        if kind == 'normal':
            result['normal'] += 1
            result['normal_false_lower'] += int(expected not in pred)
            result['normal_false_upper'] += int(any(p != expected for p in pred))
        else:
            result['errors'] += 1
            result['exact_lower'] += int(pred == frozenset([actual]))
            result['exact_upper'] += int(actual in pred)
    return result


def input_windows(samples, sample_rate=16000, core_seconds=12., context_seconds=.4):
    """Audio length only; each sample belongs to one core, halos may overlap."""
    if samples < 0 or sample_rate <= 0 or core_seconds <= 0 or context_seconds < 0:
        raise ValueError('invalid window geometry')
    core = int(core_seconds * sample_rate); halo = int(context_seconds * sample_rate)
    if core <= 0:
        raise ValueError('zero core length')
    return [dict(core_start=start, core_end=min(samples, start + core),
                 read_start=max(0, start - halo), read_end=min(samples, start + core + halo))
            for start in range(0, samples, core)]


def owned_ctc_frames(labels, window, hop=320, center=280):
    """w2v-BERT frontend centers: 20 ms * i + 17.5 ms, in 16k samples."""
    return [(window['read_start'] + center + hop * i, label)
            for i, label in enumerate(labels)
            if window['core_start'] <= window['read_start'] + center + hop * i < window['core_end']]


def collapse_owned_frames(frames, blank):
    """Collapse only after concatenating owned frames across chunk boundaries."""
    result = []; last_label = blank; last_time = -math.inf
    for time, label in frames:
        if time <= last_time:
            raise ValueError('duplicate or unordered global frame centers')
        if label != last_label and label != blank:
            result.append(label)
        last_label, last_time = label, time
    return result

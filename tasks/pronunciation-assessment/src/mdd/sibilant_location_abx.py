"""Privileged TAG location diagnosis; no component is a service localizer."""
import numpy as np


def crop_indices(times, consonant, vowel, reference):
    times = np.asarray(times, np.float64)
    empty = np.empty(0, np.int64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not (np.diff(times) > 0).all() or not np.isfinite(reference)):
        raise ValueError('original monotone native centers required')
    if consonant is None or vowel is None:
        return empty, empty, 'missing_interval'
    c, v = consonant, vowel
    if not (np.isfinite(c[:2]).all() and np.isfinite(v[:2]).all()
            and c[0] < c[1] <= v[0] < v[1]):
        return empty, empty, 'invalid_C_V_relation'
    end = min(v[1], v[0]+.12)
    if not ((times >= c[0]) & (times < c[1])).any():
        return empty, empty, 'no_observed_C_frame'
    if not ((times >= v[0]) & (times < end)).any():
        return empty, empty, 'no_observed_initialV_frame'
    tag = np.flatnonzero((times >= c[0]) & (times < end))
    n = len(tag)
    stop = int(np.argmin(abs(times-(reference+.12))))+1
    start = max(0, min(stop-n, len(times)-n))
    control = np.arange(start, start+n)
    return tag, control, 'available'


def retained_abx(distance, left, right, valid_left, valid_right):
    """Unavailable triplets contribute chance; keep all original denominators."""
    distance, left, right = np.asarray(distance), np.asarray(left), np.asarray(right)
    valid_left, valid_right = np.asarray(valid_left, bool), np.asarray(valid_right, bool)
    if (distance.shape != (len(left), len(right)) or valid_left.shape != left.shape
            or valid_right.shape != right.shape
            or any(set(z.tolist()) != {0, 1} for z in (left, right))
            or not np.isfinite(distance).all()):
        raise ValueError('original two-class cross-half populations required')
    half_results, half_valid, coverage = [], [], []
    for d, query, ref, qvalid, rvalid in [(distance, left, right, valid_left, valid_right),
            (distance.T, right, left, valid_right, valid_left)]:
        classes, classes_valid, classes_coverage = [], [], []
        for cls in (0, 1):
            query_values, valid_values, query_coverage = [], [], []
            for i in np.flatnonzero(query == cls):
                same, other = ref == cls, ref != cls
                delta = d[i, same][:, None]-d[i, other][None, :]
                usable = qvalid[i] & rvalid[same][:, None] & rvalid[other][None, :]
                correct = (delta < 0)+.5*(delta == 0)
                query_values.append(float(np.where(usable, correct, .5).mean()))
                query_coverage.append(float(usable.mean()))
                if usable.any():
                    valid_values.append(float(correct[usable].mean()))
            classes.append(float(np.mean(query_values)))
            classes_coverage.append(float(np.mean(query_coverage)))
            if valid_values:
                classes_valid.append(float(np.mean(valid_values)))
        half_results.append(float(np.mean(classes)))
        coverage.append(float(np.mean(classes_coverage)))
        if classes_valid:
            half_valid.append(float(np.mean(classes_valid)))
    return {'retained_accuracy': float(np.mean(half_results)),
            'valid_only_accuracy': float(np.mean(half_valid)) if half_valid else None,
            'triplet_coverage': float(np.mean(coverage))}


def location_result(scores, vowels, availability, seed):
    # Columns: TAG, equal-length runtime location, original complete ROI.
    scores, vowels = np.asarray(scores), np.asarray(vowels)
    rng = np.random.RandomState(seed)
    sample = rng.randint(0, len(scores), (2000, len(scores)))
    diff = scores[:, :1]-scores[:, 1:]
    bootstrap = np.percentile(diff[sample].mean(1), [2.5, 50, 97.5], axis=0)
    groups = {}
    for group in ('all', 'ㅣ', 'ㅡ', 'ㅜ'):
        mask = np.ones(len(scores), bool) if group == 'all' else vowels == group
        groups[group] = {'cells': int(mask.sum()), 'accuracy': scores[mask].mean(0).tolist() if mask.any() else None}
    all_scores, i_scores = np.asarray(groups['all']['accuracy']), np.asarray(groups['ㅣ']['accuracy'])
    passed = (groups['ㅣ']['cells'] > 0 and all_scores[0] >= max(all_scores[1:])+.05
              and i_scores[0] >= max(i_scores[1:])+.03
              and (bootstrap[0] > 0).all()
              and availability['all'] >= .9 and availability['ㅣ'] >= .9)
    return {'columns': ['TAG_location', 'equal_length_runtime_location', 'original_full_ROI'],
        'groups': groups, 'paired_cell_bootstrap_p025_p50_p975': bootstrap.tolist(),
        'bootstrap_scope': 'context descriptive reference; shared speakers and TRAIN exposure remain',
        'proceed_automatic_localization_candidate': bool(passed)}

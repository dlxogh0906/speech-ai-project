"""Untrained TRAIN diagnostics for across-speaker observed frame trajectories.

Context and actual labels are privileged diagnostic matching, never runtime
features. DTW measures representational similarity, not physical boundaries.
"""
import hashlib
import numpy as np

METHODS = ('ordered', 'permuted', 'mean')


def unit_rows(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('nonempty finite observed frames required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= 1e-12).any():
        raise ValueError('zero norm is measurement failure, not an omitted row')
    return x/norm


def permutation(identity, size, seed):
    value = 'trajectory-permutation-v1:'+str(seed)+':'+identity
    state = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)
    return np.random.RandomState(state).permutation(size)


def angular_matrix(a, b):
    return np.arccos(np.clip(a@b.T, -1., 1.))/np.pi


def dtw(cost):
    """Symmetric steps with total path weight n+m, no tunable time band."""
    cost = np.asarray(cost, np.float64)
    if cost.ndim != 2 or not cost.size or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError('nonnegative finite frame costs required')
    n, m = cost.shape
    previous = np.full(m+1, np.inf)
    previous[0] = 0.
    for i in range(n):
        current = np.full(m+1, np.inf)
        for j in range(m):
            d = cost[i, j]
            current[j+1] = min(previous[j]+2*d, previous[j+1]+d, current[j]+d)
        previous = current
    return float(previous[m]/(n+m))


def class_balanced_abx(distances, actual_left, actual_right):
    """All opposite-half reference pairs; macro class then query half."""
    distances = np.asarray(distances)
    actual_left, actual_right = np.asarray(actual_left), np.asarray(actual_right)
    if (distances.shape != (len(actual_left), len(actual_right))
            or not np.isfinite(distances).all()
            or any(set(z.tolist()) != {0, 1} for z in (actual_left, actual_right))):
        raise ValueError('two actual classes in both speaker halves required')
    halves = []
    for matrix, query, reference in [(distances, actual_left, actual_right),
                                      (distances.T, actual_right, actual_left)]:
        scores = []
        for i, cls in enumerate(query):
            same = matrix[i, reference == cls]
            other = matrix[i, reference != cls]
            difference = same[:, None]-other[None, :]
            scores.append(float(((difference < 0)+.5*(difference == 0)).mean()))
        scores = np.asarray(scores)
        halves.append([float(scores[query == cls].mean()) for cls in (0, 1)])
    return float(np.mean(halves)), halves


def summarize(scores, vowels, seed, bootstraps=2000):
    scores, vowels = np.asarray(scores), np.asarray(vowels)
    if scores.ndim != 2 or scores.shape[1] != 3 or len(vowels) != len(scores) or not len(scores):
        raise ValueError('paired context scores required')
    rng = np.random.RandomState(seed)
    sample = rng.randint(0, len(scores), size=(bootstraps, len(scores)))
    differences = scores[:, :1]-scores[:, 1:]
    reference = np.percentile(differences[sample].mean(1), [2.5, 50, 97.5], axis=0)
    groups = {}
    for name in ('all', 'ㅣ', 'ㅡ', 'ㅜ'):
        mask = np.ones(len(scores), bool) if name == 'all' else vowels == name
        groups[name] = {'cells': int(mask.sum()), 'accuracy':
            dict(zip(METHODS, scores[mask].mean(0).tolist())) if mask.any() else None}
    means = scores.mean(0)
    i = groups['ㅣ']
    passed = (means[0] >= max(means[1:])+.02 and (reference[0] > 0).all()
              and i['cells'] >= 10 and i['accuracy']['ordered'] >= max(
                  i['accuracy'][k] for k in METHODS[1:]))
    return {'groups': groups, 'ordered_minus_permuted_mean': float(means[0]-means[1]),
        'ordered_minus_mean_mean': float(means[0]-means[2]),
        'paired_cell_bootstrap_p025_p50_p975': reference.tolist(),
        'bootstrap_scope': 'descriptive context resampling, shared speakers and TRAIN exposure remain',
        'proceed_trajectory_detector': bool(passed)}

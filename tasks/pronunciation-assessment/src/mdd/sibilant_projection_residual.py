"""TRAIN-only split-speaker diagnostics for discarded native PCA information.

No trained classifier, no revised projection, and no physical onset labels.
Conditional permutations are a diagnostic reference, not population p-values.
"""
import collections
import hashlib
import numpy as np
import torch
from mdd.sibilant_frozen_probe import project_observed

SCHEMA = 'native_standardized_rolecontrast3_split_speaker_PCA_residual_v1'
ROLES = ('C_mean', 'earlyV_minus_C', 'lateV_minus_earlyV')


def speaker_half(speaker, seed):
    value = 'projection-halves-v1:'+str(seed)+':'+speaker
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % 2


def cell_identity(feature, label):
    return (label['item'], label['q'], label['expected'], feature.get('observed_V'),
            label['position'], label.get('language'))


def select_cells(features, labels, y, eligible, seed):
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    flow = collections.Counter(inputs=len(y), original_unknown=int((y < 0).sum()))
    for i, feature in enumerate(features):
        if y[i] < 0:
            continue
        if y[i] not in (0, 1):
            flow['known_other_actual'] += 1
            continue
        if not eligible[i]:
            flow['original_CTC_ineligible'] += 1
            continue
        label = labels[i]
        if label['expected'] != feature['canonical']:
            flow['runtime_canonical_disagrees'] += 1
            continue
        if label['actual'] != ('ㅅ', 'ㅆ')[int(y[i])]:
            raise ValueError('original actual label differs')
        grouped[cell_identity(feature, label)][label['speaker']].append(i)
        flow['eligible_S_SS_rows'] += 1
    cells, ledger = [], []
    for key, speakers in sorted(grouped.items(), key=lambda z: str(z[0])):
        chosen = []
        for speaker, indices in sorted(speakers.items()):
            def priority(i):
                identity = 'projection-dedup-v1:'+features[i]['uid']+'/'+str(features[i]['q_runtime'])
                return hashlib.sha256(identity.encode()).digest()
            chosen.append(min(indices, key=priority))
            flow['same_speaker_context_repeat_removed'] += len(indices)-1
        halves = [[i for i in chosen if speaker_half(labels[i]['speaker'], seed) == half] for half in (0, 1)]
        class_counts = [[int(sum(y[i] == cls for i in ix)) for cls in (0, 1)] for ix in halves]
        valid = all(all(n > 0 for n in z) for z in class_counts)
        item = {'cell': list(key), 'indices': chosen, 'halves': halves,
                'class_counts_S_SS': class_counts, 'both_actuals_in_both_halves': valid}
        ledger.append(item)
        if valid:
            cells.append(item)
            flow['used_diagnostic_rows'] += len(chosen)
        else:
            flow['insufficient_class_by_speaker_half_cells'] += 1
    flow.update(all_context_cells=len(grouped), eligible_split_cells=len(cells))
    return cells, ledger, dict(flow)


def full_observations(base, hidden, mask, relative):
    """Same automatic role weights; only retain original unprojected coordinates."""
    if hidden.shape[:2] != mask.shape or relative.shape != mask.shape or not bool(mask.any(1).all()):
        raise ValueError('observed native frame grid required')
    projection = {k: getattr(base, k) for k in ('mean', 'scale', 'components')}
    h = project_observed(hidden, mask, projection)
    phase = base.head.support(torch.cat([h.float(), relative[..., None]], -1))
    weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2)*mask[:, None, :]
    weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
    z = torch.zeros_like(hidden, dtype=torch.float32)
    z[mask] = (hidden[mask].float()-base.mean)/base.scale
    means = torch.bmm(weights, z)
    full = torch.stack([means[:, 0], means[:, 1]-means[:, 0], means[:, 2]-means[:, 1]], 1)
    return full, phase


def signed_weights(actual):
    actual = np.asarray(actual)
    if actual.ndim != 1 or not ((actual == 0) | (actual == 1)).all() or not (actual == 0).any() or not (actual == 1).any():
        raise ValueError('both actual S/SS identities required')
    return np.where(actual == 1, 1./(actual == 1).sum(), -1./(actual == 0).sum())


def replication(cells, full, components, y, permutations, seed):
    """Cell-specific cross-half differences, with original class counts in nulls."""
    q = np.asarray(components, np.float64)
    if full.ndim != 3 or full.shape[1] != 3 or full.shape[2] != len(q):
        raise ValueError('full standardized native role contrasts required')
    np.testing.assert_allclose(q.T@q, np.eye(q.shape[1]), atol=1e-5, rtol=1e-5)
    rng = np.random.RandomState(seed)
    names = ('all', 'ㅣ', 'ㅡ', 'ㅜ')
    real = {name: np.zeros((3, 3)) for name in names}
    null = {name: np.zeros((permutations, 3, 3)) for name in names}
    counts = collections.Counter()
    deltas = []
    direct_difference = 0.
    for cell in cells:
        actual_delta, random_delta, projected_delta, random_projected = [], [], [], []
        for ix in cell['halves']:
            observed = np.asarray(full[ix], np.float64)
            kept = observed@q
            labels = y[ix]
            weights = signed_weights(labels)
            actual_delta.append(np.einsum('n,nrd->rd', weights, observed))
            projected_delta.append(np.einsum('n,nrd->rd', weights, kept))
            shuffled = np.asarray([signed_weights(rng.permutation(labels)) for _ in range(permutations)])
            random_delta.append((shuffled@observed.reshape(len(ix), -1)).reshape(permutations, 3, -1))
            random_projected.append((shuffled@kept.reshape(len(ix), -1)).reshape(permutations, 3, -1))
        a, b = actual_delta
        ap, bp = projected_delta
        dot = np.einsum('rd,rd->r', a, b)
        kept_dot = np.einsum('rd,rd->r', ap, bp)
        direct = np.einsum('rd,rd->r', a-ap@q.T, b-bp@q.T)
        np.testing.assert_allclose(dot-kept_dot, direct, atol=1e-4, rtol=2e-5)
        direct_difference = max(direct_difference, float(abs(dot-kept_dot-direct).max()))
        values = np.stack([dot, kept_dot, dot-kept_dot], -1)
        random_dot = np.einsum('prd,prd->pr', random_delta[0], random_delta[1])
        random_kept = np.einsum('prd,prd->pr', random_projected[0], random_projected[1])
        random_values = np.stack([random_dot, random_kept, random_dot-random_kept], -1)
        for name in ('all', str(cell['cell'][3])):
            if name not in real:
                continue
            real[name] += values
            null[name] += random_values
            counts[name] += 1
        deltas.append(np.stack(actual_delta))
    report = {}
    for name in names:
        count = counts[name]
        if not count:
            report[name] = {'cells': 0, 'roles': None}
            continue
        real[name] /= count
        null[name] /= count
        roles = {}
        for ri, role in enumerate(ROLES):
            roles[role] = {space: {'cross_half_mean': float(real[name][ri, si]),
                'conditional_null_p025_p50_p975': np.percentile(null[name][:, ri, si], [2.5, 50, 97.5]).tolist(),
                'scope': 'TRAIN diagnostic reference, not population pvalue'}
                for si, space in enumerate(('full', 'kept', 'discarded'))}
        report[name] = {'cells': count, 'roles': roles}
    primary = []
    if counts['ㅣ'] >= 10 and counts['all']:
        for role in ROLES[:2]:
            all_value = report['all']['roles'][role]['discarded']
            i_value = report['ㅣ']['roles'][role]['discarded']['cross_half_mean']
            if (all_value['cross_half_mean'] > max(0, all_value['conditional_null_p025_p50_p975'][-1])
                    and i_value > 0):
                primary.append(role)
    return {'groups': report, 'primary_supported_blocks': primary, 'proceed_residual_candidate': bool(primary),
        'direct_residual_identity_max_difference': direct_difference}, null, np.asarray(deltas)

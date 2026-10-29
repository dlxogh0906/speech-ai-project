"""TRAIN-only cross-speaker actual-S/SS comparisons, with exact exposure control.

TAG identities and intended slots select supervision only. None is a readout
input. Shuffled links retain identical per-batch endpoints and actual classes.
"""
import collections
import itertools
import numpy as np
import torch

SCHEMA = 'TRAIN_same_slot_vowel_position_L1_crossspeaker_pairlinks_v1'


def context(feature, label, language=True):
    value = (label['item'], label['q'], label['expected'], feature.get('observed_V'), label['position'])
    return value+(label.get('language'),) if language else value


def make_plan(original, y, eligible, features, labels, seed):
    """Keep original per-batch class counts/rare-site exposures; replace S/SS.

    Each batch compares substitutions of one intended consonant, alternating
    directions. Every paired normal has the same full context, another speaker;
    comparisons inside one batch have different coarse phonetic contexts.
    """
    original, y, eligible = np.asarray(original), np.asarray(y), np.asarray(eligible)
    if original.ndim != 2 or original.shape[1] != 16 or eligible.shape != y.shape:
        raise ValueError('fixed16-slot original plan and eligibility required')
    rng = np.random.RandomState(seed)
    normals = collections.defaultdict(list)
    errors = {'ㅅ': [], 'ㅆ': []}
    for i, feature in enumerate(features):
        if not eligible[i] or y[i] not in (0, 1):
            continue
        label = labels[i]
        if label['expected'] != feature['canonical']:
            continue
        actual = ('ㅅ', 'ㅆ')[int(y[i])]
        if label['actual'] != actual:
            raise ValueError('known actual class disagrees with original label')
        if label['label_kind'] == 'correct':
            if actual != label['expected']:
                raise ValueError('normal label is not the expected actual')
            normals[context(feature, label)].append(i)
        elif label['label_kind'] == 'pair_error':
            if actual == label['expected']:
                raise ValueError('pair error is not an actual contrast')
            errors[label['expected']].append(i)
    pools = {}
    for expected, indices in errors.items():
        pools[expected] = [i for i in indices if len({labels[j]['speaker']
            for j in normals[context(features[i], labels[i])]
            if labels[j]['speaker'] != labels[i]['speaker']}) >= 2]
        if not pools[expected]:
            raise ValueError('missing cross-speaker context-matched error pool')
        rng.shuffle(pools[expected])
    plan = np.where(eligible[original], original, -1).astype(np.int64)
    matched = np.full((len(plan), 4, 2), -1, np.int64)
    shuffled = matched.copy()
    cursors, normal_use, records = {'ㅅ': 0, 'ㅆ': 0}, collections.Counter(), []
    for step, row in enumerate(plan):
        slots = {cls: [s for s, i in enumerate(row) if i >= 0 and y[i] == cls] for cls in (0, 1)}
        n = min(map(len, slots.values()))
        if n not in (3, 4):
            raise ValueError('original plan must support3or4 comparisons per batch')
        expected = ('ㅅ', 'ㅆ')[step % 2]
        cells, uids, examples = set(), set(), []
        for pair in range(n):
            found = None
            for _ in range(len(pools[expected])*2):
                i = pools[expected][cursors[expected] % len(pools[expected])]
                cursors[expected] += 1
                coarse = context(features[i], labels[i], False)
                if coarse not in cells and labels[i]['uid'] not in uids:
                    found = i
                    break
            if found is None:
                raise ValueError('insufficient different phonetic contexts in a batch')
            i = found
            candidates = [j for j in normals[context(features[i], labels[i])]
                          if labels[j]['speaker'] != labels[i]['speaker']]
            minimum = min(normal_use[j] for j in candidates)
            candidates = sorted(j for j in candidates if normal_use[j] == minimum)
            j = candidates[int(rng.randint(len(candidates)))]
            normal_use[j] += 1
            actual = {int(y[i]): i, int(y[j]): j}
            ss, s = slots[1][pair], slots[0][pair]
            row[ss], row[s] = actual[1], actual[0]
            matched[step, pair] = (ss, s)
            cells.add(context(features[i], labels[i], False))
            uids.add(labels[i]['uid'])
            examples.append({'error': i, 'normal': j, 'cell': list(context(features[i], labels[i]))})
        # All alternatives keep each SS and S endpoint once; do not shuffle audio.
        permutations = list(itertools.permutations(range(n)))
        rng.shuffle(permutations)
        permutation = None
        for perm in permutations:
            if any(k == v for k, v in enumerate(perm)):
                continue
            pairs = []
            for k in range(n):
                ss, s = matched[step, k]
                if expected == 'ㅅ':
                    s = matched[step, perm[k], 1]
                else:
                    ss = matched[step, perm[k], 0]
                a, b = int(row[ss]), int(row[s])
                if (labels[a]['speaker'] == labels[b]['speaker']
                        or context(features[a], labels[a], False) == context(features[b], labels[b], False)):
                    break
                pairs.append((ss, s))
            if len(pairs) == n:
                permutation = pairs
                break
        if permutation is None:
            raise ValueError('no label-correct different-context cross-speaker link control')
        shuffled[step, :n] = permutation
        before = collections.Counter(int(y[i]) for i in original[step] if eligible[i])
        after = collections.Counter(int(y[i]) for i in row if i >= 0)
        if before != after:
            raise ValueError('per-batch actual-class mass changed')
        records.append({'step': step, 'expected': expected, 'pairs': examples})
    return plan, matched, shuffled, records


def contrast_loss(logits, pairs, margin=1.):
    """Actual-SS logit contrast must exceed actual-S on another real recording."""
    links = torch.as_tensor(pairs, device=logits.device, dtype=torch.long)
    if links.ndim != 2 or links.shape[1] != 2 or not len(links):
        raise ValueError('nonempty SS/S links required')
    if bool((links < 0).any()) or bool((links >= len(logits)).any()):
        raise ValueError('comparison endpoint outside observed batch')
    contrast = logits[:, 1]-logits[:, 0]
    difference = contrast[links[:, 0]]-contrast[links[:, 1]]
    return torch.nn.functional.softplus(float(margin)-difference).mean()

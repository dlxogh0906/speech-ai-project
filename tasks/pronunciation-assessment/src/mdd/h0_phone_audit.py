"""Canonical-site error audit, context standardization and sequence diagnostics.

Surface positions are inferred from the project's flat jamo representation.
Spelling changes are structural candidates, not adjudicated phonological rules.
"""
import collections
import functools
import math
import numpy as np

VOWELS = set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
COMPOUND = set('ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ')
CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
JONG = ' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'
DELETION = '<DEL>'


def token(value):
    return DELETION if value is None else value


def surface_position(seq, i):
    t = seq[i]
    if t in VOWELS:
        return 'nucleus'
    if t == 'ㅇ' or t in COMPOUND:
        return 'coda'
    if i + 1 < len(seq) and seq[i + 1] in VOWELS:
        return 'onset'
    if t in 'ㄸㅃㅉ':
        return 'onset_fallback'
    return 'coda'


@functools.lru_cache(maxsize=50000)
def optimal_mapping(ref, target):
    """All possible target indices at each ref site over minimum edit paths."""
    if ref == target:
        return tuple(frozenset([i]) for i in range(len(ref)))
    n, m = len(ref), len(target)
    f = [[0] * (m + 1) for _ in range(n + 1)]
    b = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): f[i][0] = i; b[i][m] = n - i
    for j in range(m + 1): f[0][j] = j; b[n][j] = m - j
    for i in range(n):
        for j in range(m):
            f[i + 1][j + 1] = min(f[i][j + 1] + 1, f[i + 1][j] + 1,
                                  f[i][j] + (ref[i] != target[j]))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            b[i][j] = min(b[i + 1][j] + 1, b[i][j + 1] + 1,
                          b[i + 1][j + 1] + (ref[i] != target[j]))
    opts = []
    for i in range(n):
        choices = set()
        for j in range(m + 1):
            if f[i][j] + 1 + b[i + 1][j] == f[n][m]: choices.add(None)
            if j < m and f[i][j] + (ref[i] != target[j]) + b[i + 1][j + 1] == f[n][m]:
                choices.add(j)
        opts.append(frozenset(choices))
    return tuple(opts)


def project_gaps(ref, target):
    """Same single edlib path as the historical evaluator, retaining gap tokens."""
    import edlib
    if not target:
        return [None] * len(ref), [[] for _ in range(len(ref) + 1)]
    if not ref:
        return [], [list(target)]
    a, b = ''.join(ref), ''.join(target)
    alignment = edlib.getNiceAlignment(edlib.align(a, b, task='path'), a, b)
    sites, gaps = [], [[]]
    for c, t in zip(alignment['query_aligned'], alignment['target_aligned']):
        if c == '-':
            gaps[-1].append(t)
        else:
            sites.append(None if t == '-' else t); gaps.append([])
    assert len(sites) == len(ref)
    return sites, gaps


def spelling_tokens(prompt):
    out = []; syllables = [c for c in prompt if '가' <= c <= '힣']
    for owner, c in enumerate(syllables):
        n = ord(c) - 0xAC00; onset, vowel, coda = CHO[n // 588], JUNG[n // 28 % 21], JONG[n % 28]
        if onset != 'ㅇ': out.append((onset, 'onset', owner))
        out.append((vowel, 'nucleus', owner))
        if coda != ' ': out.append((coda, 'coda', owner))
    return out, syllables


@functools.lru_cache(maxsize=4096)
def contexts(canon, prompt):
    spelling, syllables = spelling_tokens(prompt)
    mapping = optimal_mapping(canon, tuple(t[0] for t in spelling))
    positions = [surface_position(canon, i) for i in range(len(canon))]
    result = []
    for i, expected in enumerate(canon):
        position = positions[i]
        following = next((t for t in canon[i + 1:] if t in VOWELS), '<EOS>')
        prev = canon[i - 1] if i else '<BOS>'
        prev_pos = positions[i - 1] if i else '<BOS>'
        rules = []
        opts = mapping[i]
        if len(opts) != 1 or None in opts:
            written, written_pos = '<UNRESOLVED>', '<UNRESOLVED>'
            rules.append('spelling_mapping_unresolved')
        else:
            written, written_pos, owner = spelling[next(iter(opts))]
            if {'ㄱ':'ㄲ', 'ㄷ':'ㄸ', 'ㅂ':'ㅃ', 'ㅅ':'ㅆ', 'ㅈ':'ㅉ'}.get(written) == expected:
                rules.append('tensification_candidate')
            if written in 'ㄱㄲㅋㄷㅅㅆㅈㅊㅌㅎㅂㅍ' and expected in 'ㄴㅁㅇ':
                rules.append('nasalization_candidate')
            if written == 'ㄴ' and expected == 'ㄹ': rules.append('liquidization_candidate')
            if written_pos == 'coda' and position == 'onset' and owner + 1 < len(syllables):
                next_onset = CHO[(ord(syllables[owner + 1]) - 0xAC00) // 588]
                if next_onset == 'ㅇ': rules.append('liaison_candidate')
            if written_pos == 'coda' and position == 'coda' and written != expected and expected in 'ㄱㄷㅂ':
                rules.append('coda_neutralization_candidate')
            if not rules:
                rules.append('spelling_unchanged' if written == expected else 'other_spelling_change')
        result.append(dict(position=position, following_vowel=following,
            previous_phone=prev, previous_position=prev_pos,
            post_coda_onset=position == 'onset' and prev_pos == 'coda',
            spelling_phone=written, spelling_position=written_pos, rule_candidates=rules))
    return tuple(result)


def events(site, threshold):
    c, t, p, score = (site[k] for k in ('expected', 'actual', 'predicted', 'score'))
    normal = t == c; mismatch = p != c
    warned = mismatch and score is not None and score >= threshold
    exact = not normal and p == t
    wrong = not normal and mismatch and p != t
    return dict(total=1, normal=int(normal), errors=int(not normal),
        actual_substitution=int(not normal and t is not None), actual_deletion=int(t is None),
        normal_fp=int(normal and mismatch), normal_fp_substitution=int(normal and mismatch and p is not None),
        normal_fp_deletion=int(normal and p is None),
        error_revert=int(not normal and p == c), error_exact=int(exact), error_wrong=int(wrong),
        policy_fp=int(normal and warned), policy_exact=int(exact and warned), policy_wrong=int(wrong and warned),
        exact_suppressed=int(exact and not warned), wrong_suppressed=int(wrong and not warned),
        scored=int(score is not None))


def rate(n, d):
    return n / d if d else None


def metric_counts(counts):
    c = dict(counts)
    for key in ('total','normal','errors','normal_fp','error_exact','error_revert','error_wrong',
                'policy_fp','policy_exact','policy_wrong','scored'):
        c.setdefault(key, 0)
    for name, n, d in [('normal_fpr','normal_fp','normal'), ('exact_recall','error_exact','errors'),
                        ('policy_fpr','policy_fp','normal'), ('policy_exact_recall','policy_exact','errors')]:
        c[name] = rate(c.get(n, 0), c.get(d, 0))
    return c


def standardize(context_groups, numerator, denominator):
    """Use common unseen context weights for both sets; do not match outcomes."""
    common = [(k, v['valid'], v['valid_unseen']) for k, v in context_groups.items()
              if v.get('valid', {}).get(denominator, 0) and v.get('valid_unseen', {}).get(denominator, 0)]
    unseen_n = sum(u[denominator] for _, s, u in common)
    seen_n = sum(s[denominator] for _, s, u in common)
    if not common: return dict(common_contexts=0, seen_denominator=0, unseen_denominator=0)
    seen_raw = sum(s.get(numerator, 0) for _, s, u in common) / seen_n
    unseen = sum(u.get(numerator, 0) for _, s, u in common) / unseen_n
    seen_std = sum(u[denominator] * s.get(numerator, 0) / s[denominator] for _, s, u in common) / unseen_n
    return dict(common_contexts=len(common), seen_denominator=seen_n, unseen_denominator=unseen_n,
                seen_raw_common=seen_raw, unseen_raw_common=unseen,
                seen_standardized_to_unseen=seen_std, within_context_gap=unseen-seen_std,
                composition_component=seen_std-seen_raw,
                common_unseen_weights=True, causal_claim=False)


def paired_cluster_ci(item_counts, numerator, denominator, repeats=2000, seed=42):
    """Paired checkpoint difference, resampling intact groups, denominator fixed per site."""
    if len(item_counts) < 2:
        return dict(clusters=len(item_counts), ci95=None)
    vals = np.array([[c.get(numerator, 0), c.get(denominator, 0)] for c in item_counts], dtype=float)
    rng = np.random.RandomState(seed)
    sample = vals[rng.randint(0, len(vals), (repeats, len(vals)))].sum(axis=1)
    rates = sample[sample[:,1] > 0, 0] / sample[sample[:,1] > 0, 1]
    return dict(clusters=len(vals), ci95=np.quantile(rates, [.025, .975]).tolist() if len(rates) else None,
                repeats=repeats, conditional_on_fixed_models_and_thresholds=True)


def sequence_site_audit(canon, actual, prediction):
    """Site-level sensitivity; possible-value overlap is not a corrected label."""
    truth, _ = project_gaps(canon, actual)
    pred, _ = project_gaps(canon, prediction)
    true_opts = optimal_mapping(tuple(canon), tuple(actual))
    pred_opts = optimal_mapping(tuple(canon), tuple(prediction))
    direct, _ = project_gaps(actual, prediction)
    results = []
    for i, expected in enumerate(canon):
        c = collections.Counter(canonical_sites=1)
        tv = {None if j is None else actual[j] for j in true_opts[i]}
        pv = {None if j is None else prediction[j] for j in pred_opts[i]}
        bad = pred[i] != truth[i]
        c['truth_index_ambiguous'] += len(true_opts[i]) > 1
        c['truth_value_ambiguous'] += len(tv) > 1
        c['pred_index_ambiguous'] += len(pred_opts[i]) > 1
        c['pred_value_ambiguous'] += len(pv) > 1
        c['canonical_incorrect'] += bad
        c['incorrect_with_mapping_ambiguity'] += bad and (len(tv) > 1 or len(pv) > 1)
        c['incorrect_invariant_value_across_paths'] += bad and len(tv) == len(pv) == 1
        c['incorrect_with_overlap_in_possible_values'] += bad and bool(tv & pv)
        if len(true_opts[i]) == 1 and None not in true_opts[i]:
            j = next(iter(true_opts[i]))
            c['canonical_wrong_but_direct_TAG_path_correct'] += bad and direct[j] == actual[j]
        if actual == prediction:
            assert not bad, 'identical sequences cannot disagree under deterministic projection'
        if truth[i] == expected and pred[i] != expected:
            c['normal_fp_mapping_sensitive'] += len(tv) > 1 or len(pv) > 1
            c['normal_fp_mapping_invariant'] += len(tv) == len(pv) == 1
        results.append(dict(c))
    return results


def sequence_audit(canon, actual, prediction, site_diagnostics=None):
    _, truth_gaps = project_gaps(canon, actual)
    _, pred_gaps = project_gaps(canon, prediction)
    c = collections.Counter(utterances=1, sequence_exact=int(actual == prediction), gaps=len(canon)+1)
    if site_diagnostics is None:
        site_diagnostics = sequence_site_audit(canon, actual, prediction)
    for diagnostic in site_diagnostics:
        c.update(diagnostic)
    for a, p in zip(truth_gaps, pred_gaps):
        c['true_insertion_gaps'] += bool(a); c['predicted_insertion_gaps'] += bool(p)
        c['normal_gaps'] += not a; c['false_insertion_gaps'] += not a and bool(p)
        c['insertion_detected_gaps'] += bool(a) and bool(p)
        c['insertion_exact_gaps'] += bool(a) and p == a
        c['insertion_wrong_gaps'] += bool(a) and bool(p) and p != a
        c['insertion_missed_gaps'] += bool(a) and not p
        c['true_insertion_tokens'] += len(a); c['predicted_insertion_tokens'] += len(p)
    return dict(c)

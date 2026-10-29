"""Supplement cached diagnosis with spelling context and paired uncertainty."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'), str(Path(__file__).parent)]
import argparse
import collections
import copy
import functools
import hashlib
import json
import re
import numpy as np
from shared.paths import ensure_kog2p, G2P_TOOLS
from eval_sibilant_directional import all_optimal_mapping, phonemes
from mdd.speechsuper import decompose
from mdd.canonical_fix import CanonicalBuilder, _adnominal, LEXICAL_RESPELL
from mdd.sibilant_diagnostic_audit import exact, flagged, summary, grouped_bootstrap
from mdd.sibilant_presence_policy import decision


def read(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@functools.lru_cache(maxsize=4096)
def mapping(a, b):
    return all_optimal_mapping(a, b)


def view_for(record, row, label):
    if record['rejected']:
        return {'state': 'input_rejected', 'phone': None}
    canon = record['raw']['canon'];opts = mapping(tuple(row['canonical']), tuple(canon))[label['q']]
    q = next(iter(opts)) if len(opts) == 1 and None not in opts else None
    if q is None:
        return {'state': 'runtime_mapping_ambiguous', 'phone': None}
    return {'state': 'available' if canon[q] == label['expected'] else 'runtime_canonical_disagrees',
            'phone': phonemes(record['body'])[q]}


def spelling_context(row, label):
    if label['expected'] != 'ㅆ':
        return 'expected_S', None
    spelling = decompose(row['prompt']);opts = mapping(tuple(row['canonical']), tuple(z[2] for z in spelling))[label['q']]
    if len(opts) != 1 or None in opts:
        return 'spelling_mapping_unresolved', None
    i = next(iter(opts));owner, ch, token = spelling[i]
    if token == 'ㅆ':
        return 'written_SS', owner
    if token != 'ㅅ':
        return 'other_spelling_at_SS_reference', owner
    text = ''.join(c for c in row['prompt'] if '가' <= c <= '힣')
    prev = text[owner - 1] if owner else ''
    jong = (ord(prev) - 0xAC00) % 28 if prev else 0
    for match in re.finditer(r'(?<!열)여덟\s+시(?!간|작|청|민)', row['prompt']):
        target = match.end() - 1
        if owner == sum('가' <= c <= '힣' for c in row['prompt'][:target]):
            return 'time_phrase_review_candidate', owner
    words = re.findall('[가-힣]+', row['prompt'])
    consumed = 0;word = '';previous_word = '';word_initial = False
    for k, w in enumerate(words):
        if consumed <= owner < consumed + len(w):
            word = w;previous_word = words[k - 1] if k else ''
            word_initial = owner == consumed
            break
        consumed += len(w)
    if jong == 8 and word_initial and _adnominal(previous_word):
        return 'adnominal_L_candidate_not_morphological_gold', owner
    if jong == 8 and any(w in word for w in LEXICAL_RESPELL):
        return 'lexical_L_candidate_not_etymological_gold', owner
    if jong == 19:
        return 'written_S_coda_saisiot_unadjudicated', owner
    if jong in (1, 2, 3, 7, 9, 11, 13, 14, 17, 18, 20, 22, 23, 24, 25, 26):
        return 'post_obstruent_coda_candidate', owner
    if jong == 8:
        return 'other_L_context_unadjudicated', owner
    return 'other_S_to_SS_reference', owner


def bootstrap_pair(rows, before='prior', after='candidate', seed=20261026):
    arrays = []
    for z in rows:
        e = z['label_kind'] not in ('correct', 'ambiguous');n = z['label_kind'] == 'correct'
        b, a = z[before], z[after]
        arrays.append({'speaker': z['speaker'], 'errors': int(e), 'normal': int(n),
            'normal_known': int(n and b['phone'] is not None),
            'before_exact': int(exact(z, b)), 'after_exact': int(exact(z, a)),
            'delta_exact': int(exact(z, a)) - int(exact(z, b)),
            'before_FP': int(n and flagged(b)), 'after_FP': int(n and flagged(a)),
            'delta_FP': int(n and flagged(a)) - int(n and flagged(b))})
    return grouped_bootstrap(arrays, {'before_recall': ('before_exact', 'errors'),
        'after_recall': ('after_exact', 'errors'), 'paired_recall_change': ('delta_exact', 'errors'),
        'before_FP_known': ('before_FP', 'normal_known'), 'after_FP_known': ('after_FP', 'normal_known'),
        'paired_FP_known_change': ('delta_FP', 'normal_known')}, 2000, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True);ap.add_argument('--audit', required=True);ap.add_argument('--out', required=True)
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());root = Path(a.audit);out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ensure_kog2p();argv = sys.argv;sys.argv = sys.argv[:1]
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV
    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / 'KoG2P/rulebook.txt'))
    @functools.lru_cache(maxsize=2048)
    def plain(text):
        text = re.sub(r'\s+', ' ', re.sub(r'[^0-9A-Za-z가-힣\s]', ' ', text)).strip();result = []
        for j in phones_to_jamo([v for v in graph2prono(text, ri, ro).split() if v in INV]):
            result.extend({'ㅙ': ['ㅗ', 'ㅐ'], 'ㅞ': ['ㅜ', 'ㅔ']}.get(j, [j]))
        return result
    builder = CanonicalBuilder(plain)
    report = {'scope': 'Reference disagreement is not normative error; spelling taxonomy has explicit unresolved categories.',
        'baseline_variants': {}, 'SS_contexts': {}, 'paired_intervals': {}, 'production_changed': False}
    for name in ('D5', 'CAL', 'primary'):
        spec = p['partitions'][name];ledger = list(read(root / (name + '.sites.jsonl')))
        by_uid = collections.defaultdict(list)
        for z in ledger:by_uid[z['uid']].append(z)
        variants = collections.defaultdict(list);contexts = collections.defaultdict(list)
        review = []
        for record in read(spec['responses']):
            row = record['row'];plain_seq = plain(row['prompt']);fixed = builder.fixed(row['prompt'])
            for z in by_uid[row['uid']]:
                kind, owner = spelling_context(row, z)
                zz = dict(z, context=kind, spelling_owner=owner)
                for method, seq in [('plain_KoG2P', plain_seq), ('fixed_KoG2P', fixed)]:
                    opts = mapping(tuple(row['canonical']), tuple(seq))[z['q']]
                    zz[method + '_LAB_agreement'] = bool(len(opts) == 1 and None not in opts and seq[next(iter(opts))] == z['expected'])
                contexts[kind].append(zz)
                for variant in ('original', 'existing', 'prior'):
                    v = view_for(record[variant], row, z)
                    variants[variant].append(dict(z, view=v))
                if name == 'CAL' and ((z['label_kind'] == 'correct' and flagged(z['view']))
                    or (z['label_kind'] not in ('correct', 'ambiguous') and not exact(z, z['view']))):
                    review.append({'uid': z['uid'], 'q': z['q'], 'wav': row['wav'], 'prompt': row['prompt'],
                        'kind': 'false_warning' if z['label_kind'] == 'correct' else 'missed_error',
                        'TAG_expected': z['expected'], 'TAG_actual': z.get('actual'),
                        'audio_listened': False, 'independent_adjudication': None})
        report['baseline_variants'][name] = {}
        for variant, rows in variants.items():
            report['baseline_variants'][name][variant] = {'summary': summary(rows),
                'directions': {t + '→' + alt: summary([z for z in rows if z['expected'] == t and z.get('actual') == alt])
                               for t, alt in [('ㅅ', 'ㅆ'), ('ㅆ', 'ㅅ')]},
                'accepted_error_exact_recall': {'exact': sum(exact(z, z['view']) for z in rows),
                    'accepted_errors': sum(z['label_kind'] not in ('correct', 'ambiguous') and z['view']['state'] != 'input_rejected' for z in rows)}}
        report['SS_contexts'][name] = {k: dict(summary(v), unique_items=len({z['item'] for z in v}),
            plain_KoG2P_LAB_disagreements=sum(not z['plain_KoG2P_LAB_agreement'] for z in v),
            fixed_KoG2P_LAB_disagreements=sum(not z['fixed_KoG2P_LAB_agreement'] for z in v)) for k,v in contexts.items()}
        with (out / (name + '.SS_contexts.jsonl')).open('x') as f:
            for values in contexts.values():
                for z in values:f.write(json.dumps(z,ensure_ascii=False)+'\n')
        if name == 'CAL':
            fp = [z for z in review if z['kind']=='false_warning']
            misses = sorted([z for z in review if z['kind']=='missed_error'], key=lambda z: hashlib.sha256((z['uid']+'/'+str(z['q'])).encode()).hexdigest())[:50]
            write(out/'review_manifest_NOT_LISTENED.json',fp+misses)
        print('context complete', name, flush=True)
    for name, path in p['paired_site_ledgers'].items():
        rows = list(read(path));report['paired_intervals'][name] = bootstrap_pair(rows)
    # Latest native model has no feasible pair policy; reproduce its frozen deferrals only.
    spec = p['native_followup'];d = json.loads(Path(spec['report']).read_text())['CAL_screen']
    data = np.load(spec['scores']);features = list(read(p['partitions']['CAL']['features']))
    rows = list(read(root/'CAL.sites.jsonl'))
    for z in rows:
        z['prior']=z['view'];z['candidate']=copy.deepcopy(z['view']);i=z['feature_index']
        if i is not None:
            action=decision(features[i],data['probabilities'][i],data['available'][i],d['defer_policy'])
            if action is not None:
                if action['kind']!='defer':raise ValueError('unexpected nondefer policy')
                z['candidate']['phone'].pop('errorType',None);z['candidate']['phone']['uncertain']=True
    assert sum(z['label_kind']=='correct' and flagged(z['candidate']) for z in rows)==86
    assert sum(exact(z,z['candidate']) for z in rows)==20
    report['paired_intervals']['native_followup_CAL']=bootstrap_pair(rows)
    write(out/'report.json',report)
    print('finished',flush=True)


if __name__=='__main__':main()

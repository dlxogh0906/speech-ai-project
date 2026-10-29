"""Inventory original TAG rows unused by declared auxiliary studies, not a test.

No model scores/admission filter/audio transformation. New UID and never-used
speaker scopes stay separate; exact PCM alias verification remains required.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import itertools
import json
import time
import eval_sibilant_directional as service
from prepare_sibilant_rank_pilot import destination
from mdd.sibilant_labels import label_sites, VOWELS


def protection(paths):
    sets = {k: set() for k in ('uid', 'speaker', 'item', 'norm_prompt')}
    for path in paths:
        for line in Path(path).open():
            value = json.loads(line)
            row = value.get('row', value)
            for key in sets:
                if key not in row:
                    raise ValueError('missing protected identity '+key+' in '+str(path))
                sets[key].add(row[key])
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('inventory source changed: '+name)
    train, cal, prior = [protection(p[k]) for k in ('protected_train', 'protected_cal', 'prior_evaluation')]
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    counts, strata, speakers, items = collections.Counter(), {}, {}, {}
    for scope in ('unused_UID', 'unused_UID_and_never_evaluated_speaker'):
        strata[scope] = {k: collections.Counter() for k in ('labels', 'directions', 'vowels', 'positions', 'other_contrasts')}
        speakers[scope], items[scope] = set(), set()
    seen = set()
    tick = time.monotonic()
    manifest = Path(p['manifest'])
    families = [set(z) for z in ('ㄱㄲㅋ', 'ㄷㄸㅌ', 'ㅂㅃㅍ', 'ㅈㅉㅊ')]
    with (out/'candidate_rows_PROVISIONAL.jsonl').open('x') as stream:
        for split in p['source_splits']:
            with (manifest/(split+'.tsv')).open() as ft, (manifest/(split+'.ltr')).open() as fa, \
                    (manifest/(split+'.canon')).open() as fc, (manifest/(split+'.meta.jsonl')).open() as fm:
                audio_root = Path(next(ft).strip())
                for index, values in enumerate(itertools.zip_longest(ft, fa, fc, fm)):
                    if any(z is None for z in values):
                        raise ValueError('manifest metadata/transcript length differs')
                    t, a, c, m = values
                    meta = json.loads(m)
                    rel, n = t.rstrip().split('\t')
                    if Path(rel).stem != meta['stem']:
                        raise ValueError('manifest UID differs')
                    counts['source_rows'] += 1
                    if '/mdd_gold_wav16k/' not in str(audio_root/rel):
                        counts['not_original_TAG_source'] += 1
                        continue
                    actual, canonical = a.split(), c.split()
                    if not any(z in ('ㅅ', 'ㅆ') and canonical[q+1] in VOWELS for q, z in enumerate(canonical[:-1])):
                        counts['no_target_onset'] += 1
                        continue
                    if destination(meta['item']) != 'test':
                        counts['not_preexisting_test_item_bucket'] += 1
                        continue
                    uid, speaker, item, prompt = meta['stem'], meta['spk'], meta['item'], meta['norm_prompt']
                    if uid in seen:
                        counts['duplicate_source_UID'] += 1
                        continue
                    seen.add(uid)
                    identities = {'uid': uid, 'speaker': speaker, 'item': item, 'norm_prompt': prompt}
                    if any(identities[k] in train[k] for k in train):
                        counts['protected_TRAIN_identity'] += 1
                        continue
                    if any(identities[k] in cal[k] for k in cal):
                        counts['protected_CAL_identity'] += 1
                        continue
                    if uid in prior['uid']:
                        counts['previously_evaluated_UID'] += 1
                        continue
                    if not actual or not canonical or any(meta.get('dropped', {}).values()) or not meta.get('tag_phones'):
                        counts['original_actual_annotation_missing_or_dropped_NOT_deletion'] += 1
                        continue
                    mapping = service.all_optimal_mapping(canonical, actual)
                    sites = label_sites(canonical, actual, mapping)
                    scopes = ['unused_UID']
                    if speaker not in prior['speaker']:
                        scopes.append('unused_UID_and_never_evaluated_speaker')
                    for scope in scopes:
                        counts[scope+'_rows'] += 1
                        speakers[scope].add(speaker)
                        items[scope].add(item)
                        for site in sites:
                            strata[scope]['labels'][site['label_kind']] += 1
                            strata[scope]['vowels'][site['following']+'/'+site['label_kind']] += 1
                            strata[scope]['positions'][site['position']+'/'+site['label_kind']] += 1
                            if site['label_kind'] == 'pair_error':
                                strata[scope]['directions'][site['canon']+'→'+site['actual']] += 1
                        # Other contrast counts cover this sibilant-containing subset only.
                        for q, expected in enumerate(canonical[:-1]):
                            family = next((z for z in families if expected in z), None)
                            if family is None or canonical[q+1] not in VOWELS:
                                continue
                            options = mapping[q]
                            if len(options) != 1:
                                value = 'ambiguous'
                            else:
                                at = next(iter(options))
                                value = actual[at] if at is not None else 'deleted'
                            strata[scope]['other_contrasts'][expected+'→'+value] += 1
                    service.write_line(stream, {'uid': uid, 'speaker': speaker, 'item': item,
                        'prompt': meta['prompt'], 'norm_prompt': prompt, 'wav': str(audio_root/rel),
                        'frames': int(n), 'actual': actual, 'canonical': canonical, 'sites': sites,
                        'source_split': split, 'source_manifest': str(manifest), 'manifest_index': index,
                        'language': meta.get('language'), 'inventory_scopes': scopes,
                        'label_provenance': 'existing_TAG_actual__LAB_phonemic_expected',
                        'exact_PCM_alias_verified': False})
    report = {'scope': p['scope'], 'counts': dict(counts),
        'strata': {scope: {k: dict(z) for k, z in groups.items()} for scope, groups in strata.items()},
        'speakers': {scope: len(z) for scope, z in speakers.items()},
        'items': {scope: sorted(z) for scope, z in items.items()},
        'protected_counts': {name: {k: len(z) for k, z in value.items()} for name, value in [('train', train), ('cal', cal), ('prior_evaluation', prior)]},
        'model_outputs_or_admission_used': False, 'exact_PCM_alias_checked': False,
        'not_certified_unused_projectwide_or_unseen_in_pretraining': True,
        'not_new_independent_human_validation': True, 'seconds': time.monotonic()-tick,
        'rows_sha256': service.sha(out/'candidate_rows_PROVISIONAL.jsonl'),
        'deployment': False, 'goal_complete': False}
    strict = strata['unused_UID_and_never_evaluated_speaker']['directions']
    report['strict_pair_count_preflight_passed'] = (all(strict[d] >= p['minimum_pairs_each_direction']
        for d in ('ㅅ→ㅆ', 'ㅆ→ㅅ')) and len(items['unused_UID_and_never_evaluated_speaker']) >= p['minimum_items'])
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps({k: report[k] for k in ('counts', 'speakers', 'strict_pair_count_preflight_passed')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

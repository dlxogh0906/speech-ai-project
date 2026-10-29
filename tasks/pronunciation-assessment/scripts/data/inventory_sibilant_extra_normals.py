"""Reaudit uncapped original TRAIN normals; never read model predictions.

Reproduce the original full-training population before applying newer heldout
protections. Preserve other-phone errors and report every source exclusion.
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
import functools
import gzip
import hashlib
import json
import time
import wave
from prepare_sibilant_rank_pilot import destination, priority
from inventory_sibilant_unseen_labels import protection
from eval_sibilant_directional import all_optimal_mapping, sha
from mdd.sibilant_labels import label_sites, VOWELS


@functools.lru_cache(maxsize=8192)
def annotated_sites(canonical, actual):
    mapping = ([{q} for q in range(len(canonical))] if canonical == actual else
               all_optimal_mapping(canonical, actual))
    return label_sites(canonical, actual, mapping)


def protected_identity(row, protected):
    return any(row[k] in protected[k] for k in protected)


def original_row(meta, frames, index, wav, manifest, vocab):
    actual, canonical = meta['ltr'], meta['canon']
    if not 8000 <= frames <= 192000:
        return None, 'duration'
    if (not actual or not canonical or any(t not in vocab for t in actual+canonical)
            or any(meta.get('dropped', {}).values())):
        return None, 'annotation_missing_dropped_or_oov'
    if not any(t in ('ㅅ', 'ㅆ') and canonical[q+1] in VOWELS
               for q, t in enumerate(canonical[:-1])):
        return None, 'no_target_onset'
    sites = annotated_sites(tuple(canonical), tuple(actual))
    return {'uid':meta['stem'], 'speaker':meta['spk'], 'item':meta['item'],
        'prompt':meta['prompt'], 'norm_prompt':meta['norm_prompt'],
        'wav':str(wav), 'frames':frames, 'actual':actual, 'canonical':canonical,
        'sites':sites, 'utterance_all_correct':actual == canonical,
        'source_split':'train', 'manifest_index':index, 'source_manifest':str(manifest),
        'language':meta.get('language'),
        'label_provenance':'human_TAG_actual__LAB_phonemic_canonical'}, 'eligible'


def pcm_hashes(row):
    with wave.open(row['wav'], 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                stream.getnframes()) != (16000, 1, 2, row['frames']):
            raise ValueError('original PCM format/frame mismatch: '+row['uid'])
        raw = stream.readframes(stream.getnframes())
    return hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw[:32000*20]).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for name, digest in p['sha256'].items():
        if sha(name) != digest:
            raise ValueError('frozen inventory input changed: '+name)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(a.protocol).read_text())
    old_protocol = json.loads(Path(p['old_protocol']).read_text())
    old_protected = protection(old_protocol['args']['heldout'])
    current_protected = protection(p['current_heldout'])
    old_rows = [json.loads(z) for z in Path(p['old_train']).open()]
    old_uids = {r['uid'] for r in old_rows}
    blocked_full, blocked_prefix = set(), set()
    for path in p['pcm_catalogues']:
        for line in Path(path).open():
            row = json.loads(line)
            blocked_full.add(row.get('PCM16_sha256', row.get('full_PCM16_sha256')))
            blocked_prefix.add(row['runtime_prefix_PCM16_sha256'])
    if None in blocked_full:
        raise ValueError('missing protected PCM hash')
    manifest = Path(old_protocol['args']['manifest'])
    vocab = json.loads(Path(p['vocab']).read_text())['vocab']
    membership, excluded, seen = {}, collections.Counter(), set()
    with (manifest/'train.tsv').open() as stream:
        audio_root = Path(next(stream).strip())
        for index, line in enumerate(stream):
            rel, frames = line.rstrip().split('\t')
            uid, parts = Path(rel).stem, Path(rel).stem.split('-')
            if (len(parts) < 6 or '/mdd_gold_wav16k/' not in '/'+rel
                    or destination(parts[5]) != 'train'):
                continue
            ids = {'uid':uid, 'speaker':parts[0], 'item':parts[5]}
            if any(ids[k] in old_protected[k] for k in ids):
                excluded['original_protected_uid_speaker_or_item'] += 1
                continue
            membership.setdefault(uid, (int(frames), index, audio_root/rel))
    inventory, audit, full_choice = collections.Counter(), collections.Counter(), {}
    tick = time.monotonic()
    with gzip.open(old_protocol['args']['cache'], 'rt', encoding='utf-8') as stream, \
            gzip.open(out/'eligible_rows_PROVISIONAL.jsonl.gz', 'wt', encoding='utf-8') as dest:
        for line in stream:
            meta = json.loads(line)
            uid = meta['stem']
            if uid not in membership or uid in seen:
                continue
            seen.add(uid)
            if meta['norm_prompt'] in old_protected['norm_prompt']:
                excluded['original_protected_prompt'] += 1
                continue
            frames, index, wav = membership[uid]
            row, state = original_row(meta, frames, index, wav, manifest, vocab)
            if row is None:
                excluded[state] += 1
                continue
            inventory['utterances'] += 1
            inventory.update(s['canon']+'/'+s['label_kind'] for s in row['sites'])
            if uid in old_uids:
                audit['old_selected_UID'] += 1
                continue
            if any(s['label_kind'] != 'correct' for s in row['sites']):
                audit['unselected_target_error_or_ambiguous'] += 1
                continue
            if protected_identity(row, current_protected):
                audit['current_heldout_identity'] += 1
                continue
            full, prefix = pcm_hashes(row)
            if full in blocked_full or prefix in blocked_prefix:
                audit['protected_PCM_or_runtime_prefix_alias'] += 1
                continue
            row['full_PCM16_sha256'], row['runtime_prefix_PCM16_sha256'] = full, prefix
            audit['provisional_extra_normal_rows'] += 1
            previous = full_choice.get(full)
            if previous is None or priority(uid) < priority(previous):
                full_choice[full] = uid
            dest.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
            if inventory['utterances'] % 20000 == 0:
                print('source', inventory['utterances'], 'extra', audit['provisional_extra_normal_rows'],
                      'seconds', round(time.monotonic()-tick, 1), flush=True)
    expected = json.loads(Path(p['old_report']).read_text())['inventory']
    if dict(inventory) != expected:
        raise AssertionError('original population/labels changed: '+str(dict(inventory)))
    counts, speakers, items, by_item, strata = collections.Counter(), set(), set(), collections.Counter(), collections.Counter()
    kept = set(full_choice.values())
    with gzip.open(out/'eligible_rows_PROVISIONAL.jsonl.gz', 'rt', encoding='utf-8') as stream, \
            gzip.open(out/'candidate_rows.jsonl.gz', 'wt', encoding='utf-8') as dest:
        for line in stream:
            row = json.loads(line)
            if row['uid'] not in kept:
                audit['duplicate_new_exact_PCM'] += 1
                continue
            dest.write(line)
            counts['utterances'] += 1
            counts['whole_utterance_correct' if row['utterance_all_correct'] else 'other_phone_errors_retained'] += 1
            speakers.add(row['speaker'])
            items.add(row['item'])
            by_item[row['item']] += 1
            strata.update(s['canon']+'/'+s['following'] for s in row['sites'])
    minimum = p['minimum']
    gate = (counts['utterances'] >= minimum['utterances'] and len(speakers) >= minimum['speakers']
        and len(items) >= minimum['items'] and all(strata[c+'/'+v] >= minimum['each_high_vowel_normal_sites']
        for c in ('ㅅ', 'ㅆ') for v in ('ㅣ', 'ㅡ', 'ㅜ')))
    report = {'scope':__doc__, 'original_population_exactly_reproduced':True,
        'original_inventory':dict(inventory), 'source_exclusions':dict(excluded), 'new_protection_ledger':dict(audit),
        'extra_normal_counts':dict(counts), 'speakers':len(speakers), 'items':len(items),
        'by_item':dict(by_item), 'actual_consonant_following_vowel':dict(strata),
        'current_protected_counts':{k:len(z) for k, z in current_protected.items()},
        'protected_PCM_hashes':len(blocked_full), 'minimum':minimum,
        'proceed_new_normal_pilot':gate, 'annotation_cache':annotated_sites.cache_info()._asdict(),
        'rows_sha256':sha(out/'candidate_rows.jsonl.gz'), 'seconds':time.monotonic()-tick,
        'model_scores_or_admission_used':False, 'new_independent_human_validation':False,
        'deployment':False, 'goal_complete':False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'report_sha256':sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

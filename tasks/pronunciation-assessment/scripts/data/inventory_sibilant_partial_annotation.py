"""Inventory locally certain TAG C/V targets in globally rejected TRAIN annotation.

Unknown spans remain unknown in BOTH sequences. No complete transcript or
whole-utterance correctness is invented. Candidate labels need a later runtime
mapping/admission/coverage check before they can enter any training experiment.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import gzip
import hashlib
import json
import time
from eval_sibilant_directional import sha,jsonable
from inventory_sibilant_unseen_labels import protection
from inventory_sibilant_extra_normals import pcm_hashes
from prepare_sibilant_rank_pilot import destination,priority
from mdd.sibilant_labels import PAIR,VOWELS
from mdd.sibilant_partial_annotation import annotation_sequence,recover_sites,SCHEMA


def write(path,data):
    Path(path).write_text(json.dumps(jsonable(data),ensure_ascii=False,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for path,digest in p['sha256'].items():
        if sha(path) != digest:
            raise ValueError('frozen partial annotation source changed: '+path)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',p)
    old = json.loads(Path(p['old_protocol']).read_text())
    old_protected,current = protection(old['args']['heldout']),protection(p['current_heldout'])
    vocab = set(json.loads(Path(p['vocab']).read_text())['vocab'])
    old_uids = {json.loads(z)['uid'] for z in Path(p['old_train']).open()}
    blocked_full,blocked_prefix = set(),set()
    for path in p['pcm_catalogues']:
        for line in Path(path).open():
            z = json.loads(line)
            blocked_full.add(z.get('PCM16_sha256',z.get('full_PCM16_sha256')))
            blocked_prefix.add(z['runtime_prefix_PCM16_sha256'])
    if None in blocked_full:
        raise ValueError('missing protected source PCM hash')
    membership = {}
    manifest = Path(old['args']['manifest'])
    with (manifest/'train.tsv').open() as stream:
        audio_root = Path(next(stream).strip())
        for index,line in enumerate(stream):
            rel,n = line.rstrip().split('\t');uid = Path(rel).stem;parts = uid.split('-')
            if len(parts)<6 or '/mdd_gold_wav16k/' not in '/'+rel or destination(parts[5]) != 'train':
                continue
            ids = {'uid':uid,'speaker':parts[0],'item':parts[5]}
            if any(ids[k] in old_protected[k] for k in ids):
                continue
            membership.setdefault(uid,(int(n),index,str(audio_root/rel)))
    old_exclusions,ledger,states,dropped = [collections.Counter() for _ in range(4)]
    seen,choice,conflicts = set(),{},set()
    tick = time.monotonic()
    scanned = 0
    with gzip.open(old['args']['cache'],'rt',encoding='utf-8') as stream, \
            gzip.open(out/'candidate_rows_PROVISIONAL.jsonl.gz','wt',encoding='utf-8') as dest:
        for line in stream:
            meta = json.loads(line);scanned += 1;uid = meta['stem']
            if uid not in membership or uid in seen:
                continue
            seen.add(uid)
            if meta['norm_prompt'] in old_protected['norm_prompt']:
                continue
            frames,index,wav = membership[uid]
            if not 8000 <= frames <= 192000:
                old_exclusions['duration'] += 1;continue
            actual,canonical = meta['ltr'],meta['canon']
            rejected = (not actual or not canonical or any(z not in vocab for z in actual+canonical)
                        or any(meta.get('dropped',{}).values()))
            if not rejected:
                ledger['previous_global_annotation_eligible_not_reinvented'] += 1
                continue
            old_exclusions['annotation_missing_dropped_or_oov'] += 1
            ids = {'uid':uid,'speaker':meta['spk'],'item':meta['item'],'norm_prompt':meta['norm_prompt']}
            if any(ids[k] in current[k] for k in ids):
                ledger['current_protected_identity'] += 1;continue
            dropped.update(meta.get('dropped',{}))
            ref = annotation_sequence(meta.get('lab_phones',[]),vocab)['tokens']
            targets = sum(t in PAIR and ref[q+1] in VOWELS for q,t in enumerate(ref[:-1]))
            ledger['globally_rejected_after_current_protection'] += 1
            ledger['literal_expected_target_sites_before_mapping'] += targets
            if not targets:
                ledger['no_literal_expected_CV_target'] += 1;continue
            recovery = recover_sites(meta,vocab,frames)
            states.update(s['state'] for s in recovery['sites'])
            stable = [s for s in recovery['sites'] if s['label_kind'] in ('correct','pair_error')]
            if not stable:
                ledger['no_stable_literal_actual_CV_target'] += 1;continue
            if uid in old_uids:
                raise ValueError('globally rejected metadata unexpectedly already selected')
            if not meta.get('prompt'):
                ledger['missing_runtime_prompt'] += 1;continue
            row = dict(ids,prompt=meta['prompt'],language=meta.get('language'),wav=wav,frames=frames,
                source_split='train',source_manifest=str(manifest),manifest_index=index,
                annotation_schema=SCHEMA,label_provenance='existing_TAG_literal_actual__LAB_literal_expected__all_optimal_partial_mapping',
                expected_partial=recovery['expected_partial'],actual_partial=recovery['actual_partial'],
                sites=recovery['sites'],minimum_known_edit_cost=recovery['minimum_known_edit_cost'],
                expected_unknown_spans=recovery['expected_unknown_spans'],actual_unknown_spans=recovery['actual_unknown_spans'],
                tag_phones=meta['tag_phones'],lab_phones=meta['lab_phones'],dropped=meta.get('dropped',{}),
                legacy_actual=meta['ltr'],legacy_canonical=meta['canon'],
                full_actual_transcript_available=False,whole_utterance_correctness_asserted=False,
                runtime_mapping_checked=False,usable_training_label=False)
            full,prefix = pcm_hashes(row)
            if full in blocked_full or prefix in blocked_prefix:
                ledger['previous_auxiliary_or_protected_PCM_alias'] += 1;continue
            row['full_PCM16_sha256'],row['runtime_prefix_PCM16_sha256'] = full,prefix
            signature = hashlib.sha256(json.dumps([meta['norm_prompt'],recovery['expected_partial'],
                recovery['actual_partial'],stable],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            row['local_annotation_sha256'] = signature
            previous = choice.get(full)
            if previous is not None and previous[1] != signature:
                conflicts.add(full)
            if previous is None or priority(uid) < priority(previous[0]):
                choice[full] = uid,signature
            dest.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            ledger['provisional_unique_UID_rows'] += 1
            if ledger['globally_rejected_after_current_protection']%100 == 0:
                progress = {'scanned_metadata':scanned,'global_rejected':old_exclusions['annotation_missing_dropped_or_oov'],
                    'locally_provisional_UID':ledger['provisional_unique_UID_rows'],'seconds':time.monotonic()-tick}
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    expected_exclusions = json.loads(Path(p['old_report']).read_text())['exclusions']
    for key in ('duration','annotation_missing_dropped_or_oov'):
        if old_exclusions[key] != expected_exclusions[key]:
            raise ValueError('original global exclusion population changed: '+key)
    kept,counts,sites = {v[0] for k,v in choice.items() if k not in conflicts},collections.Counter(),collections.Counter()
    speakers,items,groups = set(),set(),{k:collections.Counter() for k in ('item','following','position','language','directions')}
    with gzip.open(out/'candidate_rows_PROVISIONAL.jsonl.gz','rt',encoding='utf-8') as stream, \
            gzip.open(out/'candidate_rows.jsonl.gz','wt',encoding='utf-8') as dest:
        for line in stream:
            row = json.loads(line)
            if row['uid'] not in kept:
                ledger['new_PCM_duplicate_or_conflicting_annotation'] += 1;continue
            dest.write(line);counts['rows'] += 1;speakers.add(row['speaker']);items.add(row['item'])
            stable = [s for s in row['sites'] if s['label_kind'] in ('correct','pair_error')]
            counts['rows_with_pair_error'] += any(s['label_kind']=='pair_error' for s in stable)
            counts['unresolved_sites_retained'] += sum(s['label_kind'] is None for s in row['sites'])
            for s in stable:
                key = s['expected']+'/'+s['label_kind'];sites[key] += 1
                groups['item'][row['item']+'/'+key] += 1
                groups['following'][s['following']+'/'+key] += 1
                groups['position'][s['position']+'/'+key] += 1
                groups['language'][str(row.get('language'))+'/'+key] += 1
                if s['label_kind']=='pair_error':groups['directions'][s['expected']+'→'+s['actual']] += 1
    report = {'scope':__doc__,'schema':SCHEMA,'source_manifest_TRAIN_membership':len(membership),
        'original_global_exclusions_exactly_reproduced':dict(old_exclusions),
        'source_ledger':dict(ledger),'dropped_marker_counts_after_protection':dict(dropped),
        'annotation_site_states_before_PCM_checks':dict(states),'candidate_counts':dict(counts),
        'candidate_site_labels':dict(sites),'speakers':len(speakers),'items':len(items),
        'groups':{k:dict(v) for k,v in groups.items()},'conflicting_new_PCM_groups_excluded':len(conflicts),
        'protected_identity_counts':{k:len(z) for k,z in current.items()},'protected_or_used_PCM_hashes':len(blocked_full),
        'all_candidate_PCM16_format_and_frames_checked':True,'usable_training_labels_created':0,
        'runtime_mapping_admission_and_ROI_support_checked':False,'whole_utterance_correctness_inferred':False,
        'not_certified_unused_by_historical_base_encoder':True,'new_independent_human_labels':False,
        'model_inference':False,'deployment':False,'goal_complete':False,'seconds':time.monotonic()-tick,
        'candidate_rows_sha256':sha(out/'candidate_rows.jsonl.gz')}
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':sha(out/'report.json')})
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','dropped_marker_counts_after_protection')},ensure_ascii=False),flush=True)


if __name__ == '__main__':main()

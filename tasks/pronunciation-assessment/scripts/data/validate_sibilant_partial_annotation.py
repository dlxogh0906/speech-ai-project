"""Require original LABEL/TAG co-located singleton slots after partial recovery."""
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
import json
import time
import numpy as np
from eval_sibilant_directional import sha,jsonable
from inventory_sibilant_extra_normals import pcm_hashes
from inventory_sibilant_unseen_labels import protection
from mdd.sibilant_partial_annotation import recover_sites
from mdd.sibilant_partial_witness import witness_sites,SCHEMA


def write(path,value):
    Path(path).write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for path,digest in p['sha256'].items():
        if sha(path) != digest:
            raise ValueError('frozen partial-witness source changed: '+path)
    source = Path(p['inventory'])
    prior = json.loads((source/'report.json').read_text())
    if json.loads((source/'finished.json').read_text())['report_sha256'] != sha(source/'report.json'):
        raise ValueError('completed inventory report changed')
    if sha(source/'candidate_rows.jsonl.gz') != prior['candidate_rows_sha256']:
        raise ValueError('completed inventory candidate rows changed')
    original_p = json.loads((source/'protocol.json').read_text())
    protected = protection(original_p['current_heldout'])
    allowed = set(json.loads(Path(original_p['vocab']).read_text())['vocab'])
    blocked_full,blocked_prefix = set(),set()
    for path in original_p['pcm_catalogues']:
        for line in Path(path).open():
            z = json.loads(line)
            blocked_full.add(z.get('PCM16_sha256',z.get('full_PCM16_sha256')))
            blocked_prefix.add(z['runtime_prefix_PCM16_sha256'])
    checks = set(np.linspace(0,prior['candidate_counts']['rows']-1,64,dtype=np.int64).tolist())
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    before,after,counts,states = [collections.Counter() for _ in range(4)]
    groups = {k:collections.Counter() for k in ('directions','following','item','position','language')}
    seen_uid,seen_pcm,speakers,items = set(),set(),set(),set()
    audit_rows = [];tick = time.monotonic()
    with gzip.open(source/'candidate_rows.jsonl.gz','rt',encoding='utf-8') as stream, \
            gzip.open(out/'candidate_rows.jsonl.gz','wt',encoding='utf-8') as dest, \
            gzip.open(out/'rejected_source_slots.jsonl.gz','wt',encoding='utf-8') as rejected:
        for j,line in enumerate(stream):
            row = json.loads(line);counts['input_rows'] += 1
            if (row['uid'] in seen_uid or row['full_PCM16_sha256'] in seen_pcm
                    or row['source_split'] != 'train' or any(row[k] in protected[k] for k in protected)
                    or row['full_PCM16_sha256'] in blocked_full
                    or row['runtime_prefix_PCM16_sha256'] in blocked_prefix):
                raise ValueError('candidate identity/PCM partition changed')
            seen_uid.add(row['uid']);seen_pcm.add(row['full_PCM16_sha256'])
            if j in checks:
                meta = {'tag_phones':row['tag_phones'],'lab_phones':row['lab_phones'],
                        'ltr':row['legacy_actual'],'canon':row['legacy_canonical']}
                recovered = recover_sites(meta,allowed,row['frames'])
                if recovered['sites'] != row['sites']:
                    raise ValueError('original partial recovery not reproducible')
                full,prefix = pcm_hashes(row)
                if (full,prefix) != (row['full_PCM16_sha256'],row['runtime_prefix_PCM16_sha256']):
                    raise ValueError('candidate original PCM changed')
                audit_rows.append(row['uid'])
            before.update(s['expected']+'/'+s['label_kind'] for s in row['sites'] if s['label_kind'] is not None)
            validated = witness_sites(row,allowed)
            states.update(s['witness_state'] for s in validated)
            for site in validated:
                site['provisional_label_kind'] = site['label_kind']
                if not site['source_time_witness']:
                    site['label_kind'] = None
            row['sites'] = validated;row['annotation_schema'] = SCHEMA
            stable = [s for s in validated if s['source_time_witness']]
            if not stable:
                counts['rows_without_source_slot_witness'] += 1
                rejected.write(json.dumps({'uid':row['uid'],'site_states':validated},ensure_ascii=False)+'\n')
                continue
            dest.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            counts['rows_with_source_slot_witness'] += 1
            counts['rows_with_pair_error'] += any(s['label_kind']=='pair_error' for s in stable)
            counts['unknown_sites_retained_in_kept_rows'] += sum(s['label_kind'] is None for s in validated)
            speakers.add(row['speaker']);items.add(row['item'])
            for s in stable:
                key = s['expected']+'/'+s['label_kind'];after[key] += 1
                groups['following'][s['following']+'/'+key] += 1
                groups['item'][row['item']+'/'+key] += 1
                groups['position'][s['position']+'/'+key] += 1
                groups['language'][str(row.get('language'))+'/'+key] += 1
                if s['label_kind']=='pair_error':groups['directions'][s['expected']+'→'+s['actual']] += 1
            if (j+1)%2000 == 0:
                print(json.dumps({'source_rows':j+1,'kept':counts['rows_with_source_slot_witness'],
                                  'seconds':time.monotonic()-tick}),flush=True)
    if dict(before) != prior['candidate_site_labels'] or len(audit_rows) != 64:
        raise ValueError('full original candidate coverage/raw audit changed')
    report = {'scope':__doc__,'schema':SCHEMA,'source_slot_precision_seconds':.001,
        'counts':dict(counts),'provisional_site_counts_reproduced':dict(before),'source_witness_site_counts':dict(after),
        'source_witness_states':dict(states),'speakers':len(speakers),'items':len(items),
        'groups':{k:dict(v) for k,v in groups.items()},'uniform64_raw_PCM_and_recovery_reproduced':audit_rows,
        'all_candidate_identity_and_PCM_partition_hashes_rechecked':True,
        'annotation_time_is_not_physical_acoustic_boundary_truth':True,
        'usable_training_labels_created':0,'runtime_expected_mapping_admission_ROI_unchecked':True,
        'model_inference':False,'whole_actual_transcript_or_correctness_invented':False,
        'new_independent_human_validation':False,'deployment':False,'goal_complete':False,
        'candidate_rows_sha256':sha(out/'candidate_rows.jsonl.gz'),'seconds':time.monotonic()-tick}
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':sha(out/'report.json')})
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','uniform64_raw_PCM_and_recovery_reproduced')},ensure_ascii=False),flush=True)


if __name__ == '__main__':main()

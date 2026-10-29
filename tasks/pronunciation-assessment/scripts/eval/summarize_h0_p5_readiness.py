"""Derived P5 preparation summaries, preserving the frozen audit and design."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
import argparse
import collections
import csv
import gzip
import json
import numpy as np
from audit_h0_p5_readiness import write,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit',required=True)
    ap.add_argument('--balance',required=True);a=ap.parse_args();root=Path(a.audit);balance=Path(a.balance)
    rows=list(csv.DictReader((balance/'contribution_groups.tsv').open(),delimiter='\t'))
    lookup={(r['arm'],r['dimension'],r['key']):r for r in rows};shifts=[]
    for dim in ('source','l1','speaker','item'):
        keys=sorted({r['key'] for r in rows if r['dimension']==dim and not r['key'].startswith('<native') and r['key']!='native'})
        b=np.array([float(lookup['C0',dim,k]['ctc_coefficient']) for k in keys]);b/=b.sum()
        for arm in ('C1','C2'):
            v=np.array([float(lookup[arm,dim,k]['ctc_coefficient']) for k in keys]);v/=v.sum();diff=v-b
            shifts.append(dict(arm=arm,dimension=dim,total_variation=float(np.abs(diff).sum()/2),
                max_absolute_share_change_pp=float(np.abs(diff).max()*100),
                largest_increase=keys[int(diff.argmax())],largest_decrease=keys[int(diff.argmin())]))
    write(balance/'contribution_shifts.json',dict(scope='foreign weighted CTC contribution; physical sampling unchanged',
        shifts=shifts,source_sha256=sha(__file__)))
    rr=[json.loads(l) for l in (root/'recordings.jsonl').open()];reserved=[r for r in rr if r['historical_role']=='test']
    rreport=json.loads((root/'report.json').read_text());timing=[];unknown=collections.Counter()
    with gzip.open(root/'source_records.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            for tier in ('tag_phones','lab_phones'):
                prev=-1.
                for start,end,value in r[tier]:
                    if start<prev-.002:timing.append((r['stem'],tier,prev-start))
                    prev=max(prev,end)
                    if any(c not in rreport['vocabulary'] and not c.isspace() for c in value):unknown[tier,value]+=1
    windows=[]
    for mode in ('fixed','energy'):
        subset=[r for r in rreport['windows'] if r['role']=='test' and r['mode']==mode]
        counts=sum((collections.Counter(r['conditional_counts']) for r in subset),collections.Counter())
        windows.append(dict(mode=mode,chunks=sum(r['chunks'] for r in subset),
            complete_annotation_chunks=sum(r['complete_annotation_chunks'] for r in subset),
            conditional_normal=counts['normal'],conditional_errors=counts['errors'],
            conditional_deletions=counts['deletion'],conditional_insertion_gaps=counts['insertion_gaps'],
            absent_normal_phones=[t for t in rreport['vocabulary'] if not counts['normal:'+t]]))
    bindings=json.loads((root/'manifest_bindings.json').read_text())
    prompts=list(csv.DictReader((balance/'prompt_caps.tsv').open(),delimiter='\t'))
    missing=sorted(set(bindings['H0_train']['items'])-{r['item'] for r in prompts})
    write(root/'candidate_summary.json',dict(recordings=len(rr),speakers=len({r['speaker'] for r in rr}),
        reserved_recordings=len(reserved),reserved_speakers=len({r['speaker'] for r in reserved}),
        total_audio_hours=sum(r['seconds'] for r in rr)/3600,
        reserved_audio_hours=sum(r['seconds'] for r in reserved)/3600,
        unknown_tag_or_reference_records=sum(bool(r['tag_check']['kinds'].get('unknown') or r['lab_check']['kinds'].get('unknown')) for r in rr),
        non_jamo_label_inventory=[dict(tier=k[0],label=k[1],intervals=v) for k,v in unknown.most_common()],
        interval_overlap=dict(count=len(timing),min_seconds=min(t[2] for t in timing),max_seconds=max(t[2] for t in timing),
            interpretation='previous interval end exceeds next start by >2ms in the raw stored times; includes roundoff near 2ms; not independently adjudicated alignment errors'),
        windows=windows,full_train_items_without_continuation_exposure=missing,source_sha256=sha(__file__)))


if __name__=='__main__':main()

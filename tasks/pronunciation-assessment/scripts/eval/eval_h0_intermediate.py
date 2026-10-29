"""P3 held-out-speaker development evaluation, including complete gap strings.

The one seen-CAL threshold is frozen before selection evaluation. Gap results
use greedy edit alignment and are not a calibrated production insertion policy.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import itertools
import json
import time
import numpy as np
import torch
from train_h0_intermediate import P3Model, bind
from train_h0_backbone_control import file_sha
from train_w2vbert_ctc import read_split, read_wav
from eval_h0_backbone_control import (infer_utterance_sites, threshold_for_normal_budget,
                                     metrics, cluster_intervals)
from audit_h0_backbone_protocol import calibration_role
from audit_h0_all_phones import tsv, dimensions, ckey
from mdd.h0_phone_audit import (contexts, events, metric_counts, project_gaps,
                                sequence_audit, standardize, rate)


def evaluate_ledgers(out, p, condition, step, checkpoint_hash):
    sites = [json.loads(line) for line in (out/'sites.jsonl').open()]
    cal = [s for s in sites if s['split']=='valid' and s['role']=='calibration']
    threshold = threshold_for_normal_budget(cal)
    selected = [s for s in sites if s['role']=='selection']
    results, uncertainty, composition = {}, {}, {}
    counts = collections.defaultdict(collections.Counter)
    context_counts = collections.defaultdict(dict)
    confusions = collections.Counter()
    for s in selected:
        ev = events(s, threshold)
        for dimension, key in dimensions(s):
            counts[(s['split'],dimension,key)].update(ev)
        confusions[(s['split'],s['expected'],s['actual'],s['predicted'],s['position'])] += 1
    for split in ('valid','valid_unseen'):
        subset = [s for s in selected if s['split']==split]
        groups = dict(all=subset, sentences=[s for s in subset if s['task']=='RS'],
                      strict_native_unseen=[s for s in subset if s['strict_native_unseen']],
                      deletions=[s for s in subset if s['actual'] is None])
        results[split] = {k:dict(greedy=metrics(v), calibrated=metrics(v,threshold))
                          for k,v in groups.items()}
        for group in results[split].values():
            fpr = group['calibrated']['normal_fpr']
            group['calibrated']['measured_fpr_within_1pct'] = fpr <= .01 if fpr is not None else None
        uncertainty[split] = {k:{axis:cluster_intervals(v,threshold,axis)
                                for axis in ('speaker','item')} for k,v in groups.items()}
        composition[split] = dict(utterances=len({s['uid'] for s in subset}),
                                 speakers=len({s['speaker'] for s in subset}),
                                 items=len({s['item'] for s in subset}))
    # Explicit zero-support rows prevent missing phones from disappearing.
    vocab = json.loads(Path(p['parent_protocol']).read_text())['stage2']['manifest']
    phones = [line.split()[0] for line in (Path(vocab)/'dict.ltr.txt').read_text().splitlines()]
    for split in ('valid','valid_unseen'):
        for phone in phones:
            counts[(split,'phone',phone)]
    group_rows = [dict(split=k[0],dimension=k[1],key=k[2],**metric_counts(v))
                  for k,v in sorted(counts.items())]
    tsv(out/'all_phone_groups.tsv',group_rows)
    tsv(out/'confusions.tsv',[dict(split=k[0],expected=k[1],actual=k[2],predicted=k[3],
                                  position=k[4],n=v) for k,v in confusions.items()])
    for (split,dimension,key), c in counts.items():
        if dimension=='context_simple': context_counts[key][split]=dict(c)
    seq_counts = collections.defaultdict(collections.Counter)
    normal_confusions = collections.Counter()
    suri = collections.Counter()
    for line in (out/'sequences.jsonl').open():
        r = json.loads(line)
        if r['role']!='selection': continue
        c = sequence_audit(r['canonical'],r['actual'],r['predicted'])
        seq_counts[r['split']].update(c)
        if r['actual']==r['canonical']:
            normal_confusions[(r['split'],r['item'],''.join(r['canonical']),''.join(r['predicted']))] += 1
            if r['split']=='valid_unseen' and r['canonical']==list('ㅅㅜㄹㅣ'):
                suri['total'] += 1
                suri['whole_correct'] += r['predicted']==r['canonical']
                suri['whole_sori'] += r['predicted']==list('ㅅㅗㄹㅣ')
                pred, _ = project_gaps(r['canonical'],r['predicted'])
                suri['u_correct'] += pred[1]=='ㅜ'
    for c in seq_counts.values():
        c['insertion_exact_recall'] = rate(c['insertion_exact_gaps'],c['true_insertion_gaps'])
        c['insertion_exact_precision'] = rate(c['insertion_exact_gaps'],c['predicted_insertion_gaps'])
        c['false_insertion_gap_rate'] = rate(c['false_insertion_gaps'],c['normal_gaps'])
    tsv(out/'normal_sequence_confusions.tsv',[dict(split=k[0],item=k[1],canonical=k[2],predicted=k[3],n=v)
                                               for k,v in normal_confusions.items()])
    report = dict(condition=condition,backbone=condition,step=step,checkpoint_sha256=checkpoint_hash,
        protocol_sha256=file_sha(p['_path']),calibration=metrics(cal,threshold),
        threshold=None if threshold==-np.inf else threshold,threshold_none_means='all scored mismatches',
        results=results,uncertainty=uncertainty,composition=composition,sequences=dict(seq_counts),suri=dict(suri),
        phone_counts={split:{phone:metric_counts(counts[(split,'phone',phone)]) for phone in phones}
                      for split in ('valid','valid_unseen')},
        matched_contexts=standardize(context_counts,'normal_fp','normal'),
        limitations=['reused validation development prompts, not independent unseen sentence confirmation',
            'fixed seen CAL threshold, speaker/prompt bootstrap separately; no seed or CAL resampling',
            'whole audio offline inference; no production VAD, G2P variant selection, or API display policy',
            'gap strings use one edit path; alignment sensitivity is reported separately',
            'original LAB/TAG unchanged; allowable-pronunciation sensitivity is separate from primary metrics'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args(); p=bind(a.protocol); p['_path']=a.protocol
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(a.checkpoint,map_location='cpu')
    assert ck['protocol_sha256']==file_sha(a.protocol)
    condition,step,vocab=ck['condition'],ck['step'],ck['vocab']
    model=P3Model(p['source'],vocab,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),condition)
    model.load_state_dict(ck['model'],strict=True); model.eval(); del ck
    audit=json.loads(Path(p['input_audit']).read_text())
    manifest=Path(p['manifest']); start=time.time()
    with (out/'sites.jsonl').open('x') as ledger, (out/'sequences.jsonl').open('x') as sequence_ledger:
        for split in ('valid','valid_unseen'):
            audio,actual=read_split(str(manifest),split)
            canon=[line.split() for line in (manifest/(split+'.canon')).read_text().splitlines()]
            meta=[json.loads(line) for line in (manifest/(split+'.meta.jsonl')).open()]
            for i,(path,frames) in enumerate(audio):
                m=meta[i]; role=calibration_role(m['spk'])
                if split=='valid_unseen' and role!='selection': continue
                x=read_wav(path,20.); assert len(x)==frames
                seq={}
                inferred=infer_utterance_sites(model,x,canon[i],actual[i],vocab,m,
                    audit['native_text_overlap'].get(m['item']),split,i,sequence_out=seq)
                ctx=contexts(tuple(canon[i]),m['prompt'])
                for s in inferred:
                    s.update(ctx[s['q']]); ledger.write(json.dumps(s,ensure_ascii=False)+'\n')
                seq.update(split=split,row=i,uid=m['stem'],speaker=m['spk'],item=m['item'],
                           role=role,canonical=canon[i],actual=actual[i])
                sequence_ledger.write(json.dumps(seq,ensure_ascii=False)+'\n')
                if i%1000==0:
                    ledger.flush(); sequence_ledger.flush()
                    print(json.dumps(dict(condition=condition,step=step,split=split,row=i,
                                          elapsed_seconds=time.time()-start)),flush=True)
    report=evaluate_ledgers(out,p,condition,step,file_sha(a.checkpoint))
    (out/'finished.json').write_text(json.dumps(dict(condition=condition,step=step,
        elapsed_seconds=time.time()-start,report_sha256=file_sha(out/'report.json'),
        sites_sha256=file_sha(out/'sites.jsonl'),sequences_sha256=file_sha(out/'sequences.jsonl')),indent=2))


if __name__=='__main__':
    main()

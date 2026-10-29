"""TRAIN-only feasibility of universal prompt/context weighting; no optimizer.

Preserves the exact B0 audio stream. These are utterance CTC loss weights, not
token-local losses or changed physical exposure. The readiness gate is separate.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import csv
import hashlib
import json
import math
import numpy as np
from train_h0_backbone_control import file_sha,stream
from train_w2vbert_ctc import read_split
from mdd.h0_phone_audit import surface_position
from audit_h0_p5_readiness import write


def capped_mass(mass, multiple):
    """Water fill with a common cap, preserving total mass; never singled-out IDs."""
    x=np.asarray(mass,dtype=float);cap=float(x.mean()*multiple)
    lo,hi=1.,2.
    while np.minimum(x*hi,cap).sum()<x.sum()-1e-10:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if np.minimum(x*mid,cap).sum()<x.sum():lo=mid
        else:hi=mid
    return np.minimum(x*hi,cap),cap


def contexts(tokens):
    return [(t,surface_position(tokens,j),tokens[j-1] if j else '<BOS>',
             tokens[j+1] if j+1<len(tokens) else '<EOS>') for j,t in enumerate(tokens)]


def tsv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--p3-protocol',required=True)
    ap.add_argument('--readiness',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    p=json.loads(Path(a.p3_protocol).read_text());manifest=Path(p['manifest'])
    for suffix in ('tsv','ltr','source'):
        f=manifest/('train.'+suffix)
        if file_sha(f)!=p['bindings'][str(f)]:raise ValueError('TRAIN changed')
    design=dict(status='TRAIN_only_design_no_model_training',parent=p['parent_checkpoint'],
        start_step=2000,stop_step=4000,seed=p['seed'],optimizer='all arms AdamW reset as P3 B0',
        C0='exact existing B0 recipe; historical B0 reusable only after code/exposure equivalence',
        C1='within foreign clean/mis strata, cap each item contribution at 1.5 times mean item mass; water-fill preserves stratum mass',
        C2='actual TAG (phone,flat surface position,left,right); inverse-square-root frequency within phone-position, clipped [0.5,2]; utterance geometric mean; stratum normalization',
        context_frequency='planned TRAIN continuation token occurrences only; no validation/test outcomes',
        native='same native draw schedule and weight 1 in every arm',
        loss='mean(weight_i * CTC_NLL_i / target_length_i); same microbatches; no batch-dependent renormalization',
        preserved='physical audio draws, samples, UID order, speaker/L1/clean-error sampling, steps and batch budget',
        not_preserved='weighted loss contribution by speaker/L1/duration/phone; report residual imbalance',
        interpretation='utterance weighting cannot independently control every token context; measure the achieved treatment before adopting',
        training_gate='no run until evaluation cohort, incomplete-label policy and independent-data scope are closed',
        primary_comparison='C2 4k vs C0 4k; C1 mechanism control; no candidate/checkpoint replacement',
        scoring='freeze seen CAL 1% thresholds; require measured seen/unseen FPR <=1%, exact diagnosis improvement, and prior deletion/insertion/phone protection',
        readiness_root=a.readiness,bindings={str(Path(a.p3_protocol)):file_sha(a.p3_protocol),
            str(Path(__file__)):file_sha(__file__)})
    write(out/'design.json',design)
    rows,labels=read_split(str(manifest),'train');sources=(manifest/'train.source').read_text().splitlines()
    batches=stream(rows,p['batch_samples'],p['seed']);h=hashlib.sha256();draws=samples=0
    counts=collections.Counter();coeff=collections.Counter();receipts=[]
    for micro in range(p['stop_step']*p['accum']):
        ids=next(batches);h.update((' '.join(map(str,ids))+'\n').encode())
        draws+=len(ids);samples+=sum(rows[i][1] for i in ids)
        if micro>=p['start_step']*p['accum']:
            counts.update(ids)
            for i in ids:coeff[i]+=1./len(ids)
        if micro+1 in (p['start_step']*p['accum'],p['stop_step']*p['accum']):
            receipt=dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest())
            expected=p['parent_exposure'] if len(receipts)==0 else p['expected_final_exposure']
            if receipt!=expected:raise ValueError('stream mismatch')
            receipts.append(receipt)
    ids=sorted(counts);metadata={};context_counts=collections.Counter();mass=collections.Counter()
    for i in ids:
        source=sources[i];uid=Path(rows[i][0]).stem;parts=uid.split('-');foreign=source.startswith('foreign')
        metadata[i]=dict(uid=uid,source=source,item=parts[5] if foreign else '<native>',
            speaker=parts[0] if foreign else '<native_unavailable>',l1=parts[3] if foreign else '<native>')
        if foreign:
            mass[source,parts[5]]+=coeff[i]
            for c in contexts(labels[i]):context_counts[c]+=counts[i]
    weights={arm:np.ones(len(rows),dtype=np.float64) for arm in ('C0','C1','C2')};caps={}
    for source in sorted(set(sources)-{'native'}):
        keys=sorted(k for k in mass if k[0]==source);values=[mass[k] for k in keys]
        target,cap=capped_mass(values,1.5);caps[source]=cap
        factors={k:float(v/mass[k]) for k,v in zip(keys,target)}
        for i in ids:
            m=metadata[i]
            if m['source']==source:weights['C1'][i]=factors[source,m['item']]
    groups=collections.defaultdict(list)
    for c,n in context_counts.items():groups[c[:2]].append(n)
    avg={k:float(np.mean(v)) for k,v in groups.items()}
    cf={c:float(np.clip(math.sqrt(avg[c[:2]]/n),.5,2.)) for c,n in context_counts.items()}
    for i in ids:
        if sources[i].startswith('foreign'):
            cs=contexts(labels[i]);weights['C2'][i]=math.exp(sum(math.log(cf[c]) for c in cs)/len(cs))
    for source in sorted(set(sources)-{'native'}):
        ix=[i for i in ids if sources[i]==source];base=sum(coeff[i] for i in ix)
        factor=base/sum(coeff[i]*weights['C2'][i] for i in ix)
        weights['C2'][ix]*=factor
    # Hash and save weights by manifest row, without copying audio or model states.
    for arm,w in weights.items():np.save(out/(arm+'_weights.npy'),w)
    aggregate=collections.defaultdict(collections.Counter);context_mass=collections.defaultdict(collections.Counter)
    records=[]
    for arm,w in weights.items():
        for i in ids:
            m=metadata[i];base=coeff[i]*float(w[i]);n=counts[i]
            for dim,key in [('source',m['source']),('item',m['item']),('speaker',m['speaker']),('l1',m['l1'])]:
                aggregate[arm,dim,key].update(dict(row_draws=n,audio_samples=n*rows[i][1],
                    ctc_coefficient=base,weighted_audio_samples=n*rows[i][1]*float(w[i])))
            if m['source'].startswith('foreign'):
                for c in contexts(labels[i]):
                    context_mass[arm,c].update(dict(token_draws=n,weighted_token_draws=n*float(w[i]),
                        ctc_coefficient=base))
        for source in set(sources):
            base=sum(coeff[i] for i in ids if sources[i]==source)
            after=aggregate[arm,'source',source]['ctc_coefficient']
            if abs(base-after)>1e-7*max(1,base):raise ValueError('source loss mass not preserved')
        records.append(dict(arm=arm,min_weight=float(w[ids].min()),median_weight=float(np.median(w[ids])),
            max_weight=float(w[ids].max()),nonunit_rows=int(sum(abs(w[i]-1)>1e-12 for i in ids))))
    tsv(out/'contribution_groups.tsv',[dict(arm=k[0],dimension=k[1],key=k[2],**dict(v)) for k,v in sorted(aggregate.items())])
    tsv(out/'context_contributions.tsv',[dict(arm=k[0],phone=k[1][0],position=k[1][1],left=k[1][2],right=k[1][3],**dict(v))
        for k,v in sorted(context_mass.items())])
    prompt_rows=[]
    for (source,item),base in sorted(mass.items()):
        ix=[i for i in ids if metadata[i]['source']==source and metadata[i]['item']==item]
        prompt_rows.append(dict(source=source,item=item,row_draws=sum(counts[i] for i in ix),
            C0_mass=base,C1_mass=sum(coeff[i]*weights['C1'][i] for i in ix),
            C2_mass=sum(coeff[i]*weights['C2'][i] for i in ix),C1_cap=caps[source]))
    tsv(out/'prompt_caps.tsv',prompt_rows)
    # Explicitly quantify the manipulation, including pairs that motivated P4.
    changes=[]
    for arm in ('C0','C1','C2'):
        actual={c:v['weighted_token_draws'] for (a,c),v in context_mass.items() if a==arm}
        entropy=[]
        for key in groups:
            ns=np.array([v for c,v in actual.items() if c[:2]==key]);prob=ns/ns.sum()
            entropy.append(dict(phone=key[0],position=key[1],contexts=len(ns),
                effective_contexts=float(np.exp(-(prob*np.log(prob)).sum())),largest_share=float(prob.max())))
        changes.append(dict(arm=arm,phone_position_contexts=entropy,
            s_u_l_weighted_tokens=sum(v for c,v in actual.items() if (c[0],c[2],c[3])==('ㅜ','ㅅ','ㄹ')),
            s_o_l_weighted_tokens=sum(v for c,v in actual.items() if (c[0],c[2],c[3])==('ㅗ','ㅅ','ㄹ'))))
    write(out/'report.json',dict(status='weight_design_measured_not_trained',receipts=receipts,
        actual_continuation_rows=sum(counts.values()),actual_continuation_samples=sum(counts[i]*rows[i][1] for i in ids),
        actual_foreign_items=len({m['item'] for m in metadata.values() if m['source']!='native'}),
        foreign_context_cells=len(context_counts),weights=records,context_balance=changes,
        item_caps=caps,sori=[r for r in prompt_rows if r['item']=='RW052'],
        identical_physical_exposure=True,token_local_CTC_weighting=False,optimizer_updates=0,
        evaluation_gate_passed=False,training_authorized_by_readiness=False))
    write(out/'finished.json',dict(status='TRAIN_only_design_complete',optimizer_updates=0))


if __name__=='__main__':main()

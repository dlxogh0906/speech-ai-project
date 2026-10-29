"""P4-B/C: stored GOP cross-threshold audit and exact Stage2 exposure replay."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import hashlib
import json
import numpy as np
from train_h0_backbone_control import file_sha, stream
from train_w2vbert_ctc import read_split
from eval_h0_backbone_control import metrics, threshold_for_normal_budget
from diagnose_h0_p4 import read_jsonl, write_json
from fit_h0_representation_probes import table
from mdd.h0_phone_audit import surface_position


def calibration(root,p):
    dest=root/'calibration';dest.mkdir(exist_ok=False)
    ledgers={a:read_jsonl(Path(p['p3'])/(a+'_eval_4000')/'sites.jsonl') for a in p['arms']}
    if [[(s['uid'],s['q'],s['actual']) for s in ledgers[a]] for a in p['arms']][0] != [
            (s['uid'],s['q'],s['actual']) for s in ledgers['B3']]: raise ValueError('unpaired ledgers')
    taus={}; distribution=[];cross=[];decomposition=[]
    for a,ss in ledgers.items():
        cal=[s for s in ss if s['role']=='calibration' and s['split']=='valid']
        taus[a]=threshold_for_normal_budget(cal)
        old=json.loads((Path(p['p3'])/(a+'_eval_4000')/'report.json').read_text())
        if taus[a]!=old['threshold']:raise ValueError('CAL threshold reconstruction')
        groups=collections.defaultdict(list)
        for s in ss:
            if s['actual']==s['expected']:
                for dim,key in [('all','all'),('phone',s['expected']),('item',s['item'])]:
                    groups[s['split'],s['role'],dim,key].append(s)
        for (split,role,dim,key),g in groups.items():
            v=[s['score'] for s in g if s['score'] is not None and s['predicted']!=s['expected']]
            distribution.append(dict(arm=a,split=split,role=role,dimension=dim,key=key,normal=len(g),
                items=len({s['item'] for s in g}),mismatches=len(v),threshold=taus[a],
                quantiles=np.quantile(v,[0,.25,.5,.75,.9,.95,.99,1]).tolist() if v else None))
    for a,ss in ledgers.items():
        for tau_arm,tau in taus.items():
            for split in ('valid','valid_unseen'):
                g=[s for s in ss if s['role']=='selection' and s['split']==split]
                cross.append(dict(output_arm=a,threshold_arm=tau_arm,split=split,**metrics(g,tau)))
    old=[s for s in ledgers['B0'] if s['role']=='selection']
    new=[s for s in ledgers['B3'] if s['role']=='selection']
    groups=collections.defaultdict(list)
    for i,s in enumerate(old):
        for dim,key in [('all','all'),('phone',s['expected']),('item',s['item'])]:groups[s['split'],dim,key].append(i)
    for (split,dim,key),ids in groups.items():
        x=[old[i] for i in ids];y=[new[i] for i in ids]
        aa=metrics(x,taus['B0']);ba=metrics(y,taus['B0']);bb=metrics(y,taus['B3']);ab=metrics(x,taus['B3'])
        decomposition.append(dict(split=split,dimension=dim,key=key,normal=aa['normal'],errors=aa['errors'],
            B0_B0=aa['normal_false_alarms'],B3_B0=ba['normal_false_alarms'],B3_B3=bb['normal_false_alarms'],
            B0_B3=ab['normal_false_alarms'],output_score_change=ba['normal_false_alarms']-aa['normal_false_alarms'],
            threshold_change=bb['normal_false_alarms']-ba['normal_false_alarms'],
            reverse_order_threshold_change=ab['normal_false_alarms']-aa['normal_false_alarms'],
            reverse_order_output_change=bb['normal_false_alarms']-ab['normal_false_alarms'],
            exact_B0=aa['exact'],exact_B3_fixed_B0_tau=ba['exact'],exact_B3=bb['exact']))
    table(dest/'score_distributions.tsv',distribution);table(dest/'cross_threshold.tsv',cross)
    table(dest/'decomposition.tsv',decomposition)
    write_json(dest/'report.json',dict(thresholds=taus,cross=cross,all_decomposition=[r for r in decomposition if r['dimension']=='all'],
        causal_decomposition=False,labels_changed=False,threshold_retuned_on_selection=False,
        source_sha256=file_sha(__file__)))


def exposure(root,p):
    dest=root/'exposure';dest.mkdir(exist_ok=False)
    prior=json.loads((Path(p['p3'])/'design_v2/protocol.json').read_text())
    manifest=Path(p['manifest'])
    for fn in ('train.tsv','train.ltr','train.source'):
        if file_sha(manifest/fn)!=prior['bindings'][str(manifest/fn)]: raise ValueError('training data changed')
    rows,labels=read_split(str(manifest),'train');sources=(manifest/'train.source').read_text().splitlines()
    batches=stream(rows,prior['batch_samples'],prior['seed']);h=hashlib.sha256();draws=samples=0
    exposure=[collections.Counter(),collections.Counter()];receipts=[]
    for i in range(prior['stop_step']*prior['accum']):
        ids=next(batches);phase=int(i>=prior['start_step']*prior['accum'])
        exposure[phase].update(ids);h.update((' '.join(map(str,ids))+'\n').encode())
        draws+=len(ids);samples+=sum(rows[j][1] for j in ids)
        if i+1 in (prior['start_step']*prior['accum'],prior['stop_step']*prior['accum']):
            observed=dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest())
            expected=prior['parent_exposure'] if phase==0 else prior['expected_final_exposure']
            if observed!=expected: raise ValueError('actual exposure stream mismatch')
            receipts.append(observed)
    for arm in p['arms']:
        finished=json.loads((Path(p['p3'])/arm/'finished.json').read_text())['final_exposure']
        if any(finished[k]!=receipts[-1][k] for k in receipts[-1]):raise ValueError('training receipt mismatch')
    counts=collections.Counter();unique=collections.defaultdict(set);allphone=collections.Counter();drawsource=collections.Counter()
    for phase,row_counts in zip(('steps_1_2000','steps_2001_4000'),exposure):
        for idx,multiplicity in row_counts.items():
            uid=Path(rows[idx][0]).stem;source=sources[idx];parts=uid.split('-');foreign=source.startswith('foreign')
            item=parts[5] if foreign else '<native>';speaker=parts[0] if foreign else '<unknown_native_speaker>'
            target=labels[idx];drawsource[phase,source]+=multiplicity
            allphone.update({(phase,source,t):n*multiplicity for t,n in collections.Counter(target).items()})
            for j,t in enumerate(target):
                if t not in ('ㅜ','ㅗ'):continue
                left=target[j-1] if j else '<BOS>';right=target[j+1] if j+1<len(target) else '<EOS>'
                context=left+'_'+right
                for dimension,key in [('all','all'),('item',item),('speaker',speaker),('context',context),
                                      ('item_context',item+'|'+context),('position',surface_position(target,j))]:
                    k=(phase,source,t,dimension,key);counts[k]+=multiplicity;unique[k].add((uid,j))
    table(dest/'phone_context_exposure.tsv',[dict(phase=k[0],source=k[1],phone=k[2],dimension=k[3],key=k[4],
        token_draws=v,unique_recording_positions=len(unique[k])) for k,v in sorted(counts.items())])
    table(dest/'all_phone_draws.tsv',[dict(phase=k[0],source=k[1],phone=k[2],token_draws=v) for k,v in sorted(allphone.items())])
    concentration=[]
    for phase in ('steps_1_2000','steps_2001_4000'):
        for phone in ('ㅜ','ㅗ'):
            byitem=collections.Counter();bycontext=collections.Counter();suri_context=collections.Counter()
            for (ph,src,t,dim,key),n in counts.items():
                if ph==phase and t==phone and src.startswith('foreign'):
                    if dim=='item':byitem[key]+=n
                    if dim=='context':bycontext[key]+=n
                    if dim=='item_context' and key.endswith('|ㅅ_ㄹ'):suri_context[key.split('|')[0]]+=n
            total=sum(byitem.values());weights=np.array(list(byitem.values()),float)/total
            concentration.append(dict(phase=phase,phone=phone,foreign_token_draws=total,items=len(byitem),
                top5_items=byitem.most_common(5),top5_share=sum(n for _,n in byitem.most_common(5))/total,
                effective_items=1/float((weights**2).sum()),top10_contexts=bycontext.most_common(10),
                s_phone_l_context_draws=sum(suri_context.values()),s_phone_l_items=dict(suri_context),
                heldout_RW054_draws=byitem['RW054']))
    write_json(dest/'report.json',dict(stream_receipts=receipts,unique_manifest_rows=[len(c) for c in exposure],
        source_draws=[dict(phase=k[0],source=k[1],row_draws=v) for k,v in drawsource.items()],
        concentration=concentration,source_sha256=file_sha(__file__),
        limitations=['Stage2 exact deterministic batch replay validated against optimizer receipts; Stage1 actual exposure not newly audited',
            'counts include repeated rows; unique recording positions reported separately',
            'foreign targets are existing TAG; native targets are transcript/G2P, not independent actual-phone TAG',
            'no native speaker metadata inferred from filename; no causal exposure-effect claim']))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True)
    a=ap.parse_args();root=Path(a.root);p=json.loads((root/'protocol.json').read_text())
    calibration(root,p);exposure(root,p)
    write_json(root/'controls_finished.json',dict(status='completed',optimizer_updates=0))


if __name__=='__main__':main()

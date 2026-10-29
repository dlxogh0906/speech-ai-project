"""Frozen B0 long-form RP stress evaluation; never trains or tunes on RP."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import concurrent.futures
import csv
import datetime
import gzip
import hashlib
import itertools
import json
import time
import wave
import numpy as np
from mdd.h0_partial_reference import UNKNOWN, DELETED, input_windows
from mdd.h0_partial_scoring import (mapping_indices, value_options, gap_value,
    annotation_mask, unique_index, partial_ctc_viterbi, metric_record, rates)
from audit_h0_p5_readiness import write, sha


def emit(f, x): f.write(json.dumps(x, ensure_ascii=False)+'\n')


def read_audio(path):
    with wave.open(str(path),'rb') as w:
        assert (w.getframerate(),w.getnchannels(),w.getsampwidth()) == (16000,1,2)
        pcm=w.readframes(w.getnframes())
    return np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768., hashlib.sha256(pcm).hexdigest()


def prepare(args):
    out=args.out; out.mkdir(parents=True,exist_ok=False)
    parent=args.partial
    p3=json.loads(args.p3_protocol.read_text())
    baseline=json.loads(args.baseline_report.read_text())
    assert baseline['condition']=='B0' and baseline['step']==4000
    assert sha(args.checkpoint)==baseline['checkpoint_sha256']
    files=[Path(__file__).resolve(), ROOT/'tasks/pronunciation-assessment/src/mdd/h0_partial_scoring.py',
           ROOT/'tasks/pronunciation-assessment/src/mdd/h0_partial_reference.py']
    protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),
        role='secondary_three_script_long_form_stress_not_model_selection_or_independent_final',
        parent_partial=str(parent),p3_protocol=str(args.p3_protocol),source=p3['source'],
        checkpoint=str(args.checkpoint),baseline_report=str(args.baseline_report),
        threshold=baseline['threshold'],threshold_origin='unchanged P3 B0 4k seen CAL; no RP fitting',
        manifest=p3['manifest'],candidate_policy='all previously reserved 1118 UIDs, no exclusions',
        input='12s cores, 400ms halos; all samples/tails, frame centers 280+320*i samples',
        canonical='original LAB; unknown spans retain their positions and accept zero or more phones',
        truth='all optimal partial edit paths, fixed known actual values; no repaired labels',
        annotation_boundaries='adjacent unique known target occurrences; deletions need two real anchors',
        model_observation='unique predicted phone, available GOP, anchored deletions; report missing predictions',
        primary_seam_band_samples=640,sensitivity_seam_bands_samples=[0,6400],
        GOP='ordinary CTC Viterbi if fully known; exact wildcard-language CTC Viterbi otherwise; same margin',
        gap_policy='internal gaps only, unique known bounding occurrences in both truth and prediction; greedy only',
        selection='no condition, model, threshold, seam band or wildcard rule selection from RP results',
        uncertainty='speaker clustered 2000 bootstrap conditional on model/threshold; report each of only 3 prompts',
        model_training=False,independent_listening=False,production_changes=False,
        input_bindings={str(p):sha(p) for p in (args.p3_protocol,args.baseline_report,args.checkpoint,
            parent/'annotation_records.jsonl.gz',parent/'inputs.jsonl',parent/'verification.json',
            args.readiness/'reserved_unscored_candidates.jsonl')},
        source_bindings={str(p):sha(p) for p in files},readiness=str(args.readiness))
    write(out/'protocol.json',protocol)
    counts=collections.defaultdict(collections.Counter)
    with gzip.open(str(parent/'annotation_records.jsonl.gz'),'rt') as src, gzip.open(str(out/'annotations.jsonl.gz'),'wt') as dest:
        for r in map(json.loads,src):
            idx=mapping_indices(r['reference'],r['actual'])
            assert [None if x is None else sorted(x) for x in (value_options(v,r['actual']) for v in idx)] == [s['possible_actual'] for s in r['sites']]
            mask=annotation_mask(r,idx)
            r['annotation_mask']=mask
            r['truth_gaps']=[gap_value(r['reference'],r['actual'],idx,g) for g in range(len(idx)+1)]
            emit(dest,r)
            for group in ('all',r['item']):
                c=counts[group]; c['recordings']+=1
                for s,why in zip(r['sites'],mask):
                    c['known_reference']+=int(s['expected']!=UNKNOWN)
                    c[why or ('eligible_'+s['status'])]+=1
                c['gap_candidates_including_unknown_reference_entries']+=len(r['truth_gaps'])
                c['identified_normal_gaps']+=sum(g==() for g in r['truth_gaps'])
                c['identified_insertion_gaps']+=sum(bool(g) for g in r['truth_gaps'])
    write(out/'annotation_support.json',dict(counts))
    write(out/'frozen.json',dict(protocol_sha256=sha(out/'protocol.json'),annotations_sha256=sha(out/'annotations.jsonl.gz'),
        counts_sha256=sha(out/'annotation_support.json'),model_scored=False))


def load_model(p):
    import torch
    from train_h0_intermediate import P3Model
    assert sha(p['checkpoint']) == p['input_bindings'][p['checkpoint']]
    ck=torch.load(p['checkpoint'],map_location='cpu')
    assert ck['condition']=='B0' and ck['step']==4000
    model=P3Model(p['source'],ck['vocab'],torch.device('cuda' if torch.cuda.is_available() else 'cpu'),'B0')
    model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,ck['vocab']


def forward(model,x):
    import torch
    with torch.inference_mode(),torch.cuda.amp.autocast(enabled=model.device.type=='cuda'):
        logits,lengths=model([x])
    lp=logits[0,:int(lengths[0])].float().log_softmax(-1).cpu().numpy()
    assert np.isfinite(lp).all()
    return lp


def stitch(model,audio):
    pieces=[];times=[];logs=[]
    for w in input_windows(len(audio)):
        x=audio[w['read_start']:w['read_end']]
        lp=forward(model,x)
        center=w['read_start']+280+320*np.arange(len(lp))
        keep=(center>=w['core_start'])&(center<w['core_end'])
        pieces.append(lp[keep]);times.extend(center[keep].tolist())
        logs.append(dict(w,frontend_frames=len(lp),owned_frames=int(keep.sum())))
    assert all(a<b for a,b in zip(times,times[1:]))
    assert all((t-280)%320==0 for t in times)
    # The final <20ms may have no frame; keep input samples and report the tail.
    if len(times)>1: assert max(b-a for a,b in zip(times,times[1:]))==320
    return np.concatenate(pieces),np.array(times,np.int64),logs


def check(args,p):
    from mdd.gop_feats_v2 import viterbi_align
    model,vocab=load_model(p);inv={v:k for k,v in vocab.items()}
    root=Path(p['baseline_report']).parent
    by_uid=collections.defaultdict(list)
    for line in (root/'sites.jsonl').open():
        s=json.loads(line)
        if s['role']=='calibration':by_uid[s['uid']].append(s)
    manifest=Path(p['manifest']);lines=(manifest/'valid.tsv').read_text().splitlines();audio_root=Path(lines[0])
    candidates=[]
    for line in lines[1:]:
        name,n=line.split('\t');path=audio_root/name;uid=path.stem
        if uid in by_uid and 16000<=int(n)<=192000:candidates.append((hashlib.sha256(uid.encode()).hexdigest(),uid,path))
    assert len(candidates)>=4
    checks=[]
    for _,uid,path in sorted(candidates)[:4]:
        audio,_=read_audio(path);direct=forward(model,audio);lp,times,windows=stitch(model,audio)
        np.testing.assert_array_equal(lp,direct[:len(lp)])
        # Full direct result must also be retained for these ordinary short inputs.
        assert len(direct)==len(lp)
        pred=[inv[t] for t,_ in itertools.groupby(lp.argmax(1)) if t in inv]
        sites=sorted(by_uid[uid],key=lambda s:s['q']);ref=[s['expected'] for s in sites]
        seg=viterbi_align(lp,[vocab[c] for c in ref],blank=0)
        pi=mapping_indices(ref,pred)
        from eval_actual_ctc import project_on_canon
        single=project_on_canon(ref,pred)
        diffs=[]
        for i,s in enumerate(sites):
            block=lp[seg[i]];other=block.copy();other[:,vocab[ref[i]]]=-np.inf
            score=float((other.max(1)-block[:,vocab[ref[i]]]).mean())
            assert single[i]==s['predicted'];diffs.append(abs(score-s['score']))
        assert max(diffs)==0
        checks.append(dict(uid=uid,sites=len(sites),frames=len(lp),logit_difference=0,gop_difference=0))
    write(args.out/'runtime_check.json',dict(status='passed',short_audio_checks=checks,
          evidence='native and core-stitched outputs/GOP match original P3 baseline on four short recordings',
          limitation='chunked long recordings are not claimed equivalent to full-context inference',optimizer_updates=0))


def extract(args,p):
    assert json.loads((args.out/'runtime_check.json').read_text())['status']=='passed'
    lpdir=args.out/'log_probs';lpdir.mkdir(exist_ok=False)
    model,vocab=load_model(p);start=time.time();seen=[];seq_hash=hashlib.sha256();total_frames=0
    candidates=[json.loads(x) for x in (Path(p['readiness'])/'reserved_unscored_candidates.jsonl').open()]
    with (args.out/'inference.jsonl').open('x') as ledger:
        for i,r in enumerate(candidates):
            x,digest=read_audio(r['path']);assert len(x)==r['frames'] and digest==r['pcm_sha256']
            lp,centers,windows=stitch(model,x)
            np.savez_compressed(lpdir/(r['uid']+'.npz'),lp=lp,centers=centers)
            seq_hash.update((r['uid']+':'+digest+'\n').encode());seen.append(r['uid']);total_frames+=len(lp)
            emit(ledger,dict(uid=r['uid'],frames=len(lp),audio_samples=len(x),windows=windows,
                last_center_to_audio_end_samples=int(len(x)-centers[-1]),pcm_sha256=digest))
            if (i+1)%50==0:
                ledger.flush();status=dict(status='fixed_B0_RP_inference',recordings=i+1,total=len(candidates),elapsed_seconds=time.time()-start)
                write(args.out/'live_status.json',status);print(json.dumps(status),flush=True)
    write(args.out/'extraction_finished.json',dict(recordings=len(seen),total_frames=total_frames,
        audio_order_sha256=seq_hash.hexdigest(),vocab=vocab,elapsed_seconds=time.time()-start,
        model_updates=0,protocol_sha256=sha(args.out/'protocol.json')))


def score_one(job):
    r,lpdir,vocab=job
    from mdd.gop_feats_v2 import viterbi_align
    data=np.load(Path(lpdir)/(r['uid']+'.npz'));lp=data['lp'];centers=data['centers'];data.close()
    inv={v:k for k,v in vocab.items()};pred=[inv[t] for t,_ in itertools.groupby(lp.argmax(1)) if t in inv]
    reference=r['reference'];idx=mapping_indices(reference,pred)
    ids=[None if x==UNKNOWN else vocab[x] for x in reference]
    if None in ids:
        aligned=partial_ctc_viterbi(lp,ids,list(vocab.values()))
        segments=None if aligned is None else aligned['segments']
    else: segments=viterbi_align(lp,ids,blank=0)
    # Input seams are known without either annotation's times.
    last=int(centers[-1]);seams=np.arange(192000,last+1,192000)
    sites=[]
    for q,s in enumerate(r['sites']):
        frames=[] if segments is None else segments[q]
        score=None;distance=10**12
        if ids[q] is not None and frames:
            block=lp[frames];other=block.copy();other[:,ids[q]]=-np.inf
            score=float((other.max(1)-block[:,ids[q]]).mean())
            if len(seams):distance=int(np.abs(centers[frames,None]-seams[None,:]).min())
        opts=value_options(idx[q],pred)
        anchored=(q>0 and q+1<len(idx) and unique_index(idx[q-1]) is not None and unique_index(idx[q+1]) is not None)
        sites.append(dict(q=q,expected=s['expected'],actual=s['identified_actual'],
            annotation_exclusion=r['annotation_mask'][q],prediction_options=[] if opts is None else sorted(opts),
            score=score,nearest_seam_samples=distance,prediction_deletion_anchored=anchored))
    gaps=[]
    for g,truth in enumerate(r['truth_gaps']):
        guess=gap_value(reference,pred,idx,g)
        dist=min(sites[g-1]['nearest_seam_samples'],sites[g]['nearest_seam_samples']) if 0<g<len(sites) else 0
        gaps.append(dict(g=g,actual=truth,predicted=None if guess is None else list(guess),nearest_seam_samples=dist))
    return dict(uid=r['uid'],speaker=r['speaker'],item=r['item'],predicted=pred,sites=sites,gaps=gaps,
        alignment_available=segments is not None,source_timing_flags=r['source_timing_flags'])


def gap_counts(gaps,band):
    c=collections.Counter()
    for g in gaps:
        c['gap_entries']+=1
        if g['actual'] is None:c['annotation_unknown_gap']+=1;continue
        error=bool(g['actual']);k='error' if error else 'normal';c[k]+=1
        if g['predicted'] is None or g['nearest_seam_samples']<=band:
            c['unobserved_'+k]+=1;c['exact_upper']+=error;c['fp_upper']+=not error;continue
        c['observed_'+k]+=1
        fp=not error and bool(g['predicted']);exact=error and g['actual']==g['predicted']
        c['fp']+=fp;c['exact']+=exact;c['fp_upper']+=fp;c['exact_upper']+=exact
    return dict(c)


def bootstrap(records,threshold,band):
    by=collections.defaultdict(collections.Counter)
    for r in records:by[r['speaker']].update(metric_record(r['sites'],threshold,band))
    keys=['normal','error','observed_normal','observed_error','fp','exact','fp_upper','exact_upper']
    x=np.array([[v.get(k,0) for k in keys] for _,v in sorted(by.items())],dtype=float)
    rng=np.random.RandomState(20261029);draws=rng.randint(len(x),size=(2000,len(x)));totals=x[draws].sum(1)
    result={}
    for name,num,den in [('observed_fpr',4,2),('observed_recall',5,3),('fixed_fpr_lower',4,0),('fixed_fpr_upper',6,0),('fixed_recall_lower',5,1)]:
        valid=totals[:,den]>0
        result[name]=np.quantile(totals[valid,num]/totals[valid,den],[.025,.975]).tolist() if valid.any() else None
    return dict(speakers=len(by),replicates=2000,ci95=result,scope='fixed three prompts, model, annotation mask and seen CAL threshold')


def score(args,p):
    vocab=json.loads((args.out/'extraction_finished.json').read_text())['vocab']
    records=[];start=time.time()
    with gzip.open(str(args.out/'annotations.jsonl.gz'),'rt') as f:
        annotations=list(map(json.loads,f))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool, gzip.open(str(args.out/'evaluated.jsonl.gz'),'wt') as out:
        jobs=((r,str(args.out/'log_probs'),vocab) for r in annotations)
        for i,r in enumerate(pool.map(score_one,jobs,chunksize=1),1):
            emit(out,r);records.append(r)
            if i%100==0:
                status=dict(status='partial_reference_scoring',recordings=i,total=len(annotations),elapsed_seconds=time.time()-start)
                write(args.out/'live_status.json',status);print(json.dumps(status),flush=True)
    assert len(records)==len(annotations)
    grouped=collections.defaultdict(collections.Counter);gapgroups=collections.defaultdict(collections.Counter)
    for r in records:
        for band in [p['primary_seam_band_samples']]+p['sensitivity_seam_bands_samples']:
            for policy,threshold in [('greedy',None),('seen_CAL_fixed',p['threshold'])]:
                for dimension,key,sites in [('all','all',r['sites']),('item',r['item'],r['sites'])]+[
                    ('phone',phone,[s for s in r['sites'] if s['expected']==phone]) for phone in vocab]:
                    grouped[(band,policy,dimension,key)].update(metric_record(sites,threshold,band))
            for key in ('all',r['item']):gapgroups[(band,key)].update(gap_counts(r['gaps'],band))
    rows=[]
    for k,c in grouped.items():rows.append(dict(band_samples=k[0],policy=k[1],dimension=k[2],key=k[3],counts=dict(c),rates=rates(c)))
    report=dict(status='fixed_B0_three_prompt_stress_evaluated',recordings=len(records),
        speakers=len({r['speaker'] for r in records}),prompts=len({r['item'] for r in records}),
        threshold=p['threshold'],alignment_failures=sum(not r['alignment_available'] for r in records),
        site_results=rows,gap_results=[dict(band_samples=k[0],item=k[1],counts=dict(c),
            observed_fpr=c['fp']/c['observed_normal'] if c['observed_normal'] else None,
            observed_exact_recall=c['exact']/c['observed_error'] if c['observed_error'] else None)
            for k,c in gapgroups.items()],
        speaker_bootstrap=bootstrap(records,p['threshold'],p['primary_seam_band_samples']),
        optimizer_updates=0,RP_threshold_fits=0,production_changes=False,
        limitations=['three scripts; historically evaluated speakers, different recordings',
          'original TAG conditional truth, no independent listening',
          'masked/unknown outcomes are not normal; observed rates are conditional on coverage',
          'wildcard CTC is a partial-reference extension; fully known CTC uses existing alignment',
          'chunked model context differs from full waveform inference; seam masks do not prove boundary correctness',
          'gap scoring is greedy and not a production insertion policy'])
    write(args.out/'report.json',report)
    with (args.out/'metrics.tsv').open('w') as f:
        fields=['band_samples','policy','dimension','key']+list(rates({}).keys())
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader()
        for r in rows:writer.writerow(dict({k:r[k] for k in fields[:4]},**r['rates']))
    write(args.out/'finished.json',dict(status=report['status'],recordings=len(records),optimizer_updates=0,
        protocol_sha256=sha(args.out/'protocol.json'),report_sha256=sha(args.out/'report.json'),
        evaluated_sha256=sha(args.out/'evaluated.jsonl.gz')))
    write(args.out/'live_status.json',dict(status=report['status'],recordings=len(records),optimizer_updates=0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['prepare','check','extract','score'],required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--partial',type=Path)
    ap.add_argument('--readiness',type=Path);ap.add_argument('--p3-protocol',type=Path)
    ap.add_argument('--baseline-report',type=Path);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.mode=='prepare':prepare(a);return
    p=json.loads((a.out/'protocol.json').read_text());frozen=json.loads((a.out/'frozen.json').read_text())
    assert sha(a.out/'protocol.json')==frozen['protocol_sha256']
    assert sha(a.out/'annotations.jsonl.gz')==frozen['annotations_sha256']
    for path,digest in p['source_bindings'].items():assert sha(path)==digest,path
    {'check':check,'extract':extract,'score':score}[a.mode](a,p)


if __name__=='__main__':main()

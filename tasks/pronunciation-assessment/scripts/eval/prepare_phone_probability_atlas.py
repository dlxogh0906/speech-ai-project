"""Replay frozen M3 failures and retain exact-frame alternative probabilities."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
from shared.paths import DATA_ROOT, MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)
import argparse
import gzip
import hashlib
import json
import time
from collections import defaultdict,Counter
import numpy as np
import torch
from mdd.phone_atlas import sha,read_audio
from mdd.phone_probability_atlas import greedy_runs,ranked_frame
from prepare_phone_atlas import normalized_timing,write_json,write_csv
from eval_canonical_source import a2b


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--atlas',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False); (args.out/'log_probs').mkdir()
    torch.set_num_threads(2)
    atlas=json.loads((args.atlas/'report.json').read_text())
    weak=sorted([m for m in atlas['actual_phone_metrics'] if m['error_n']>=20 and m['error_accuracy_pct']<40],
                key=lambda m:(m['error_accuracy_pct'],m['phone']))
    phones={m['phone'] for m in weak}
    with gzip.open(args.atlas/'sites.jsonl.gz','rt',encoding='utf-8') as f:
        all_sites=[s for s in map(json.loads,f) if s['accepted']]
    by_uid=defaultdict(list); targets=defaultdict(list)
    for s in all_sites:
        by_uid[s['uid']].append(s)
        if s['actual'] in phones and s['actual']!=s['expected'] and s['actual']!=s['predicted']:
            targets[s['uid']].append(s)
    manifest=MDD_OUTPUTS/'manifest_full_tag_replay10'
    checkpoint=DATA_ROOT/'checkpoints/w2vbert_synth_tense_v2/step_3000'
    protocol=dict(model=str(checkpoint),training_updates=0,selection='accepted; actual phone error_n>=20 and error transcription<40%; LAB!=TAG and saved prediction!=TAG; all qualifying sites',
        selected_phones=[m['phone'] for m in weak],target_sites=sum(map(len,targets.values())),target_recordings=len(targets),
        anchor='Within the greedy run aligned to this canonical site, choose the frame maximizing the predicted token probability; no TAG-dependent frame selection.',
        rank='Competition rank among 46 jamo; also retain full-50-token rank and blank probability. Display top5 jamo with original full-softmax probabilities, without renormalizing.',
        deletion='No emitted token run, therefore no output-peak ranking. Retain as missing, do not invent a frame or probability.',
        reproduction='Require the whole utterance canonical-to-LAB projection, output jamo, and rounded3 peak probabilities to match the saved evaluation before using any site.',
        input='Original PCM16 -> float32 -> existing VAD trim -> GPU FP16 M3 -> float32 log_softmax; one utterance per forward.',
        sources_sha256={str(p):sha(p) for p in [args.atlas/'sites.jsonl.gz',args.atlas/'report.json',manifest/'test.tsv',manifest/'test.meta.jsonl']},
        checkpoint_sha256={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},
        limitations=['One representative frame, not a whole-phone posterior, CTC sequence probability, or calibrated correctness probability.',
                     'Input-rejected/unmapped positions outside cohort; training prompts, held-out recordings.',
                     'TAG was not independently listened to. Single edit path may be ambiguous.',
                     'Scope is incorrect pronunciations missed by transcription; correctly transcribed but display-suppressed errors are not included.'])
    write_json(args.out/'protocol.json',protocol)
    metas={}; paths={}
    with (manifest/'test.meta.jsonl').open() as f:
        for line in f:
            m=json.loads(line)
            if m['stem'] in targets:metas[m['stem']]=m
    with (manifest/'test.tsv').open() as f:
        root=Path(next(f).strip())
        for line in f:
            rel,n=line.rstrip().split('\t'); uid=Path(rel).stem
            if uid in targets:paths[uid]=root/rel
    assert set(metas)==set(paths)==set(targets)
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf
    from mdd.gop_feats_v2 import pred_probs
    emu=ApiEmulator(str(checkpoint),scorer=None);builder=cf.CanonicalBuilder(emu.canonical)
    vocab=[emu.dec.d[i] for i in range(len(emu.dec.d))]
    write_json(args.out/'vocab.json',dict(tokens=vocab,blank=emu.dec.blank))
    result=[]; utterances=[];started=time.monotonic()
    for n,uid in enumerate(sorted(targets),1):
        m=metas[uid]; x,sr,_=read_audio(paths[uid]); lp,dur,vad=emu.lprobs(x.astype(np.float32))
        runs=greedy_runs(lp,vocab,emu.dec.blank);hyp=[r['token'] for r in runs]
        base,alts=builder.variants(m['prompt'])
        if base: canon,_=cf.choose(lp,base,alts,emu.dec.d.index,blank=emu.dec.blank,margin=1.)
        else: canon=emu.canonical(m['prompt'])
        aligned=emu.align(canon,hyp); probs=pred_probs(emu.dec,lp)
        assert len(runs)==len(probs) and np.allclose([r['probability'] for r in runs],probs,atol=1e-8)
        lab,_=normalized_timing(m['lab_phones']); mp=a2b(canon,lab)
        qmap=[q for q,k in enumerate(mp) if k is not None]
        now=[(lab[mp[q]],canon[q],aligned[q]['pred'],round(probs[aligned[q]['j']],3) if aligned[q]['j'] is not None else None) for q in qmap]
        saved=[(s['expected'],s['canonical'],s['predicted'],s['prediction_probability']) for s in by_uid[uid]]
        match=now==saved
        # Lossless float32 output cache permits an independent full-rank check.
        np.savez_compressed(args.out/'log_probs'/(uid+'.npz'),log_probs=lp)
        utterance=dict(uid=uid,matched=match,frames=len(lp),greedy=hyp,canonical=canon,
            runtime_indices=qmap,processed_audio_seconds=dur,vad=vad,audio_sha256=sha(paths[uid]),
            cache_sha256=sha(args.out/'log_probs'/(uid+'.npz')))
        if not match:utterance.update(recomputed=now,saved=saved)
        utterances.append(utterance)
        for s in targets[uid]:
            row=dict(s);row.update(key=uid+':'+str(s['mapped_index']),prompt=m['prompt'],
                                   audio_path=str(paths[uid]),frame_axis='VAD-processed grid; not original recording seconds')
            if not match: row['status']='reproduction_mismatch'
            elif s['predicted'] is None: row['status']='deleted_no_emitted_token'
            else:
                q=qmap[s['mapped_index']];j=aligned[q]['j'];run=runs[j]
                assert run['token']==s['predicted']
                row.update(status='ranked',runtime_index=q,greedy_index=j,run=run,
                           **ranked_frame(lp[run['peak']],vocab,s['actual'],emu.dec.blank))
                row['actual_rank_min_in_run']=min(ranked_frame(lp[t],vocab,s['actual'],emu.dec.blank)['actual_rank_jamo'] for t in range(run['start'],run['end']))
                row['actual_peak_probability_in_run']=float(np.exp(lp[run['start']:run['end'],vocab.index(s['actual'])]).max())
            result.append(row)
        if n%50==0 or n==len(targets):
            print('%d/%d recordings, %.1fs, reproduced %d' % (n,len(targets),time.monotonic()-started,sum(u['matched'] for u in utterances)),flush=True)
    assert len(result)==protocol['target_sites']
    summary=[]; examples=[]
    for m in weak:
        rows=[r for r in result if r['actual']==m['phone']]; ranked=[r for r in rows if r['status']=='ranked']
        r=dict(phone=m['phone'],error_n=m['error_n'],error_correct=m['error_correct'],error_accuracy_pct=m['error_accuracy_pct'],
            misses=len(rows),ranked=len(ranked),deleted=sum(r['status']=='deleted_no_emitted_token' for r in rows),
            unreproduced=sum(r['status']=='reproduction_mismatch' for r in rows),
            rank2=sum(r['actual_rank_jamo']==2 for r in ranked),rank_le3=sum(r['actual_rank_jamo']<=3 for r in ranked),
            median_rank=float(np.median([r['actual_rank_jamo'] for r in ranked])) if ranked else None,
            median_actual_probability=float(np.median([r['actual_probability'] for r in ranked])) if ranked else None)
        summary.append(r)
        if ranked:examples.append(min(ranked,key=lambda r:hashlib.sha256(('rank-example-v1:'+r['key']).encode()).hexdigest())['key'])
    ss_cases=json.loads((args.atlas/'sibilant_context_v1/cases.json').read_text())
    old_ids={'5ef1f8738137','c2b5795d915f','d308d8549211'}
    old_keys=[r['uid']+':'+str(r['saved_mapped_index']) for r in ss_cases if r['case_id'] in old_ids]
    report=dict(summary=summary,target_sites=len(result),recordings=len(utterances),
        reproduced_recordings=sum(u['matched'] for u in utterances),status_counts=dict(Counter(r['status'] for r in result)),
        example_keys=examples,previous_ss_case_keys=old_keys,training_updates=0,model_inference_runs=len(utterances),
        elapsed_seconds=time.monotonic()-started,source_code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'tasks/pronunciation-assessment/src/mdd/phone_probability_atlas.py',ROOT/'tasks/pronunciation-assessment/src/mdd/gop_feats_v2.py',ROOT/'tasks/pronunciation-assessment/src/mdd/api_offline.py',ROOT/'tasks/pronunciation-assessment/src/mdd/canonical_fix.py',ROOT/'tasks/pronunciation-assessment/src/mdd/vad.py']})
    write_json(args.out/'cases.json',result);write_json(args.out/'utterances.json',utterances);write_json(args.out/'report.json',report)
    write_csv(args.out/'summary.csv',summary)
    flat=[]
    for r in result:
        row={k:r[k] for k in ('key','prompt','expected','actual','predicted','warning','status')}
        row.update(actual_rank=None,actual_probability_pct=None,blank_probability_pct=None,top1=None,top2=None,top3=None,top4=None,top5=None)
        if r['status']=='ranked':
            row.update(actual_rank=r['actual_rank_jamo'],actual_probability_pct=100*r['actual_probability'],blank_probability_pct=100*r['blank_probability'])
            row.update({'top%d'%(i+1):'%s %.6f%%'%(p['phone'],100*p['probability']) for i,p in enumerate(r['top_phones'])})
        flat.append(row)
    write_csv(args.out/'사례별_확률순위.csv',flat)
    write_json(args.out/'finished.json',dict(status='complete',counts=report['status_counts'],outputs_sha256={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(json.dumps({k:report[k] for k in ('status_counts','reproduced_recordings','recordings','elapsed_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

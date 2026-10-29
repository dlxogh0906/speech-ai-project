"""P4: frozen TRAIN readouts on P3 B0/B3; no optimizer or label changes.

Parent 2k window bounds are fixed across encoders. The B3 trained auxiliary
readout is also applied to B0 as an explicitly cross-model diagnostic: B0 has
no trained auxiliary head. TAG windows are diagnostic inputs only.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import hashlib
import itertools
import json
import time
import numpy as np
import torch
from train_h0_intermediate import P3Model
from train_h0_backbone_control import file_sha
from train_w2vbert_ctc import read_wav
from mdd.gop_feats_v2 import viterbi_align
from mdd.h0_phone_audit import project_gaps


MODES = ('parent_auto', 'own_auto', 'tag_same_readout', 'tag_matched_readout')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def prepare(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    p3 = Path(a.p3); p1 = Path(a.p1)
    prior = json.loads((p3/'design_v2/protocol.json').read_text())
    paths = [p3/'design_v2/protocol.json', p1/'protocol.json', p1/'probes/fit_protocol.json',
             p1/'utterances.jsonl', p1/'sites.jsonl', p1/'w2vbert_2000/window_bounds.npy', Path(__file__)]
    paths += [p3/arm/'step_4000.pt' for arm in ('B0','B3')]
    for arm in ('B0','B3'):
        paths += [p3/(arm+'_eval_4000')/f for f in ('report.json','sites.jsonl','sequences.jsonl')]
    for window in ('auto','tag'):
        paths += [p1/'probes'/('w2vbert_2000_L12_'+window)/f for f in ('probe.pt','calibration.json')]
    # Bind the imported inference implementation without editing the P3 recipe.
    paths += [ROOT/'tasks/pronunciation-assessment/scripts/train'/f for f in
              ('train_h0_intermediate.py','train_h0_backbone_control.py','train_w2vbert_ctc.py')]
    paths += [ROOT/'tasks/pronunciation-assessment/src/mdd'/f for f in
              ('h0_intermediate.py','h0_phone_audit.py','gop_feats_v2.py')]
    p = dict(version='h0_p4_v1', p3=str(p3), p1=str(p1), source=prior['source'],
        parent_checkpoint=prior['parent_checkpoint'], manifest=prior['manifest'],
        arms=['B0','B3'], step=4000, encoder_updates=0, readout_updates=0,
        primary='fixed parent-2k L12 auto probe and normalization, parent-2k auto bounds',
        auxiliary='one trained B3 LayerNorm+aux head on both encoders; B0 cross-readout is not a native trained B0 head',
        windows=dict(parent_auto='parent 2k canonical Viterbi bounds; identical across arms',
            own_auto='each arm final canonical Viterbi; includes window movement',
            tag_same_readout='same auto-trained probe on fixed TAG bounds; window distribution shift',
            tag_matched_readout='separate existing parent 2k TAG-trained probe; never refit'),
        calibration='report original parent-probe threshold transfer and seen-CAL-only recalibration separately; never retune on selection',
        evaluation='all 14009 prior evaluation recordings; 46 phones and deletion; auxiliary/final sequence gaps separately',
        limitations=['repeated nine unseen development prompts', 'TAG time is not a physical boundary ground truth',
            'fixed-readout failure alone cannot prove absence of information',
            'probe has no insertion policy; aux/final gaps are greedy only',
            'no listening, independent labels, new test, or production changes'],
        bindings={str(path):file_sha(path) for path in paths})
    write_json(out/'protocol.json',p)


def score_ctc(logits, length, canonical, vocab):
    lp = logits[0,:length].float().log_softmax(-1).cpu().numpy()
    if not np.isfinite(lp).all(): raise ValueError('nonfinite CTC')
    inv = {v:k for k,v in vocab.items()}
    pred = [inv[t] for t,_ in itertools.groupby(lp.argmax(-1)) if t in inv]
    projected, _ = project_gaps(canonical,pred)
    seg = viterbi_align(lp,[vocab[t] for t in canonical],blank=0)
    scores=[]
    for q,t in enumerate(canonical):
        if seg is None or not seg[q]: scores.append(None); continue
        l=lp[seg[q]]; other=l.copy(); other[:,vocab[t]]=-np.inf
        scores.append(float((other.max(-1)-l[:,vocab[t]]).mean()))
    return pred,projected,scores,seg


def pool(hidden, bounds):
    # Exactly reproduce P1: FP32 frame mean -> FP16 cache -> FP32 readout.
    result=np.zeros((len(bounds),hidden.shape[-1]),np.float32)
    available=np.zeros(len(bounds),bool)
    for q,(left,right) in enumerate(bounds):
        if left < 0: continue
        if not (left < right <= len(hidden)): raise ValueError('bound outside encoder frames')
        result[q]=hidden[left:right].mean(0).astype(np.float16).astype(np.float32)
        available[q]=True
    return result,available


def apply_probe(x, state):
    x=(x-state['mean'])/state['std']
    return torch.nn.functional.linear(torch.from_numpy(x),state['weight'],state['bias']).numpy()


def extract(a):
    root=Path(a.out); p=json.loads((root/'protocol.json').read_text())
    for path,digest in p['bindings'].items():
        if file_sha(path)!=digest: raise ValueError('changed bound input: '+path)
    dest=root/(('smoke_' if a.smoke else '')+a.arm); dest.mkdir(exist_ok=False)
    qroot=Path(p['p1']); p3=Path(p['p3'])
    records=[r for r in read_jsonl(qroot/'utterances.jsonl') if r['split']!='train']
    all_sites=read_jsonl(qroot/'sites.jsonl')
    if a.smoke:
        predicates=[lambda r:r['role']=='calibration',lambda r:r['role']=='selection' and r['split']=='valid',
                    lambda r:r['item']=='RW054' and r['whole_tag_normal'],
                    lambda r:r['split']=='valid_unseen' and any(all_sites[r['offset']+i]['actual']=='ㅗ' for i in range(len(r['canonical'])))]
        records=list({r['uid']:r for r in [next(r for r in records if fn(r)) for fn in predicates]}.values())
    ids=np.array([r['offset']+i for r in records for i in range(len(r['canonical']))],np.int64)
    np.save(dest/'site_ids.npy',ids)
    bounds=np.load(qroot/'w2vbert_2000/window_bounds.npy',mmap_mode='r')
    auto_probe=torch.load(str(qroot/'probes/w2vbert_2000_L12_auto/probe.pt'),map_location='cpu')
    tag_probe=torch.load(str(qroot/'probes/w2vbert_2000_L12_tag/probe.pt'),map_location='cpu')
    # Obtain B3's learned readout even for B0; do not use B0's inactive aux copy.
    b3=torch.load(str(p3/'B3/step_4000.pt'),map_location='cpu')
    norm=torch.nn.LayerNorm(1024); aux=torch.nn.Linear(1024,50)
    for module,name in ((norm,'norm'),(aux,'aux')):
        module.load_state_dict({k:b3['model']['added.'+name+'.'+k] for k in ('weight','bias')})
    del b3
    parent=a.arm=='parent'
    ck=torch.load(p['parent_checkpoint'] if parent else str(p3/a.arm/'step_4000.pt'),map_location='cpu')
    vocab=ck['vocab']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=P3Model(p['source'],vocab,device,'B0' if parent else a.arm,
                  parent_model=ck['model'] if parent else None)
    if not parent: model.load_state_dict(ck['model'],strict=True)
    del ck
    model.eval(); model.requires_grad_(False); norm.to(device).eval(); aux.to(device).eval()
    norms={}; handle=model.added.register_forward_pre_hook(lambda module, inputs:norms.update(middle=inputs[0]))
    old_sites={} if parent else {(s['uid'],s['q']):s for s in read_jsonl(p3/(a.arm+'_eval_4000')/'sites.jsonl')}
    old_seq={} if parent else {r['uid']:r for r in read_jsonl(p3/(a.arm+'_eval_4000')/'sequences.jsonl')}
    arrays={mode:np.lib.format.open_memmap(str(dest/(mode+'.npy')),mode='w+',dtype=np.float32,
                                           shape=(len(ids),47)) for mode in MODES}
    available=np.zeros((len(ids),len(MODES)),bool)
    start=time.time(); offset=0; maxdiff=0.; pcm=hashlib.sha256(); matched=0; native_aux_diff=0.
    parent_features=np.load(qroot/'w2vbert_2000/L12_auto.npy',mmap_mode='r') if parent else None
    with (dest/'aux_sites.jsonl').open('x') as f, (dest/'aux_sequences.jsonl').open('x') as sf, torch.inference_mode():
        for i,r in enumerate(records):
            x=read_wav(r['path'],20.)
            if len(x)!=r['frames']: raise ValueError('truncated audio')
            pcm.update(r['uid'].encode());pcm.update(x.tobytes())
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                final,lengths,native_aux=model([x],auxiliary=True)
                auxiliary=aux(norm(norms['middle']))
            if native_aux is not None:
                native_aux_diff=max(native_aux_diff,float((native_aux-auxiliary).abs().max()))
                if native_aux_diff!=0: raise ValueError('B3 readout copy differs')
            length=int(lengths[0]); h=norms['middle'][0,:length].float().cpu().numpy()
            fp,proj,scores,segments=score_ctc(final,length,r['canonical'],vocab)
            ap,aproj,asc,_=score_ctc(auxiliary,length,r['canonical'],vocab)
            n=len(r['canonical']); rid=np.arange(r['offset'],r['offset']+n)
            if not parent:
                if fp!=old_seq[r['uid']]['predicted']: raise ValueError('final sequence reproduction failed')
                for j in range(n):
                    s=old_sites[r['uid'],j]
                    if proj[j]!=s['predicted']: raise ValueError('final projection changed')
                    if (scores[j] is None)!=(s['score'] is None): raise ValueError('score availability changed')
                    if scores[j] is not None: maxdiff=max(maxdiff,abs(scores[j]-s['score']))
                if maxdiff>1e-6: raise ValueError(('final GOP mismatch',maxdiff))
                matched+=n
            own=np.array([[min(s),max(s)+1] if s else [-1,-1] for s in segments]) if segments is not None else np.full((n,2),-1)
            windows=[bounds[rid,:2],own,bounds[rid,4:6],bounds[rid,4:6]]
            for k,(mode,b) in enumerate(zip(MODES,windows)):
                xx,ok=pool(h,b)
                if parent and mode=='parent_auto' and not np.array_equal(xx,parent_features[rid].astype(np.float32)):
                    raise ValueError('parent pooled features differ from P1')
                arrays[mode][offset:offset+n]=apply_probe(xx,tag_probe if mode=='tag_matched_readout' else auto_probe)
                available[offset:offset+n,k]=ok
            for j in range(n):
                s=dict(old_sites[r['uid'],j]) if not parent else dict(all_sites[r['offset']+j])
                s.update(predicted=aproj[j],score=asc[j])
                f.write(json.dumps(s,ensure_ascii=False)+'\n')
            sr={k:r[k] for k in ('uid','speaker','item','split','role','canonical','actual')}
            sr.update(predicted=ap);sf.write(json.dumps(sr,ensure_ascii=False)+'\n')
            offset+=n
            if (i+1)%100==0 or i+1==len(records):
                status=dict(arm=a.arm,smoke=a.smoke,utterances=i+1,target=len(records),elapsed_seconds=time.time()-start,
                            estimated_remaining_seconds=(time.time()-start)/(i+1)*(len(records)-i-1))
                write_json(dest/'live_status.json',status); write_json(root/'live_status.json',status)
                print(json.dumps(status),flush=True)
    handle.remove()
    for array in arrays.values():array.flush()
    np.save(dest/'available.npy',available)
    write_json(dest/'finished.json',dict(status='extracted',smoke=a.smoke,arm=a.arm,utterances=len(records),sites=len(ids),
        final_reproduced_sites=matched,final_GOP_max_difference=maxdiff,native_aux_max_difference=native_aux_diff,
        parent_feature_reproduction=parent,encoder_updates=0,probe_updates=0,available=available.sum(0).tolist(),
        protocol_sha256=file_sha(root/'protocol.json'),audio_order_and_PCM_sha256=pcm.hexdigest(),elapsed_seconds=time.time()-start))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['prepare','extract'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--p1');ap.add_argument('--p3')
    ap.add_argument('--arm',choices=['B0','B3','parent']);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.stage=='prepare': prepare(a)
    else:
        if a.arm=='parent' and not a.smoke: raise ValueError('parent is reproduction smoke only')
        extract(a)


if __name__=='__main__':main()

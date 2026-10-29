"""Fit a predeclared fixed-size TRAIN-only probe, freeze seen CAL thresholds.

No encoder updates, checkpoint/layer selection, or deployment. Scores on all
canonical selection sites retain unavailable windows in their denominators.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import csv
import hashlib
import json
import math
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_h0_backbone_control import file_sha
from diagnose_h0_representations import LAYERS, WINDOWS, write_json


def table(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter='\t'); w.writeheader()
        for r in rows: w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def threshold(scores, normal, mismatch, available, fpr=.01):
    budget = math.floor(int(normal.sum()) * fpr)
    values = np.sort(scores[normal & mismatch & available].astype(np.float64))[::-1]
    return float(np.nextafter(values[budget], np.inf)) if len(values) > budget else -np.inf


def counts(idx, expected, actual, pred, scores, avail, tau):
    e, t, y = expected[idx], actual[idx], pred[idx]
    available = avail[idx]; mismatch = y != e; normal = t == e; errors = ~normal
    alarm = mismatch & available & (scores[idx].astype(np.float64) >= tau)
    exact = errors & (y == t) & alarm
    n, err = int(normal.sum()), int(errors.sum())
    fp = int((normal & alarm).sum()); hit = int(exact.sum())
    return dict(total=len(idx), normal=n, errors=err, available=int(available.sum()),
                available_normal=int((available & normal).sum()), available_errors=int((available & errors).sum()),
                normal_fp=fp, exact=hit, wrong_actual=int((errors & alarm & (y!=t)).sum()),
                reverted=int((errors & available & (y==e)).sum()),
                normal_fpr=fp/n if n else None, exact_recall=hit/err if err else None,
                actual_phone_accuracy=float((y[available]==t[available]).mean()) if available.any() else None,
                coverage=float(available.mean()) if len(idx) else None,
                fpr_within_1pct=(fp/n <= .01) if n else None)


def intervals(idx, clusters, expected, actual, pred, scores, avail, tau):
    normal = actual == expected
    alarm = (pred != expected) & avail & (scores.astype(np.float64) >= tau)
    columns = np.stack([normal & alarm,normal,(~normal)&alarm&(pred==actual),~normal],1).astype(int)
    keys, inv = np.unique(clusters[idx], return_inverse=True)
    if len(keys) < 2: return dict(clusters=len(keys), ci95=None)
    sums = np.zeros((len(keys),4),dtype=int); np.add.at(sums,inv,columns[idx])
    rng = np.random.RandomState(20261027)
    z = sums[rng.randint(len(keys),size=(2000,len(keys)))].sum(1)
    result = dict(clusters=len(keys), repeats=2000)
    for name, num, den in [('fpr',0,1),('exact_recall',2,3)]:
        values = z[z[:,den]>0,num]/z[z[:,den]>0,den]
        result[name] = np.quantile(values,[.025,.975]).tolist() if len(values) else None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',required=True)
    a = ap.parse_args(); root = Path(a.root); p = json.loads((root/'protocol.json').read_text())
    out = root/'probes'; out.mkdir(exist_ok=False)
    sites = [json.loads(s) for s in (root/'sites.jsonl').open()]
    n = len(sites); classes = p['probe']['classes']; ids = {c:i for i,c in enumerate(classes)}
    expected=np.array([ids[s['expected']] for s in sites]); actual=np.array([ids[s['actual'] or '<DEL>'] for s in sites])
    split=np.array([s['split'] for s in sites]); role=np.array([s['role'] for s in sites])
    item=np.array([s['item'] for s in sites]); speaker=np.array([s['speaker'] for s in sites])
    train=role=='train'; cal=(split=='valid')&(role=='calibration'); normal=actual==expected
    conditions=['%s_%d'%(kind,step) for kind in ('w2vbert','xlsr300m') for step in (2000,4000)]
    masks={}; audio_digests=set()
    for name in conditions:
        receipt=json.loads((root/name/'finished.json').read_text())
        if receipt['smoke'] or receipt['sites']!=n or receipt['protocol_sha256']!=file_sha(root/'protocol.json'):
            raise ValueError('feature provenance mismatch')
        masks[name]=np.load(root/name/'available.npy')
        audio_digests.add(receipt['audio_order_and_pcm_sha256'])
        train &= masks[name].all(1)
    if len(audio_digests)!=1:raise ValueError('audio/order differs between backbones')
    train_idx=np.flatnonzero(train)
    if not len(train_idx):raise ValueError('no shared TRAIN evidence')
    np.save(out/'train_ids.npy',train_idx)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=p['probe']; rng=np.random.RandomState(cfg['seed'])
    draws=rng.randint(len(train_idx),size=(cfg['updates'],cfg['batch']))
    np.save(out/'batch_draws.npy',draws)
    np.save(out/'labels.npy',actual)
    audit=dict(shared_training_sites=len(train_idx),training_utterances=len(set(sites[i]['uid'] for i in train_idx)),
               class_support={c:int((actual[train_idx]==k).sum()) for c,k in ids.items()},
               draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),updates_per_probe=cfg['updates'],
               input_features=1024,parameters=(1024+1)*len(classes),encoder_updates=0,
               selection_used_for_fitting=False,normalization_fit_on='shared TRAIN only',
               classes=classes,protocol_sha256=file_sha(root/'protocol.json'))
    exposed=train_idx[np.unique(draws)]
    audit.update(actual_draws=int(draws.size),actual_unique_sites=len(exposed),
                 actual_unique_utterances=len(set(sites[i]['uid'] for i in exposed)),
                 audio_order_and_pcm_sha256=next(iter(audio_digests)),
                 optimizer_actual=dict(name='AdamW',betas=[.9,.999],eps=1e-8,
                    lr=cfg['lr'],weight_decay=cfg['weight_decay'],gradient_clip=1.,schedule='constant'),
                 inference_precision='FP32 linear probe on FP16 cached means standardized using TRAIN only')
    write_json(out/'fit_protocol.json',audit)
    common=np.logical_and.reduce([m.all(1) for m in masks.values()])
    np.save(out/'common_window_ids.npy',np.flatnonzero(common))
    groups={}
    for sp in ('valid','valid_unseen'):
        base=(split==sp)&(role=='selection')
        groups[sp,'all']=np.flatnonzero(base)
        groups[sp,'common_windows']=np.flatnonzero(base&common)
        for pos in ('onset','nucleus','coda'):
            groups[sp,pos]=np.flatnonzero(base & np.array([s['position'].startswith(pos) for s in sites]))
    summary=[]; phones=[]; items=[]; pairs=[]; suri=[]; uncertainty={}; transfer=[]; drift=[]
    device='cuda'; targets=torch.from_numpy(actual[train_idx]).to(device)
    batch_draws=torch.from_numpy(draws).to(device)
    started=time.time()
    for name in conditions:
        for layer in LAYERS:
            for wi,window in enumerate(WINDOWS):
                key='%s_L%d_%s'%(name,layer,window)
                dest=out/key;dest.mkdir()
                mm=np.load(root/name/('L%d_%s.npy'%(layer,window)),mmap_mode='r')
                # All feature transforms use the same eligible TRAIN rows.
                raw=mm[train_idx].astype(np.float32)
                mean=raw.mean(0);std=np.maximum(raw.std(0),.01)
                features=torch.from_numpy((raw-mean)/std).to(device); del raw
                torch.manual_seed(cfg['seed'])
                probe=torch.nn.Linear(1024,len(classes)).to(device)
                opt=torch.optim.AdamW(probe.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
                probe.train()
                for step in range(cfg['updates']):
                    draw=batch_draws[step];opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(probe(features[draw]),targets[draw])
                    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);opt.step()
                    if not torch.isfinite(loss):raise ValueError('nonfinite probe loss')
                probe.eval()
                state=dict(weight=probe.weight.detach().cpu(),bias=probe.bias.detach().cpu(),mean=mean,std=std,
                           fit_protocol_sha256=file_sha(out/'fit_protocol.json'))
                torch.save(state,str(dest/'probe.pt'))
                logits=np.empty((n,len(classes)),dtype=np.float32)
                with torch.inference_mode():
                    for lo in range(0,n,4096):
                        batch=torch.from_numpy((mm[lo:lo+4096].astype(np.float32)-mean)/std).to(device)
                        logits[lo:lo+len(batch)]=probe(batch).cpu().numpy()
                np.save(dest/'logits.npy',logits)
                pred=logits.argmax(1);scores=(logits.max(1)-logits[np.arange(n),expected]).astype(np.float64)
                avail=masks[name][:,wi]
                tau=threshold(scores,normal&cal,pred!=expected,avail)
                write_json(dest/'calibration.json',dict(threshold=None if not np.isfinite(tau) else tau,
                    negative_infinity=bool(tau==-np.inf),**counts(np.flatnonzero(cal),expected,actual,pred,scores,avail,tau)))
                for (sp,group),idx in groups.items():
                    for policy,t in [('raw',-np.inf),('calibrated',tau)]:
                        summary.append(dict(condition=name,layer=layer,window=window,split=sp,group=group,policy=policy,
                                            **counts(idx,expected,actual,pred,scores,avail,t)))
                uncertainty[key]={}
                for sp in ('valid','valid_unseen'):
                    idx=groups[sp,'all']
                    uncertainty[key][sp]={k:intervals(idx,g,expected,actual,pred,scores,avail,tau)
                                          for k,g in [('speaker',speaker),('item',item)]}
                    for phone in classes:
                        ix=idx[expected[idx]==ids[phone]]
                        phones.append(dict(condition=name,layer=layer,window=window,split=sp,phone=phone,
                            items=len(set(item[ix])),train_actual_support=audit['class_support'][phone],
                            **counts(ix,expected,actual,pred,scores,avail,tau),
                            **{'raw_'+k:v for k,v in counts(ix,expected,actual,pred,scores,avail,-np.inf).items()}))
                    for it in sorted(set(item[idx])):
                        ix=idx[item[idx]==it]
                        items.append(dict(condition=name,layer=layer,window=window,split=sp,item=it,
                            **counts(ix,expected,actual,pred,scores,avail,tau)))
                    # Pair AUC conditions on the expected phone and restricts
                    # actual labels to the two phones, preventing wrong-phone
                    # alarms from being treated as exact diagnosis.
                    for c in classes:
                        cx=idx[(expected[idx]==ids[c])&avail[idx]]
                        for t in sorted(set(actual[cx])):
                            if t==ids[c]:continue
                            ix=cx[(actual[cx]==t)|(actual[cx]==ids[c])]
                            y=actual[ix]==t;positive=int(y.sum());negative=int((~y).sum())
                            auc=float(roc_auc_score(y,logits[ix,t]-logits[ix,ids[c]])) if positive and negative else None
                            pairs.append(dict(condition=name,layer=layer,window=window,split=sp,expected=c,actual=classes[t],
                                              errors=positive,normal=negative,items=len(set(item[ix])),auc=auc))
                sx=np.flatnonzero((split=='valid_unseen')&(role=='selection')&(item=='RW054')&
                                  np.array([s['whole_tag_normal'] for s in sites])&(expected==ids['ㅜ']))
                if len(sx)!=475:raise ValueError('suri fixed cohort changed')
                margin=logits[sx,ids['ㅜ']]-logits[sx,ids['ㅗ']]
                sv=avail[sx]
                suri.append(dict(condition=name,layer=layer,window=window,total=475,available=int(sv.sum()),
                    predicts_u=int(((pred[sx]==ids['ㅜ'])&sv).sum()),predicts_o=int(((pred[sx]==ids['ㅗ'])&sv).sum()),
                    u_over_o=int(((margin>0)&sv).sum()),median_u_minus_o=float(np.median(margin[sv])) if sv.any() else None,
                    exact_actual_auc=None,auc_reason='single actual class in fixed 475 normal utterances',
                    **{'calibrated_'+k:v for k,v in counts(sx,expected,actual,pred,scores,avail,tau).items()}))
                if name.endswith('_4000'):
                    previous=name.replace('_4000','_2000');oldkey='%s_L%d_%s'%(previous,layer,window)
                    oldstate=torch.load(str(out/oldkey/'probe.pt'),map_location='cpu')
                    frozen=torch.nn.Linear(1024,len(classes)).to(device)
                    frozen.load_state_dict({k:oldstate[k].to(device) for k in ('weight','bias')});frozen.eval()
                    cross=np.empty_like(logits)
                    with torch.inference_mode():
                        for lo in range(0,n,4096):
                            xx=torch.from_numpy((mm[lo:lo+4096].astype(np.float32)-oldstate['mean'])/oldstate['std']).to(device)
                            cross[lo:lo+len(xx)]=frozen(xx).cpu().numpy()
                    np.save(dest/'frozen_2k_probe_logits.npy',cross)
                    oldcal=json.loads((out/oldkey/'calibration.json').read_text());oldt=oldcal['threshold']
                    if oldcal['negative_infinity']:oldt=-np.inf
                    cp=cross.argmax(1);cs=(cross.max(1)-cross[np.arange(n),expected]).astype(np.float64)
                    for sp in ('valid','valid_unseen'):
                        transfer.append(dict(condition=name,layer=layer,window=window,split=sp,
                            **counts(groups[sp,'all'],expected,actual,cp,cs,avail,oldt)))
                    oldfeatures=np.load(root/previous/('L%d_%s.npy'%(layer,window)),mmap_mode='r')
                    oldlogits=np.load(out/oldkey/'logits.npy',mmap_mode='r')
                    joint=sv&masks[previous][sx,wi];ix=sx[joint]
                    z0=oldfeatures[ix].astype(float);z1=mm[ix].astype(float)
                    cosine=(z0*z1).sum(1)/(np.linalg.norm(z0,axis=1)*np.linalg.norm(z1,axis=1)).clip(1e-12)
                    change=logits[ix,ids['ㅜ']]-logits[ix,ids['ㅗ']]-(oldlogits[ix,ids['ㅜ']]-oldlogits[ix,ids['ㅗ']])
                    fixedchange=cross[ix,ids['ㅜ']]-cross[ix,ids['ㅗ']]-(oldlogits[ix,ids['ㅜ']]-oldlogits[ix,ids['ㅗ']])
                    drift.append(dict(backbone=name.rsplit('_',1)[0],layer=layer,window=window,paired_recordings=len(ix),
                        same_physical_window=(window=='tag'),median_cosine=float(np.median(cosine)),
                        median_relative_l2=float(np.median(np.linalg.norm(z1-z0,axis=1)/np.linalg.norm(z0,axis=1).clip(1e-12))),
                        median_own_probe_margin_change=float(np.median(change)),median_frozen_2k_margin_change=float(np.median(fixedchange)),
                        frozen_2k_probe_u=int((cp[ix]==ids['ㅜ']).sum()),frozen_2k_probe_o=int((cp[ix]==ids['ㅗ']).sum())))
                write_json(dest/'finished.json',dict(status='fitted_and_evaluated',updates=cfg['updates'],
                    final_train_loss_execution_check_only=float(loss),encoder_updates=0,source_sha256=file_sha(__file__)))
                print(json.dumps(dict(completed=key,elapsed_seconds=time.time()-started)),flush=True)
                write_json(root/'live_status.json',dict(status='probe_fitting',completed=key,elapsed_seconds=time.time()-started))
                del features,logits,probe,opt,mm
    for filename,rows in [('summary.tsv',summary),('phones.tsv',phones),('items.tsv',items),('pairs.tsv',pairs),
                          ('suri.tsv',suri),('cross_checkpoint.tsv',transfer),('suri_representation_drift.tsv',drift)]:
        table(out/filename,rows)
    write_json(out/'uncertainty.json',uncertainty)
    write_json(out/'report.json',dict(fit=audit,summary=summary,suri=suri,drift=drift,
                                    elapsed_seconds=time.time()-started))
    write_json(out/'finished.json',dict(status='completed',probes=36,encoder_updates=0,
                probe_updates=36*cfg['updates'],protocol_sha256=file_sha(root/'protocol.json')))
    write_json(root/'live_status.json',dict(status='P1_probe_matrix_completed',probes=36,encoder_updates=0))


if __name__=='__main__':main()

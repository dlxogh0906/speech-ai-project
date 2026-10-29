"""Matched actual-vowel auxiliary learning; no expected-vowel input or gate.

Keep all actual-class examples. Only the auxiliary TRAIN loss ignores unknown,
nonadjacent or unobserved intended-vowel targets. CAL identity is evaluation
metadata without TAG boundaries; all service outcomes keep their denominators.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from train_sibilant_local_encoder import load_encoder,hidden,write
from train_sibilant_cv_support import phase_loss,calibrate
from train_sibilant_joint_vowel import aucs
from mdd.sibilant_cv_support import reliable_support
from mdd.sibilant_vowel_aux import TOKENS,IDS,VowelAuxReadout,intended_vowel_target,vowel_loss


def prepare(p,root):
    out=root/'labels';out.mkdir(parents=True,exist_ok=False)
    report={'scope':'TAG targets only, never runtime inputs. TRAIN requires observed intended V phase; CAL identity only, not boundary/support assertion.','cohorts':{}}
    for split in ('train','cal'):
        prefix=p[split+'_prefix'];fs=service.read_jsonl(prefix+'.features.jsonl');ls=service.read_jsonl(prefix+'.labels.jsonl')
        rows={r['row']['uid']:r['row'] for r in service.read_jsonl(prefix+'.responses.jsonl')}
        byindex=collections.defaultdict(list)
        for z in ls:
            if z.get('feature_index') is not None:byindex[z['feature_index']].append(z)
        phase=np.load(prefix+'.npz')['phase'] if split=='train' else None
        targets=np.full(len(fs),-100,np.int64);counts=collections.Counter();mapping_cache={};details=[]
        for i,f in enumerate(fs):
            state='label_mapping_not_unique'
            if len(byindex[i])==1:
                z=byindex[i][0];row=rows[z['uid']]
                if z['uid'] not in mapping_cache:mapping_cache[z['uid']]=service.all_optimal_mapping(row['canonical'],row['actual'])
                target,state=intended_vowel_target(z,row,mapping_cache[z['uid']],phase[i] if phase is not None else None)
                targets[i]=target
                if target>=0:
                    counts['actual_V/'+TOKENS[target]]+=1
                    counts['TAG_actual_V_differs_from_canonical']+=int(row['canonical'][z['q']+1]!=TOKENS[target])
            counts[state]+=1;details.append({'feature_index':i,'uid':f['uid'],'q_runtime':f['q_runtime'],'target':int(targets[i]),'state':state})
        np.savez(out/(split+'.npz'),targets=targets)
        with (out/(split+'.jsonl')).open('x') as stream:
            for row in details:service.write_line(stream,row)
        report['cohorts'][split]={'all_model_inputs':len(fs),'known_vowel_targets':int((targets>=0).sum()),'counts':dict(counts),'targets_sha256':service.sha(out/(split+'.npz'))}
        print('labels',split,report['cohorts'][split],flush=True)
    write(out/'report.json',report);write(out/'finished.json',{'sha256':service.sha(out/'report.json')})


def score_all(encoder,head,data):
    if set(data)!={'x','lengths','relative'}:raise ValueError('inference includes non-acoustic labels')
    ps,vs,av=[],[],[]
    with torch.no_grad():
        for i in range(0,len(data['x']),64):
            x=torch.as_tensor(data['x'][i:i+64],device='cuda');length=torch.as_tensor(data['lengths'][i:i+64],device='cuda')
            mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None]
            logits,phase,vowel=head.all_outputs(hidden(encoder,x,mask),mask,torch.as_tensor(data['relative'][i:i+64],device='cuda'))
            ps.extend(logits.softmax(-1).cpu().numpy());vs.extend(vowel.softmax(-1).cpu().numpy());av.extend(reliable_support(phase,mask).cpu().tolist())
    return np.asarray(ps),np.asarray(vs),np.asarray(av)


def fit(p,root):
    out=root/'fit';out.mkdir(parents=True,exist_ok=False)
    train=dict(np.load(p['train_prefix']+'.npz'));cal=dict(np.load(p['cal_prefix']+'.npz'))
    trainV=np.load(root/'labels/train.npz')['targets'];calV=np.load(root/'labels/cal.npz')['targets']
    plan=np.load(p['batch_plan'])[:p['steps']];np.save(out/'batch_plan.npy',plan)
    base=torch.load(p['base_artifact'],map_location='cpu')
    labels=service.read_jsonl(p['cal_prefix']+'.labels.jsonl');features=service.read_jsonl(p['cal_prefix']+'.features.jsonl');records=service.read_jsonl(p['cal_prefix']+'.responses.jsonl')
    view={r['row']['uid']:observe(r['prior'],r['row']) for r in records};labels=[dict(r,existing=view[r['uid']][r['q']]) for r in labels]
    report={'scope':p['scope'],'steps':p['steps'],'inference_batch':64,'test_scored':False,'all_CAL_sites':len(labels),'all_CAL_utterances':len(records),'arms':{}}
    initial=None
    for arm,weight in [('zero_aux',0.),('vowel_aux',p['vowel_weight'])]:
        encoder=load_encoder(p,True);encoder.encoder.layers[-2:].load_state_dict(base['last2'])
        torch.manual_seed(p['seed']);head=VowelAuxReadout(encoder.config.hidden_size).to('cuda');head.load_base(base['head'])
        digest=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
        if initial is None:initial=digest
        if initial!=digest:raise AssertionError('auxiliary controls do not have identical initialization')
        optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':.001},{'params':[v for v in encoder.parameters() if v.requires_grad],'lr':.00001}],weight_decay=.01)
        losses,phases,vowels,tick=[],[],[],time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step,ix in enumerate(plan,1):
            optimizer.zero_grad(set_to_none=True)
            x=torch.as_tensor(train['x'][ix],device='cuda');length=torch.as_tensor(train['lengths'][ix],device='cuda')
            mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None]
            logits,phase,v=head.all_outputs(hidden(encoder,x,mask),mask,torch.as_tensor(train['relative'][ix],device='cuda'))
            cls=torch.nn.functional.cross_entropy(logits,torch.as_tensor(train['y'][ix],device='cuda'))
            ph=phase_loss(phase,torch.as_tensor(train['phase'][ix],device='cuda'));vl=vowel_loss(v,torch.as_tensor(trainV[ix],device='cuda'))
            loss=cls+p['phase_weight']*ph+weight*vl
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite auxiliary loss')
            loss.backward()
            if step==1 and any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):raise AssertionError('early encoder gradient')
            torch.nn.utils.clip_grad_norm_([v for g in optimizer.param_groups for v in g['params']],1.)
            optimizer.step();losses.append(float(cls.detach()));phases.append(float(ph.detach()));vowels.append(float(vl.detach()))
            if step%100==0:print(arm,step,'class',float(np.mean(losses[-50:])),'phase',float(np.mean(phases[-50:])),'V',float(np.mean(vowels[-50:])),flush=True)
        train_seconds=time.monotonic()-tick
        ps,vp,av=score_all(encoder,head,cal)
        np.savez(out/(arm+'.cal_scores.npz'),probabilities=ps,vowel_probabilities=vp,available=av)
        thresholds,audit=calibrate(labels,features,ps,av)
        for key in thresholds:
            if key.endswith('/affricate'):thresholds[key]=None
        known=calV>=0;pred=vp.argmax(-1)
        vowelmetrics={'known_identity_inputs':int(known.sum()),'unknown_identity_inputs':int((~known).sum()),
            'top1_correct':int((pred[known]==calV[known]).sum()),'top1_accuracy':float((pred[known]==calV[known]).mean()),
            'by_actual_vowel':{token:{'all':int((calV==i).sum()),'correct':int(((calV==i)&(pred==i)).sum())} for i,token in enumerate(TOKENS)}}
        artifact={'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'last2':{k:v.detach().cpu() for k,v in encoder.encoder.layers[-2:].state_dict().items()},
            'readout':'vowel_aux','vowel_tokens':list(TOKENS),'vowel_weight':weight,'learned':True,'base_artifact_sha256':service.sha(p['base_artifact'])}
        torch.save(artifact,out/(arm+'.pt'))
        report['arms'][arm]={'initial_head_sha256':digest,'thresholds':thresholds,'calibration':audit,
            'additional_exact':sum(v['cal_additional_exact'] for k,v in audit.items() if k.endswith('/pair')),
            'unconditional_AUC':aucs(labels,ps,np.ones(len(ps),dtype=bool)),'available':int(av.sum()),'actual_vowel_identity':vowelmetrics,
            'train_seconds':train_seconds,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),
            'loss_first_last50':{'class':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],'V':[float(np.mean(vowels[:50])),float(np.mean(vowels[-50:]))]}}
        write(out/'report.json',report);del head,encoder,optimizer;torch.cuda.empty_cache()
    n={k:v['additional_exact'] for k,v in report['arms'].items()}
    report['selected_for_single_request']='vowel_aux' if n['vowel_aux']>max(0,n['zero_aux']) else 'zero_aux' if n['zero_aux']>0 else None
    write(out/'report.json',report);write(out/'finished.json',{'sha256':service.sha(out/'report.json')});print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True,choices=('prepare','fit'))
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());root=Path(a.root)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    torch.set_num_threads(2);{'prepare':prepare,'fit':fit}[a.stage](p,root)


if __name__=='__main__':main()

"""Same V-reference inputs/updates: grouped other versus explicit TAG deletion.

Both arms inherit identical coarse probabilities. Same sampler,300updates,
joint C/V phase loss and last2 updates. No C-peak gate on absence hypotheses.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_local_encoder import load_encoder,hidden,write
from mdd.sibilant_joint_vowel import joint_vowel_loss
from mdd.sibilant_vowel_presence import PresenceReadout,SCHEMA,CLASSES,observed_vowel_support


def model(p,artifact,tuned=False):
    if artifact.get('readout')!='vowel_reference_presence' or artifact.get('coordinate_schema')!=SCHEMA:
        raise ValueError('matching V-reference presence artifact required')
    encoder=load_encoder(p,tuned);encoder.encoder.layers[-2:].load_state_dict(artifact['last2'])
    head=PresenceReadout(encoder.config.hidden_size,artifact['split_absence']).to('cuda');head.load_state_dict(artifact['head'])
    return encoder,head


def scores(encoder,head,data,batch=64):
    if set(data)!={'x','relative_vowel','lengths'}:raise ValueError('V-reference acoustic inputs only')
    ps,av=[],[];tick=time.monotonic()
    with torch.no_grad():
        for i in range(0,len(data['x']),batch):
            x=torch.as_tensor(data['x'][i:i+batch],device='cuda');length=torch.as_tensor(data['lengths'][i:i+batch],device='cuda')
            mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None]
            logits,phase=head(hidden(encoder,x,mask),mask,torch.as_tensor(data['relative_vowel'][i:i+batch],device='cuda'))
            ps.extend(logits.softmax(-1).cpu().numpy());av.extend(observed_vowel_support(phase,mask).cpu().tolist())
            if batch==1 and (i+1)%500==0:print('single requests',i+1,'seconds',round(time.monotonic()-tick,1),flush=True)
    return np.asarray(ps),np.asarray(av),time.monotonic()-tick


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text());torch.set_num_threads(2)
    train=dict(np.load(Path(p['train_cache'])/'data.npz'));target=dict(np.load(Path(p['train_cache'])/'targets.npz'))
    if set(train)!={'x','relative_vowel','lengths'}:raise ValueError('invalid V-coordinate cache')
    groups=[np.flatnonzero(target['y']==0),np.flatnonzero(target['y']==1),np.flatnonzero(np.isin(target['y'],[2,3])),np.flatnonzero(target['y']==4)]
    if not all(len(g) for g in groups):raise ValueError('need S/SS/other/deleted TRAIN controls')
    rng=np.random.RandomState(p['seed']);plan=np.array([np.concatenate([rng.choice(g,4,replace=True) for g in groups]) for _ in range(p['steps'])]);np.save(out/'batch_plan.npy',plan)
    base=torch.load(p['base_artifact'],map_location='cpu');initial=None;report={'scope':p['scope'],'schema':SCHEMA,'steps':p['steps'],
        'TRAIN_groups':[len(g) for g in groups],'all_TRAIN_inputs':len(train['x']),'unique_sampled':len(np.unique(plan)),'arms':{}}
    for arm,split in [('grouped',False),('split_absence',True)]:
        encoder=load_encoder(p,True);encoder.encoder.layers[-2:].load_state_dict(base['last2'])
        head=PresenceReadout(encoder.config.hidden_size,split).to('cuda');head.load_base(base['head'])
        optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':.001},{'params':[v for v in encoder.parameters() if v.requires_grad],'lr':.00001}],weight_decay=.01)
        y=target['y'].copy()
        if not split:y[y==4]=3
        losses,phases,tick=[],[],time.monotonic();torch.cuda.reset_peak_memory_stats();initial_delta=0.
        for step,ix in enumerate(plan,1):
            optimizer.zero_grad(set_to_none=True)
            x=torch.as_tensor(train['x'][ix],device='cuda');length=torch.as_tensor(train['lengths'][ix],device='cuda')
            mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None]
            logits,phase=head(hidden(encoder,x,mask),mask,torch.as_tensor(train['relative_vowel'][ix],device='cuda'))
            if step==1:
                prob=logits.softmax(-1).detach().cpu()
                if split:prob=torch.cat([prob[:,:3],prob[:,3:].sum(-1,keepdim=True)],-1)
                if initial is None:initial=prob
                initial_delta=float((prob-initial).abs().max());torch.testing.assert_close(prob,initial,rtol=1e-6,atol=2e-7)
            cls=torch.nn.functional.cross_entropy(logits,torch.as_tensor(y[ix],device='cuda'))
            ph=joint_vowel_loss(phase,torch.as_tensor(target['phase'][ix],device='cuda'));loss=cls+p['phase_weight']*ph
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite presence loss')
            loss.backward()
            if step==1 and any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):raise AssertionError('earlier encoder updated')
            torch.nn.utils.clip_grad_norm_([v for g in optimizer.param_groups for v in g['params']],1.)
            optimizer.step();losses.append(float(cls.detach()));phases.append(float(ph.detach()))
            if step%100==0:print(arm,step,'class',float(np.mean(losses[-50:])),'phase',float(np.mean(phases[-50:])),flush=True)
        seconds=time.monotonic()-tick;ps,av,scoreseconds=scores(encoder,head,train)
        np.savez(out/(arm+'.train_scores.npz'),probabilities=ps,available=av)
        known=y>=0;pred=ps.argmax(-1)
        report['arms'][arm]={'initial_coarse_probability_max_delta':initial_delta,'train_seconds':seconds,'score_seconds':scoreseconds,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),
            'class_loss_first_last50':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],
            'phase_loss_first_last50':[float(np.mean(phases[:50])),float(np.mean(phases[-50:]))],
            'TRAIN_known':int(known.sum()),'TRAIN_correct':int((pred[known]==y[known]).sum()),'TRAIN_available':int(av.sum()),
            'TRAIN_confusion':{str(k):dict(collections.Counter(int(j) for j in pred[y==k])) for k in range(5 if split else 4)}}
        torch.save({'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'last2':{k:v.detach().cpu() for k,v in encoder.encoder.layers[-2:].state_dict().items()},
            'readout':'vowel_reference_presence','coordinate_schema':SCHEMA,'split_absence':split,'classes':list(CLASSES) if split else list(CLASSES[:3])+['other_including_TAG_deleted'],
            'base_artifact_sha256':service.sha(p['base_artifact'])},out/(arm+'.pt'))
        write(out/'report.json',report);del head,encoder,optimizer;torch.cuda.empty_cache()
    write(out/'finished.json',{'sha256':service.sha(out/'report.json')});print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

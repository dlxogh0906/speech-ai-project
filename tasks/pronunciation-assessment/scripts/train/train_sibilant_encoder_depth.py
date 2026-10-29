"""One-factor deeper local encoder adaptation versus completed last2 control.

Same learned1500 initialization,300batches, roles/actual losses and optimizer.
Keep temporal inputs zero; change only which encoder layers receive updates.
The first-pass decoder/canonical/ASR and all cached runtime windows stay fixed.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from train_sibilant_local_encoder import load_encoder,hidden,write
from train_sibilant_cv_support import phase_loss,score,calibrate
from train_sibilant_joint_vowel import aucs
from train_sibilant_role_time import head_for
from mdd.sibilant_role_time import SCHEMA


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text());torch.set_num_threads(2)
    base=torch.load(p['base_artifact'],map_location='cpu')
    train=dict(np.load(p['train_prefix']+'.npz'));cal=dict(np.load(p['cal_prefix']+'.npz'))
    if set(cal)!={'x','lengths','relative'}:raise ValueError('TAG input in inference')
    plan=np.load(p['batch_plan'])[:p['steps']];np.save(out/'batch_plan.npy',plan)
    np.testing.assert_array_equal(plan,np.load(Path(p['control_root'])/'batch_plan.npy'))
    control=json.loads((Path(p['control_root'])/'report.json').read_text())['arms']['zero']
    encoder=load_encoder(p,False);encoder.encoder.layers[-2:].load_state_dict(base['last2'])
    depth=p['encoder_tail_layers']
    if not 2<depth<=len(encoder.encoder.layers):raise ValueError('new update depth must exceed2')
    for layer in encoder.encoder.layers[-depth:]:layer.requires_grad_(True)
    head=head_for(base,False,'cuda')
    digest=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
    if digest!=control['initial_head_sha256']:raise AssertionError('initial head differs from last2 control')
    initial_tail=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in encoder.encoder.layers[-depth:].state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':.001},
        {'params':[v for v in encoder.parameters() if v.requires_grad],'lr':.00001}],weight_decay=.01)
    losses,auxes,tick=[],[],time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step,ix in enumerate(plan,1):
        optimizer.zero_grad(set_to_none=True)
        x=torch.as_tensor(train['x'][ix],device='cuda');length=torch.as_tensor(train['lengths'][ix],device='cuda')
        mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None]
        logits,roles=head(hidden(encoder,x,mask),mask,torch.as_tensor(train['relative'][ix],device='cuda'))
        actual=torch.nn.functional.cross_entropy(logits,torch.as_tensor(train['y'][ix],device='cuda'))
        aux=phase_loss(roles,torch.as_tensor(train['phase'][ix],device='cuda'));loss=actual+p['phase_weight']*aux
        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite depth loss')
        loss.backward()
        if step==1:
            if any(v.grad is not None for layer in encoder.encoder.layers[:-depth] for v in layer.parameters()):raise AssertionError('earlier layer gradient')
            if not all(any(v.grad is not None and bool(v.grad.abs().sum()) for v in layer.parameters()) for layer in encoder.encoder.layers[-depth:]):raise AssertionError('missing deeper layer gradient')
            if bool(head.temporal.weight.grad.abs().sum()):raise AssertionError('time features enabled')
        torch.nn.utils.clip_grad_norm_([v for g in optimizer.param_groups for v in g['params']],1.)
        optimizer.step();losses.append(float(actual.detach()));auxes.append(float(aux.detach()))
        if step%100==0:print('tail',depth,step,'class',float(np.mean(losses[-50:])),'phase',float(np.mean(auxes[-50:])),flush=True)
    train_seconds=time.monotonic()-tick;ps,av,seconds=score(encoder,head,cal)
    np.savez(out/'cal_scores.npz',probabilities=ps,available=av)
    labels=service.read_jsonl(p['cal_prefix']+'.labels.jsonl');features=service.read_jsonl(p['cal_prefix']+'.features.jsonl');records=service.read_jsonl(p['cal_prefix']+'.responses.jsonl')
    views={r['row']['uid']:observe(r['prior'],r['row']) for r in records};labels=[dict(r,existing=views[r['uid']][r['q']]) for r in labels]
    thresholds,audit=calibrate(labels,features,ps,av)
    for key in thresholds:
        if key.endswith('/affricate'):thresholds[key]=None
    candidate={'initial_head_sha256':digest,'initial_tail_sha256':initial_tail,'thresholds':thresholds,'calibration':audit,
        'additional_exact':sum(v['cal_additional_exact'] for k,v in audit.items() if k.endswith('/pair')),
        'unconditional_AUC':aucs(labels,ps,np.ones(len(ps),dtype=bool)),'support_available':int(av.sum()),
        'train_seconds':train_seconds,'score_seconds':seconds,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),
        'class_loss_first_last50':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],
        'phase_loss_first_last50':[float(np.mean(auxes[:50])),float(np.mean(auxes[-50:]))]}
    torch.save({'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
        'encoder_tail':{k:v.detach().cpu() for k,v in encoder.encoder.layers[-depth:].state_dict().items()},
        'encoder_tail_layers':depth,'readout':'role_time_tail','temporal_schema':SCHEMA,'temporal_active':False,
        'learned':True,'base_artifact_sha256':service.sha(p['base_artifact'])},out/'candidate.pt')
    report={'scope':p['scope'],'encoder_tail_layers':depth,'steps':p['steps'],'inference_batch':64,'test_scored':False,
        'all_CAL_utterances':len(records),'all_CAL_sites':len(labels),'all_CAL_features':len(features),
        'same_initial_head_and_batches':True,'arms':{'last2_control':control,'candidate':candidate},
        'selected_for_single_request':candidate['additional_exact']>max(0,control['additional_exact'])}
    write(out/'report.json',report);write(out/'finished.json',{'sha256':service.sha(out/'report.json')});print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

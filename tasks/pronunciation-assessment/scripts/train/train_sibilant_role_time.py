"""Matched frozen-acoustic probe: enable nine coarse-role moments versus zero.

Cache the unchanged encoder/role/acoustic path once. Fit only a zero-initialized
9x64 additive projection, preserving all old parameters and TRAIN examples.
Complete CAL is used for screening; test audio is never read here.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from train_sibilant_local_encoder import load_encoder, hidden, write
from train_sibilant_cv_support import calibrate
from train_sibilant_joint_vowel import aucs
from mdd.sibilant_cv_support import SupportReadout, reliable_support
from mdd.sibilant_role_time import RoleTimeReadout, SCHEMA


def head_for(base, active, device='cpu'):
    if base.get('readout', 'existing') != 'existing' or not base['learned']:
        raise ValueError('ordinary learned-support base artifact required')
    size = base['head']['support.0.weight'].shape[1]-1
    head = RoleTimeReadout(size, active).to(device)
    head.load_base(base['head'])
    return head


def prepare(p, root):
    out = root/'cache'
    out.mkdir(parents=True, exist_ok=False)
    base = torch.load(p['base_artifact'], map_location='cpu')
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(base['last2'])
    head = head_for(base, True, 'cuda')
    legacy = SupportReadout(encoder.config.hidden_size, True).to('cuda')
    legacy.load_state_dict(base['head'])
    report = {'schema':SCHEMA, 'TAG_inference_inputs':False, 'cohorts':{}}
    for name in ('train', 'cal'):
        stored = dict(np.load(p[name+'_prefix']+'.npz'))
        inputs = {k:stored[k] for k in ('x','lengths','relative')}
        zs, ts, ps, av = [], [], [], []
        tick = time.monotonic()
        with torch.no_grad():
            for i in range(0,len(inputs['x']),64):
                x = torch.as_tensor(inputs['x'][i:i+64],device='cuda')
                length = torch.as_tensor(inputs['lengths'][i:i+64],device='cuda')
                mask = torch.arange(x.shape[1],device='cuda')[None,:] < length[:,None]
                rel = torch.as_tensor(inputs['relative'][i:i+64],device='cuda')
                h = hidden(encoder,x,mask)
                z,t,roles = head.terms(h,mask,rel)
                original, original_roles = legacy(h,mask,rel)
                torch.testing.assert_close(roles,original_roles,rtol=0,atol=0)
                torch.testing.assert_close(head.from_terms(z,t),original,rtol=0,atol=0)
                zs.extend(z.cpu().numpy());ts.extend(t.cpu().numpy())
                ps.extend(original.softmax(-1).cpu().numpy())
                av.extend(reliable_support(roles,mask).cpu().tolist())
        value = dict(acoustic=np.asarray(zs),temporal=np.asarray(ts),base_probabilities=np.asarray(ps),available=np.asarray(av))
        np.savez(out/(name+'.npz'),**value)
        report['cohorts'][name] = {'all_inputs':len(zs),'seconds':time.monotonic()-tick,
                                  'initial_logits_and_roles_exact':True,'sha256':service.sha(out/(name+'.npz'))}
        print('cached',name,report['cohorts'][name],flush=True)
    reference = np.load(p['base_cal_scores'])
    np.testing.assert_array_equal(value['base_probabilities'],reference['probabilities'])
    np.testing.assert_array_equal(value['available'],reference['available_strict'])
    report['prior_complete_CAL_predictions_reproduced_exactly'] = True
    write(out/'report.json',report)
    write(out/'finished.json',{'sha256':service.sha(out/'report.json')})


def fit(p, root):
    cache, out = root/'cache', root/'fit'
    if json.loads((cache/'finished.json').read_text())['sha256']!=service.sha(cache/'report.json'):
        raise ValueError('incomplete cached terms')
    out.mkdir(parents=True, exist_ok=False)
    base = torch.load(p['base_artifact'],map_location='cpu')
    data = np.load(cache/'train.npz')
    y = np.load(p['train_prefix']+'.npz')['y']
    plan = np.load(p['batch_plan'])[:p['steps']]
    if len(plan)!=p['steps'] or bool((y[plan]<0).any()):
        raise ValueError('fixed TRAIN plan does not cover known classes')
    np.save(out/'batch_plan.npy',plan)
    acoustic = torch.as_tensor(data['acoustic'])
    temporal = torch.as_tensor(data['temporal'])
    target = torch.as_tensor(y)
    initial, report = None, {'schema':SCHEMA,'steps':p['steps'],'all_old_parameters_frozen':True,'arms':{}}
    for arm, active in [('zero',False),('temporal',True)]:
        head = head_for(base,active)
        head.requires_grad_(False)
        head.temporal.weight.requires_grad_(True)
        digest = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
        if initial is None:
            initial = digest
        if digest!=initial:
            raise AssertionError('initial head parameters changed between arms')
        optimizer = torch.optim.AdamW([head.temporal.weight],lr=p['learning_rate'],weight_decay=.01)
        losses, tick = [], time.monotonic()
        for step, indices in enumerate(plan,1):
            optimizer.zero_grad(set_to_none=True)
            logits = head.from_terms(acoustic[indices],temporal[indices])
            loss = torch.nn.functional.cross_entropy(logits,target[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite probe loss')
            loss.backward()
            if step==1:
                has_grad = bool(head.temporal.weight.grad.abs().sum())
                if has_grad!=active:
                    raise AssertionError('zero/active feature gradients are not isolated')
            optimizer.step()
            losses.append(float(loss.detach()))
        for key,value in base['head'].items():
            torch.testing.assert_close(head.state_dict()[key],value,rtol=0,atol=0)
        if not active and bool(head.temporal.weight.detach().count_nonzero()):
            raise AssertionError('zero-feature control changed')
        artifact = {'head':head.state_dict(),'last2':base['last2'],'learned':True,
                    'readout':'role_time','temporal_schema':SCHEMA,'temporal_active':active,
                    'base_artifact_sha256':service.sha(p['base_artifact'])}
        torch.save(artifact,out/(arm+'.pt'))
        report['arms'][arm] = {'initial_head_sha256':initial,'train_seconds':time.monotonic()-tick,
                              'class_loss_first_last50':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],
                              'temporal_weight_norm':float(head.temporal.weight.detach().norm()),
                              'trainable_parameters':int(head.temporal.weight.numel()),'old_head_unchanged':True}
        print('fit',arm,report['arms'][arm],flush=True)
    write(out/'report.json',report)
    write(out/'finished.json',{'sha256':service.sha(out/'report.json')})


def screen(p, root):
    out = root/'screen'
    out.mkdir(parents=True,exist_ok=False)
    base = torch.load(p['base_artifact'],map_location='cpu')
    data = np.load(root/'cache/cal.npz')
    records = service.read_jsonl(p['cal_prefix']+'.responses.jsonl')
    labels = service.read_jsonl(p['cal_prefix']+'.labels.jsonl')
    features = service.read_jsonl(p['cal_prefix']+'.features.jsonl')
    prior = {r['row']['uid']:observe(r['prior'],r['row']) for r in records}
    labels = [dict(r,existing=prior[r['uid']][r['q']]) for r in labels]
    report = {'schema':SCHEMA,'inference_batch':64,'test_scored':False,'all_CAL_utterances':len(records),
              'all_CAL_sites':len(labels),'all_input_features':len(features),'arms':{}}
    for arm in ('zero','temporal'):
        artifact = torch.load(root/('fit/'+arm+'.pt'),map_location='cpu')
        head = head_for(base,artifact['temporal_active'],'cuda')
        head.load_state_dict(artifact['head'])
        probabilities = []
        with torch.no_grad():
            for i in range(0,len(data['acoustic']),64):
                z = torch.as_tensor(data['acoustic'][i:i+64],device='cuda')
                t = torch.as_tensor(data['temporal'][i:i+64],device='cuda')
                probabilities.extend(head.from_terms(z,t).softmax(-1).cpu().numpy())
        ps = np.asarray(probabilities)
        if arm=='zero':
            np.testing.assert_array_equal(ps,data['base_probabilities'])
        av = data['available']
        np.savez(out/(arm+'.cal_scores.npz'),probabilities=ps,available=av)
        thresholds,audit = calibrate(labels,features,ps,av)
        for key in thresholds:
            if key.endswith('/affricate'):thresholds[key]=None
        report['arms'][arm] = {'thresholds':thresholds,'calibration':audit,
            'additional_exact':sum(v['cal_additional_exact'] for k,v in audit.items() if k.endswith('/pair')),
            'unconditional_AUC':aucs(labels,ps,np.ones(len(ps),dtype=bool)),'available':int(av.sum())}
    report['selected_for_single_request'] = report['arms']['temporal']['additional_exact']>max(0,report['arms']['zero']['additional_exact'])
    write(out/'report.json',report)
    write(out/'finished.json',{'sha256':service.sha(out/'report.json')})
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True);ap.add_argument('--root',required=True)
    ap.add_argument('--stage',required=True,choices=('prepare','fit','screen'))
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());root = Path(a.root)
    for path,digest in p['sha256'].items():
        if service.sha(path)!=digest:
            raise ValueError('frozen source changed: '+path)
    torch.set_num_threads(2)
    {'prepare':prepare,'fit':fit,'screen':screen}[a.stage](p,root)


if __name__=='__main__':
    main()

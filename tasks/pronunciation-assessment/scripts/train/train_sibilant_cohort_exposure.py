"""Fixed conditional heads: factorial new errors and same-cohort normal exposure."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_conditional_retained import inputs,score,write,head_digest
from train_sibilant_hubert_adaptation import retained_subgroups
from train_sibilant_extra_normals import proposed
from mdd.sibilant_conditional_retained import (TOKENS,new_conditional,conditional_artifact,
    load_conditional,fixed_direction_proxy)
from mdd.sibilant_cohort_exposure import ARMS,cohort_plans
from mdd.sibilant_hubert_adaptation import tensor_digest


def train(p,base,data,y,plans,device,name):
    torch.manual_seed(p['seed']);model = new_conditional(base,p['binding'],device)
    parameters = [[v for v in h.parameters() if v.requires_grad] for h in model.heads]
    if any(sum(v.numel() for v in values)!=50117 for values in parameters):
        raise ValueError('conditional actual5 head capacity differs')
    if any(tensor_digest(h.head.state_dict())!=tensor_digest(base['head']) for h in model.heads):
        raise ValueError('conditional initialization is not original parent')
    opts = [torch.optim.AdamW(z,lr=p['head_lr'],weight_decay=.01) for z in parameters]
    losses = [[],[]]
    for step in range(p['cohort_steps']):
        for branch,token in enumerate(TOKENS):
            ix = plans[name][token][step]
            if (y[ix]<0).any():raise ValueError('unknown actual class in optimizer')
            opt,head = opts[branch],model.heads[branch];opt.zero_grad(set_to_none=True)
            logits = head.from_summary(torch.as_tensor(data['summary'][ix],device=device),
                                       torch.as_tensor(data['moments'][ix],device=device))
            loss = torch.nn.functional.cross_entropy(logits,torch.as_tensor(y[ix],device=device))
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite actual loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters[branch],1.);opt.step()
            losses[branch].append(float(loss.detach()))
        if (step+1)%100==0:print(json.dumps({'arm':name,'updates_per_head':step+1}),flush=True)
    artifact = conditional_artifact(model,base)
    return model,artifact,{'actual_optimizer_updates':p['cohort_steps']*2,
        'updates_per_head':[len(z) for z in losses],'trainable_parameters':100234,
        'loss_first_last50':[[float(np.mean(z[:50])),float(np.mean(z[-50:]))] for z in losses],
        'frozen_base_and_roles_unchanged':True}


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--stage',required=True,choices=['preflight','fit','audit'])
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());out = Path(a.out)
    if p['cohort_candidate']!=ARMS[-1] or p['cohort_steps']!=400:
        raise ValueError('fixed cohort candidate or update count differs')
    data,y,features,labels,indexed,offset,_,_,_ = inputs(p)
    plans,plan_report = cohort_plans(np.load(p['old_plan']),y,indexed,features,
        service.read_jsonl(p['old_ctc_targets']),offset,p['seed'],steps=p['cohort_steps'])
    if tuple(plan_report[t]['normal_features_added'] for t in TOKENS)!=(620,256):
        raise ValueError('all recovered normal target counts differ')
    torch.set_num_threads(2);device = torch.device(p['device'])
    base = torch.load(p['control_artifact'],map_location='cpu')
    parent = new_conditional(base,p['binding'],device)
    parent_ps,av = score(parent,data,features,offset,device,True)
    np.testing.assert_array_equal(parent_ps[:offset],np.load(p['old_scores'])['probabilities'])
    np.testing.assert_array_equal(parent_ps[offset:],np.load(Path(p['new_summary'])/'summary.npz')['probabilities'])
    by_old = {i:z for i,z in indexed.items() if i<offset}
    parent_policy = fixed_direction_proxy(parent_ps[:offset],av[:offset],y[:offset],features[:offset],by_old)
    if (parent_policy['novel_TP'],parent_policy['normal_FP'],parent_policy['wrong_actual'])!=(564,87,3):
        raise ValueError('parent fixed direction diagnosis differs')
    distributions = {}
    for name,branches in plans.items():
        unique = set(np.concatenate(list(branches.values())).ravel().tolist())
        distributions[name] = {key:dict(collections.Counter(str(indexed[i].get(key)) for i in unique))
                               for key in ('following','position','item','language')}
        distributions[name]['speakers'] = len({indexed[i]['speaker'] for i in unique})
        distributions[name]['recordings'] = len({indexed[i]['uid'] for i in unique})
    if a.stage=='preflight':
        out.mkdir(parents=True,exist_ok=False)
        for name,branches in plans.items():
            for branch,token in enumerate(TOKENS):np.save(out/(name+'.'+str(branch)+'.plan.npy'),branches[token])
        write(out/'report.json',{'passed':True,'plans':plan_report,'distributions':distributions,
            'parent_original_new_scores_bitexact':True,'actual_optimizer_updates':0,
            'protocol_sha256':service.sha(a.protocol)})
        return
    preflight = Path(p['cohort_preflight'])
    check = json.loads((preflight/'report.json').read_text())
    if not check['passed'] or check['protocol_sha256']!=service.sha(a.protocol):
        raise ValueError('frozen cohort preflight missing')
    for name,branches in plans.items():
        for branch,token in enumerate(TOKENS):
            np.testing.assert_array_equal(branches[token],np.load(preflight/(name+'.'+str(branch)+'.plan.npy')))
    if a.stage=='audit':
        audit = {'passed':True,'arms':{},'production_changed':False}
        for name in ARMS:
            model,_,training = train(p,base,data,y,plans,device,name)
            restored = load_conditional(torch.load(out/(name+'.pt'),map_location='cpu'),p['binding'],device)
            if head_digest(model,True)!=head_digest(restored,True):raise ValueError('reexecuted weights differ')
            ps,valid = score(restored,data,features,offset,device,True)
            ref = np.load(out/(name+'.scores.npz'))
            np.testing.assert_array_equal(ps,ref['probabilities']);np.testing.assert_array_equal(valid,ref['available'])
            audit['arms'][name] = {'actual_reexecution_updates':training['actual_optimizer_updates'],
                'reexecuted_weights_bitexact':True,'whole_reload_scores_bitexact':True}
        write(out/'final_audit.json',audit);return
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    report = {'scope':p['cohort_scope'],'plans':plan_report,'distributions':distributions,'arms':{},
        'parent_fixed_direction_TRAIN':parent_policy,'original_retained_GT_sites':14002,
        'original_unknown':int((y[:offset]<0).sum()),
        'original_no_input':sum(z['feature_index'] is None for z in labels[:14002]),
        'extra_source_error_denominator':728,'supported_new_errors':675,'supported_new_normals':876,
        'production_changed':False,'service_improvement_proven':False,'reserved_scored':False,'goal_complete':False}
    all_ps,policies = {'parent':parent_ps},{'parent':parent_policy}
    for name in ARMS:
        tick = time.monotonic()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        model,artifact,training = train(p,base,data,y,plans,device,name)
        artifact['training_provenance'] = dict(training,arm=name,protocol_sha256=service.sha(a.protocol))
        torch.save(artifact,out/(name+'.pt'))
        restored = load_conditional(torch.load(out/(name+'.pt'),map_location='cpu'),p['binding'],device)
        ps,valid = score(model,data,features,offset,device,True)
        # Whole-population reload, preserving original/new batch64 boundaries.
        reload_ps,reload_av = score(restored,data,features,offset,device,True)
        np.testing.assert_array_equal(ps,reload_ps);np.testing.assert_array_equal(valid,reload_av)
        np.savez(out/(name+'.scores.npz'),probabilities=ps,available=valid)
        policy = fixed_direction_proxy(ps[:offset],av[:offset],y[:offset],features[:offset],by_old)
        write(out/(name+'.original.subgroups.json'),retained_subgroups(ps[:offset],av[:offset],y[:offset],features[:offset],labels[:14002],policy))
        write(out/(name+'.all.subgroups.json'),retained_subgroups(ps,av,y,features,labels,policy))
        report['arms'][name] = dict(training,TRAIN_proxy=policy,whole_reload_bitexact=True,
            seconds_fit_and_score=time.monotonic()-tick,artifact_sha256=service.sha(out/(name+'.pt')),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        all_ps[name],policies[name] = ps,policy
        write(out/'report.json',report)
    candidate = policies[ARMS[-1]]
    report['proceed_CAL'] = bool(candidate['novel_TP']>max(z['novel_TP'] for name,z in policies.items() if name!=ARMS[-1])
        and candidate['S_to_SS']['novel_TP']>=240 and candidate['SS_to_S']['novel_TP']>=324 and candidate['wrong_actual']<=3)
    changes = {name:collections.Counter() for name in ARMS}
    with (out/'paired_proposal_ledger.jsonl').open('x') as stream:
        for z in labels:
            calls = {name:proposed(z,ps,av,features,policies[name]) for name,ps in all_ps.items()}
            prefix = 'normal' if z['label_kind']=='correct' else 'error_or_unknown'
            for name,counter in changes.items():
                counter[prefix+'_added_calls'] += calls[name] and not calls['parent']
                counter[prefix+'_lost_calls'] += calls['parent'] and not calls[name]
            service.write_line(stream,{'uid':z['uid'],'q_runtime':z.get('q_runtime'),
                'feature_index':z['feature_index'],'label_kind':z['label_kind'],'proposals':calls})
    report['all_retained_call_changes'] = {k:dict(v) for k,v in changes.items()}
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({'finished':True,'proceed_CAL':report['proceed_CAL'],
        'results':{k:v['TRAIN_proxy'] for k,v in report['arms'].items()}}),flush=True)


if __name__=='__main__':main()

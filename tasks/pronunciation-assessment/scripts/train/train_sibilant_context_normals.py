"""Frozen actual5 conditional heads with matched normal-context data controls."""
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
from train_sibilant_cohort_exposure import train
from train_sibilant_hubert_adaptation import retained_subgroups
from train_sibilant_extra_normals import proposed
from mdd.sibilant_conditional_retained import (TOKENS,new_conditional,load_conditional,fixed_direction_proxy)
from mdd.sibilant_context_normals import ARMS,normal_context_plans


def load(p):
    data,y,features,labels,indexed,offset,_,_,_ = inputs(p)
    start = len(y);folder = Path(p['context_summary']);extra = np.load(folder/'summary.npz')
    data = {k:np.concatenate([data[k],extra[k]]) for k in ('summary','moments','available')}
    y = np.concatenate([y,extra['y']])
    features += [dict(z,feature_index=z['feature_index']+start) for z in service.read_jsonl(folder/'data.features.jsonl')]
    labels += [dict(z,feature_index=None if z['feature_index'] is None else z['feature_index']+start)
               for z in service.read_jsonl(folder/'data.labels.jsonl')]
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    if start!=15412 or len(indexed)!=len(features) or len(features)!=len(y):
        raise ValueError('three source feature populations differ')
    if len(indexed)!=sum(z['feature_index'] is not None for z in labels):raise ValueError('duplicate feature labels')
    for i,f in enumerate(features):
        if f['feature_index']!=i or any(indexed[i][k]!=f[k] for k in ('uid','q_runtime')):
            raise ValueError('source feature identities differ')
    if not all(np.isfinite(data[k]).all() for k in ('summary','moments')):raise ValueError('nonfinite source summary')
    plans,report = normal_context_plans(np.load(p['old_plan']),y,indexed,features,
        service.read_jsonl(p['old_ctc_targets']),offset,start,p['seed'])
    distributions = {}
    for name,branches in plans.items():
        unique = set(np.concatenate(list(branches.values())).ravel().tolist())
        distributions[name] = {key:dict(collections.Counter(str(indexed[i].get(key)) for i in unique))
                               for key in ('following','position','item','language')}
        distributions[name]['speakers'] = len({indexed[i]['speaker'] for i in unique})
        distributions[name]['recordings'] = len({indexed[i]['uid'] for i in unique})
    return data,y,features,labels,indexed,offset,start,plans,report,distributions


def scores(model,data,features,offset,start,device):
    # Each original extraction scored its own final short batch. Preserve all
    # three boundaries, including the 1671-input middle source.
    parts = []
    for left,right in ((0,offset),(offset,start),(start,len(features))):
        parts.append(score(model,{k:v[left:right] for k,v in data.items()},
            features[left:right],right-left,device,True))
    return tuple(np.concatenate([z[k] for z in parts]) for k in (0,1))


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--stage',required=True,choices=['preflight','fit','audit'])
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());out = Path(a.out)
    if p['context_candidate']!=ARMS[-1] or p['cohort_steps']!=400:
        raise ValueError('fixed context-normal candidate/update count differs')
    data,y,features,labels,indexed,offset,start,plans,plan_report,distributions = load(p)
    torch.set_num_threads(2);device = torch.device(p['device'])
    base = torch.load(p['control_artifact'],map_location='cpu')
    parent = new_conditional(base,p['binding'],device);parent_ps,av = scores(parent,data,features,offset,start,device)
    references = [np.load(p['old_scores'])['probabilities'],
        np.load(Path(p['new_summary'])/'summary.npz')['probabilities'],
        np.load(Path(p['context_summary'])/'summary.npz')['probabilities']]
    np.testing.assert_array_equal(parent_ps,np.concatenate(references));np.testing.assert_array_equal(av,data['available'])
    old_indexed = {i:z for i,z in indexed.items() if i<offset}
    parent_policy = fixed_direction_proxy(parent_ps[:offset],av[:offset],y[:offset],features[:offset],old_indexed)
    if (parent_policy['novel_TP'],parent_policy['normal_FP'],parent_policy['wrong_actual'])!=(564,87,3):
        raise ValueError('parent fixed-direction gate changed')
    if a.stage=='preflight':
        out.mkdir(parents=True,exist_ok=False)
        for name,branches in plans.items():
            for branch,token in enumerate(TOKENS):np.save(out/(name+'.'+str(branch)+'.plan.npy'),branches[token])
        write(out/'report.json',{'passed':True,'plans':plan_report,'distributions':distributions,
            'all_three_sources_initial_predictions_bitexact':True,'source_boundaries':[offset,start,len(features)],
            'actual_optimizer_updates':0,'protocol_sha256':service.sha(a.protocol)})
        return
    preflight = Path(p['context_preflight']);check = json.loads((preflight/'report.json').read_text())
    if not check['passed'] or check['protocol_sha256']!=service.sha(a.protocol):raise ValueError('frozen preflight missing')
    for name,branches in plans.items():
        for branch,token in enumerate(TOKENS):
            np.testing.assert_array_equal(branches[token],np.load(preflight/(name+'.'+str(branch)+'.plan.npy')))
    if a.stage=='audit':
        audit = {'passed':True,'arms':{},'production_changed':False}
        for name in ARMS:
            model,_,training = train(p,base,data,y,plans,device,name)
            restored = load_conditional(torch.load(out/(name+'.pt'),map_location='cpu'),p['binding'],device)
            if head_digest(model,True)!=head_digest(restored,True):raise ValueError('reexecuted head weights differ')
            ps,valid = scores(restored,data,features,offset,start,device);ref = np.load(out/(name+'.scores.npz'))
            np.testing.assert_array_equal(ps,ref['probabilities']);np.testing.assert_array_equal(valid,ref['available'])
            audit['arms'][name] = {'actual_reexecution_updates':training['actual_optimizer_updates'],
                'reexecuted_heads_bitexact':True,'whole_three_source_reload_scores_bitexact':True}
        write(out/'final_audit.json',audit);return
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    report = {'scope':p['context_scope'],'plans':plan_report,'distributions':distributions,'arms':{},
        'parent_fixed_direction_TRAIN':parent_policy,'source_boundaries':[offset,start,len(features)],
        'original_all_GT_sites':14002,'all_retained_label_records':len(labels),'source_pair_errors_retained':728,
        'original_unknown_inputs':int((y[:offset]<0).sum()),
        'original_no_input_sites':sum(z['feature_index'] is None for z in labels[:14002]),
        'production_changed':False,'service_improvement_proven':False,'reserved_scored':False,'goal_complete':False}
    np.savez(out/'parent.scores.npz',probabilities=parent_ps,available=av)
    all_ps,policies = {'parent':parent_ps},{'parent':parent_policy}
    for name in ARMS:
        tick = time.monotonic()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        model,artifact,training = train(p,base,data,y,plans,device,name)
        artifact['training_provenance'] = dict(training,arm=name,protocol_sha256=service.sha(a.protocol))
        torch.save(artifact,out/(name+'.pt'))
        ps,valid = scores(model,data,features,offset,start,device)
        restored = load_conditional(torch.load(out/(name+'.pt'),map_location='cpu'),p['binding'],device)
        reload_ps,reload_av = scores(restored,data,features,offset,start,device)
        np.testing.assert_array_equal(ps,reload_ps);np.testing.assert_array_equal(valid,reload_av)
        np.savez(out/(name+'.scores.npz'),probabilities=ps,available=valid)
        policy = fixed_direction_proxy(ps[:offset],av[:offset],y[:offset],features[:offset],old_indexed)
        write(out/(name+'.original.subgroups.json'),retained_subgroups(ps[:offset],av[:offset],y[:offset],features[:offset],labels[:14002],policy))
        write(out/(name+'.all.subgroups.json'),retained_subgroups(ps,av,y,features,labels,policy))
        report['arms'][name] = dict(training,TRAIN_proxy=policy,whole_three_sources_reload_bitexact=True,
            seconds_fit_score=time.monotonic()-tick,artifact_sha256=service.sha(out/(name+'.pt')),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        all_ps[name],policies[name] = ps,policy;write(out/'report.json',report)
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

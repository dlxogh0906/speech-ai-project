"""Fixed 2x2 shared/conditional-head and retained-old/new-error TRAIN trial."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_partial_errors import write
from train_sibilant_hubert_adaptation import retained_subgroups
from train_sibilant_extra_normals import common_scores, proposed
from train_sibilant_precision_tail import summary_scores
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_precision_tail import PrecisionTailHead, tail_artifact, load_tail_head
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_conditional_retained import (TOKENS, retained_plans, new_conditional,
    conditional_artifact, load_conditional, fixed_direction_proxy)

ARMS = ('retained_shared_old','retained_shared_new',
        'retained_conditional_old','retained_conditional_new')


def inputs(p):
    for name,digest in p['sha256'].items():
        if service.sha(name)!=digest:
            raise ValueError('frozen input/source changed: '+name)
    old, new = np.load(p['old_summary']), np.load(Path(p['new_summary'])/'summary.npz')
    data = {k:np.concatenate([old[k],new[k]]) for k in ('summary','moments','available')}
    offset = len(old['summary'])
    old_y = np.load(p['old_targets'])['y']; y = np.concatenate([old_y,new['y']])
    features = service.read_jsonl(p['old_features'])+[dict(z,feature_index=z['feature_index']+offset)
        for z in service.read_jsonl(Path(p['new_summary'])/'data.features.jsonl')]
    old_labels = service.read_jsonl(p['old_labels'])
    labels = old_labels+[dict(z,feature_index=None if z['feature_index'] is None else z['feature_index']+offset)
        for z in service.read_jsonl(Path(p['new_summary'])/'data.labels.jsonl')]
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    if (offset!=13741 or len(old_labels)!=14002 or len(y)!=len(features)
            or len(indexed)!=len(features)
            or sum(z['feature_index'] is not None for z in labels)!=len(indexed)):
        raise ValueError('original/extra populations or unique feature labels differ')
    if not all(features[i]['feature_index']==i and indexed[i]['uid']==features[i]['uid']
               and indexed[i]['q_runtime']==features[i]['q_runtime'] for i in range(len(features))):
        raise ValueError('feature/label identity differs')
    if not all(np.isfinite(data[k]).all() for k in ('summary','moments')):
        raise ValueError('nonfinite summary observation')
    eligibility = service.read_jsonl(p['old_ctc_targets'])
    plans, report = retained_plans(np.load(p['old_plan']),y,indexed,features,eligibility,offset,p['seed'])
    expected = {'ㅅ':(3279,481,297,327),'ㅆ':(1445,350,378,1070)}
    for token, values in expected.items():
        z = report[token]
        if tuple(z[k] for k in ('original_stream_draws','original_unique_errors_retained',
            'new_error_features','post_original_pair_slots'))!=values:
            raise ValueError('predeclared full-data exposure feasibility differs')
    distributions = {}
    for token in TOKENS:
        for arm in ('old','new'):
            unique = set(plans[token][arm].ravel().tolist())
            distributions[token+':'+arm] = {
                key:dict(collections.Counter(str(indexed[i].get(key)) for i in unique))
                for key in ('following','position','item','language')}
            distributions[token+':'+arm]['recordings'] = len({indexed[i]['uid'] for i in unique})
            distributions[token+':'+arm]['speakers'] = len({indexed[i]['speaker'] for i in unique})
    return data,y,features,labels,indexed,offset,plans,report,distributions


def score(model, data, features, offset, device, conditional):
    if not conditional:
        if offset==len(features):
            return summary_scores(model,data,device)
        return common_scores(model,data,offset,device)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for left,right in ((0,offset),(offset,len(features))):
            for start in range(left,right,64):
                stop = min(right,start+64)
                routing = torch.tensor([TOKENS.index(f['canonical']) for f in features[start:stop]],
                                       dtype=torch.long,device=device)
                logits = model.from_summary(torch.as_tensor(data['summary'][start:stop],device=device),
                    torch.as_tensor(data['moments'][start:stop],device=device),routing)
                probabilities.extend(logits.softmax(-1).cpu().numpy())
    return np.asarray(probabilities),data['available'].copy()


def fit_arm(name,p,base,data,y,plans,device):
    torch.manual_seed(p['seed'])
    conditional = 'conditional' in name
    model = (new_conditional(base,p['binding'],device) if conditional else
        PrecisionTailHead(load_probe(base,p['binding'],device),'ce_full').to(device).eval())
    heads = list(model.heads) if conditional else [model]
    parameters = [[v for v in head.parameters() if v.requires_grad] for head in heads]
    if any(sum(v.numel() for v in values)!=50117 for values in parameters):
        raise ValueError('original head capacity differs')
    initial = [tensor_digest(head.head.state_dict()) for head in heads]
    if any(value!=tensor_digest(base['head']) for value in initial):
        raise ValueError('not initialized from original parent')
    optimizers = [torch.optim.AdamW(values,lr=p['head_lr'],weight_decay=.01) for values in parameters]
    data_arm = 'new' if name.endswith('_new') else 'old'
    losses, updates = [[] for _ in heads], [0 for _ in heads]
    for step in range(300):
        for branch, token in enumerate(TOKENS):
            which = branch if conditional else 0
            head,optimizer,values = heads[which],optimizers[which],parameters[which]
            ix = plans[token][data_arm][step]
            if (ix<0).any() or (y[ix]<0).any():
                raise ValueError('unknown training input')
            optimizer.zero_grad(set_to_none=True)
            logits = head.from_summary(torch.as_tensor(data['summary'][ix],device=device),
                                       torch.as_tensor(data['moments'][ix],device=device))
            loss = torch.nn.functional.cross_entropy(logits,torch.as_tensor(y[ix],device=device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite actual-phone loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(values,1.);optimizer.step()
            losses[which].append(float(loss.detach()));updates[which]+=1
        if (step+1)%100==0:
            print(json.dumps({'arm':name,'optimizer_updates':sum(updates),'branch_updates':updates}),flush=True)
    if any(tensor_digest(head.head.state_dict())==value for head,value in zip(heads,initial)):
        raise ValueError('head did not learn')
    # Export also checks the immutable parent/PCA/support tensors in each branch.
    artifact = conditional_artifact(model,base) if conditional else tail_artifact(model,base)
    training = {'arm':name,'actual_optimizer_updates':sum(updates),'updates_per_head':updates,
        'trainable_parameters':sum(sum(v.numel() for v in values) for values in parameters),
        'class_loss_first_last50':[[float(np.mean(z[:50])),float(np.mean(z[-50:]))] for z in losses],
        'initial_parent_sha256':initial,'frozen_base_and_roles_unchanged':True}
    return model,artifact,training


def head_digest(model, conditional):
    return [tensor_digest(z.head.state_dict()) for z in model.heads] if conditional else [tensor_digest(model.head.state_dict())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--stage',required=True,choices=['preflight','fit','audit'])
    a = ap.parse_args();p = json.loads(Path(a.protocol).read_text());out = Path(a.out)
    if p['preselected_candidate']!=ARMS[-1] or p['steps']!=300:
        raise ValueError('predeclared candidate/update count differs')
    data,y,features,labels,indexed,offset,plans,plan_report,distributions = inputs(p)
    torch.set_num_threads(2);device = torch.device(p['device'])
    base = torch.load(p['control_artifact'],map_location='cpu')
    parent = PrecisionTailHead(load_probe(base,p['binding'],device),'ce_full').to(device).eval()
    parent_ps,av = score(parent,data,features,offset,device,False)
    np.testing.assert_array_equal(parent_ps[:offset],np.load(p['old_scores'])['probabilities'])
    np.testing.assert_array_equal(parent_ps[offset:],np.load(Path(p['new_summary'])/'summary.npz')['probabilities'])
    initial = new_conditional(base,p['binding'],device)
    initial_ps,initial_av = score(initial,data,features,offset,device,True)
    np.testing.assert_array_equal(initial_ps,parent_ps);np.testing.assert_array_equal(initial_av,av)
    old_indexed = {i:z for i,z in indexed.items() if i<offset}
    parent_policy = fixed_direction_proxy(parent_ps[:offset],av[:offset],y[:offset],features[:offset],old_indexed)
    if (parent_policy['novel_TP'],parent_policy['normal_FP'],parent_policy['wrong_actual'])!=(564,87,3):
        raise ValueError('fixed-direction parent does not reproduce')
    del initial,parent
    if a.stage=='preflight':
        out.mkdir(parents=True,exist_ok=False)
        for branch,token in enumerate(TOKENS):
            for arm in ('old','new'):
                np.save(out/('%s.%s.plan.npy'%(branch,arm)),plans[token][arm])
        write(out/'report.json',{'passed':True,'plans':plan_report,'distributions':distributions,
            'initial_all_original_and_new_predictions_bitexact':True,'parent_policy':parent_policy,
            'actual_optimizer_updates':0,'protocol_sha256':service.sha(a.protocol)})
        return
    preflight = Path(p['preflight'])
    check = json.loads((preflight/'report.json').read_text())
    if not check['passed'] or check['protocol_sha256']!=service.sha(a.protocol):
        raise ValueError('frozen successful preflight missing')
    for branch,token in enumerate(TOKENS):
        for arm in ('old','new'):
            np.testing.assert_array_equal(plans[token][arm],np.load(preflight/('%s.%s.plan.npy'%(branch,arm))))
    if a.stage=='audit':
        audit = {'passed':True,'arms':{},'production_changed':False}
        for name in ARMS:
            conditional = 'conditional' in name
            fresh,artifact,training = fit_arm(name,p,base,data,y,plans,device)
            saved = torch.load(out/(name+'.pt'),map_location='cpu')
            restored = load_conditional(saved,p['binding'],device) if conditional else load_tail_head(saved,p['binding'],'ce_full',device)
            if head_digest(fresh,conditional)!=head_digest(restored,conditional):
                raise ValueError('independent training weights differ: '+name)
            ps,valid = score(restored,data,features,offset,device,conditional)
            reference = np.load(out/(name+'.scores.npz'))
            np.testing.assert_array_equal(ps,reference['probabilities']);np.testing.assert_array_equal(valid,reference['available'])
            audit['arms'][name] = {'reexecuted_weights_bitexact':True,'whole_scores_reload_bitexact':True,
                'actual_reexecution_updates':training['actual_optimizer_updates']}
            del fresh,restored
        write(out/'final_audit.json',audit);return
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    report = {'scope':p['scope'],'plans':plan_report,'distributions':distributions,'arms':{},
        'parent_fixed_direction_TRAIN':parent_policy,'original_all_GT_sites':14002,
        'original_unknown_inputs':int((y[:offset]<0).sum()),
        'original_no_input_sites':sum(z['feature_index'] is None for z in labels[:14002]),
        'extra_source_errors_retained':728,'extra_supported_pair_errors':675,
        'service_improvement_proven':False,'production_changed':False,'reserved_scored':False,'goal_complete':False}
    all_ps,policies = {'parent':parent_ps},{'parent':parent_policy}
    for name in ARMS:
        tick = time.monotonic();conditional = 'conditional' in name
        torch.cuda.reset_peak_memory_stats(device) if device.type=='cuda' else None
        model,artifact,training = fit_arm(name,p,base,data,y,plans,device)
        artifact['training_provenance'] = dict(training,protocol_sha256=service.sha(a.protocol))
        torch.save(artifact,out/(name+'.pt'))
        ps,valid = score(model,data,features,offset,device,conditional)
        saved = torch.load(out/(name+'.pt'),map_location='cpu')
        restored = load_conditional(saved,p['binding'],device) if conditional else load_tail_head(saved,p['binding'],'ce_full',device)
        for start in (0,offset):
            subset = {k:v[start:start+64] for k,v in data.items()}
            check_ps,check_av = score(restored,subset,features[start:start+64],64,device,conditional)
            np.testing.assert_array_equal(check_ps,ps[start:start+64]);np.testing.assert_array_equal(check_av,av[start:start+64])
        np.savez(out/(name+'.scores.npz'),probabilities=ps,available=valid)
        policy = fixed_direction_proxy(ps[:offset],valid[:offset],y[:offset],features[:offset],old_indexed)
        write(out/(name+'.original.subgroups.json'),retained_subgroups(ps[:offset],valid[:offset],y[:offset],features[:offset],labels[:14002],policy))
        write(out/(name+'.all.subgroups.json'),retained_subgroups(ps,valid,y,features,labels,policy))
        report['arms'][name] = dict(training,TRAIN_proxy=policy,reload_old_new64_bitexact=True,
            offline_training_and_scoring_seconds=time.monotonic()-tick,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            artifact_sha256=service.sha(out/(name+'.pt')))
        all_ps[name],policies[name] = ps,policy
        write(out/'report.json',report);del model,restored
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


if __name__=='__main__':
    main()

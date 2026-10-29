"""Matched old/new real-error data, with original normal/third/deletion draws fixed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
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
from train_sibilant_hubert_adaptation import novel_proxy,retained_subgroups
from train_sibilant_extra_normals import common_scores,proposed
from train_sibilant_precision_tail import summary_scores
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_precision_tail import PrecisionTailHead,tail_artifact,load_tail_head
from mdd.sibilant_error_exposure import error_plans


def write(path,value):
    Path(path).write_text(json.dumps(service.jsonable(value),ensure_ascii=False,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a = ap.parse_args();p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for name,digest in p['sha256'].items():
        if service.sha(name) != digest:raise ValueError('frozen error trial changed: '+name)
    new_path = Path(p['new_summary']);old,new = np.load(p['old_summary']),np.load(new_path/'summary.npz')
    offset = len(old['summary']);old_y = np.load(p['old_targets'])['y'];y = np.concatenate([old_y,new['y']])
    data = {k:np.concatenate([old[k],new[k]]) for k in ('summary','moments','available')}
    old_features = service.read_jsonl(p['old_features']);old_labels = service.read_jsonl(p['old_labels'])
    features = old_features+[dict(f,feature_index=f['feature_index']+offset)
        for f in service.read_jsonl(new_path/'data.features.jsonl')]
    labels = old_labels+[dict(z,feature_index=None if z['feature_index'] is None else z['feature_index']+offset)
        for z in service.read_jsonl(new_path/'data.labels.jsonl')]
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    old_indexed = {z['feature_index']:z for z in old_labels if z['feature_index'] is not None}
    if (len(indexed) != len(features) or len(features) != len(y)
            or len(indexed) != sum(z['feature_index'] is not None for z in labels)):
        raise ValueError('one source label/unknown per runtime input required')
    if offset != 13741 or len(old_labels) != 14002:
        raise ValueError('original common TRAIN population changed')
    original = np.load(p['old_plan']);eligibility = service.read_jsonl(p['old_ctc_targets'])
    if original.shape != (300,16) or p['steps'] != 300:
        raise ValueError('fixed original300x16 plan required')
    plans,plan_report = error_plans(original,y,indexed,features,eligibility,offset,p['seed'])
    new_used = set(int(i) for i in plans[1].ravel() if i>=offset)
    plan_report['new_used_recordings'] = len({indexed[i]['uid'] for i in new_used})
    plan_report['new_used_speakers'] = len({indexed[i]['speaker'] for i in new_used})
    plan_report['new_used_items'] = len({indexed[i]['item'] for i in new_used})
    directions = plan_report['new_unique_direction_counts']
    feasible = (all(directions.get(d,0)>=20 for d in ('ㅅ→ㅆ','ㅆ→ㅅ'))
        and plan_report['new_used_items']>=4 and plan_report['new_used_speakers']>=20)
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    for name,plan in zip(('old_errors','new_errors'),plans):np.save(out/(name+'.plan.npy'),plan)
    report = {'scope':p['scope'],'plan':plan_report,'data_feasible':bool(feasible),'arms':{},
        'original_inputs':offset,'original_labels':len(old_labels),'new_inputs':len(y)-offset,
        'all_retained_labels':len(labels),'new_source_pair_error_denominator':728,
        'deployment':False,'goal_complete':False,'service_improvement_proven':False}
    if not feasible:
        report['proceed_CAL'] = False;write(out/'report.json',report)
        write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')});return
    torch.set_num_threads(2);device = torch.device(p.get('device','cuda'))
    base = torch.load(p['control_artifact'],map_location='cpu')
    model = PrecisionTailHead(load_probe(base,p['binding'],device),'ce_full').to(device).eval()
    parent_ps,av = common_scores(model,data,offset,device)
    np.testing.assert_array_equal(parent_ps[:offset],np.load(p['old_scores'])['probabilities'])
    np.testing.assert_array_equal(parent_ps[offset:],new['probabilities'])
    np.testing.assert_array_equal(av,data['available'])
    parent_proxy = novel_proxy(parent_ps[:offset],av[:offset],old_y,old_features,old_indexed)
    if (parent_proxy['novel_TP'],parent_proxy['normal_FP'],parent_proxy['wrong_actual']) != (564,87,3):
        raise ValueError('original parent diagnostic differs')
    report['parent_original_TRAIN'] = parent_proxy
    all_ps = {'parent':parent_ps};policies = {'parent':parent_proxy}
    initial = tensor_digest(model.head.state_dict())
    for name,plan in zip(('old_errors','new_errors'),plans):
        torch.manual_seed(p['seed'])
        model = PrecisionTailHead(load_probe(base,p['binding'],device),'ce_full').to(device).eval()
        if tensor_digest(model.head.state_dict()) != initial:raise ValueError('starting head differs')
        parameters = [v for v in model.parameters() if v.requires_grad]
        if sum(v.numel() for v in parameters) != 50117:raise ValueError('matched readout capacity differs')
        optimizer = torch.optim.AdamW(parameters,lr=p['head_lr'],weight_decay=.01)
        losses,tick = [],time.monotonic()
        for step,requested in enumerate(plan,1):
            ix = requested[requested>=0]
            if not len(ix) or (y[ix]<0).any():raise ValueError('empty/unknown class in training')
            optimizer.zero_grad(set_to_none=True)
            logits = model.from_summary(torch.as_tensor(data['summary'][ix],device=device),
                                       torch.as_tensor(data['moments'][ix],device=device))
            loss = torch.nn.functional.cross_entropy(logits,torch.as_tensor(y[ix],device=device))
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite actual-phone loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            losses.append(float(loss.detach()))
            if step%100 == 0:print(json.dumps({'arm':name,'actual_updates':step,'loss':losses[-1]}),flush=True)
        if tensor_digest(model.head.state_dict()) == initial:raise ValueError('no actual weight update')
        artifact = tail_artifact(model,base)
        artifact['training_provenance'] = {'protocol_sha256':service.sha(a.protocol),
            'plan_sha256':service.sha(out/(name+'.plan.npy')),'actual_updates':300,'data_arm':name}
        torch.save(artifact,out/(name+'.pt'))
        ps,current_av = common_scores(model,data,offset,device);np.testing.assert_array_equal(current_av,av)
        restored = load_tail_head(torch.load(out/(name+'.pt'),map_location='cpu'),p['binding'],'ce_full',device)
        for start in (0,offset):
            check,valid = summary_scores(restored,{k:v[start:start+64] for k,v in data.items()},device)
            np.testing.assert_array_equal(check,ps[start:start+64]);np.testing.assert_array_equal(valid,av[start:start+64])
        np.savez(out/(name+'.scores.npz'),probabilities=ps,available=av)
        proxy = novel_proxy(ps[:offset],av[:offset],old_y,old_features,old_indexed)
        write(out/(name+'.original.subgroups.json'),retained_subgroups(ps[:offset],av[:offset],old_y,old_features,old_labels,proxy))
        # Same original threshold on all additional labels; do not retune on the
        # new data or call an additional TRAIN subset independent validation.
        write(out/(name+'.all.subgroups.json'),retained_subgroups(ps,av,y,features,labels,proxy))
        counts = collections.Counter(indexed[int(i)]['label_kind'] for i in plan.ravel() if i>=0)
        report['arms'][name] = {'actual_optimizer_updates':300,'trainable_parameters':50117,
            'initial_head_sha256':initial,'starting_parent_same':True,'base_and_roles_unchanged':True,
            'reload_old_and_new64_exact':True,'class_loss_first_last50':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],
            'draw_kinds':dict(counts),'known_draws':int((plan>=0).sum()),'unique_input_features':len(set(plan[plan>=0].tolist())),
            'original_TRAIN_proxy':proxy,'seconds_fit_and_score':time.monotonic()-tick,
            'artifact_sha256':service.sha(out/(name+'.pt')),'scores_sha256':service.sha(out/(name+'.scores.npz'))}
        all_ps[name],policies[name] = ps,proxy
        write(out/'report.json',report);print(name,json.dumps(report['arms'][name],ensure_ascii=False),flush=True)
    candidate = policies['new_errors'];control = policies['old_errors']
    report['proceed_CAL'] = bool(candidate['novel_TP']>max(564,control['novel_TP']) and candidate['wrong_actual']<=3
        and candidate['S_to_SS']['novel_TP']>=240 and candidate['SS_to_S']['novel_TP']>=324)
    changes = {name:collections.Counter() for name in ('old_errors','new_errors')}
    with (out/'paired_proposal_ledger.jsonl').open('x') as stream:
        for label in labels:
            calls = {name:proposed(label,ps,av,features,policies[name]) for name,ps in all_ps.items()}
            for name,counter in changes.items():
                prefix = 'normal' if label['label_kind']=='correct' else 'error_or_unknown'
                counter[prefix+'_added_calls'] += calls[name] and not calls['parent']
                counter[prefix+'_lost_calls'] += calls['parent'] and not calls[name]
            service.write_line(stream,dict(label,proposals=calls))
    report['all_retained_TRAIN_call_changes'] = {k:dict(v) for k,v in changes.items()}
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({'finished':True,'proceed_CAL':report['proceed_CAL'],'plan':plan_report}),flush=True)


if __name__ == '__main__':main()

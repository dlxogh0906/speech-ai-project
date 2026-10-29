"""Compare preregistered P3 arms without retuning thresholds or changing labels."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
              str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import json
from train_h0_intermediate import bind
from train_h0_backbone_control import file_sha
from eval_h0_backbone_control import compare_trajectory
from audit_h0_all_phones import tsv
from mdd.h0_phone_audit import events, paired_cluster_ci, rate


def ledger(path):
    return {(s['split'],s['uid'],s['q']):s for s in map(json.loads,path.open()) if s['role']=='selection'}


def paired(before,after,tau0,tau1):
    if before.keys()!=after.keys(): raise ValueError('selection sites differ')
    grouped=collections.defaultdict(collections.Counter)
    preservation=collections.defaultdict(collections.Counter)
    for k,a in before.items():
        b=after[k]
        assert all(a[f]==b[f] for f in ('actual','expected','speaker','item','position'))
        ea,eb=events(a,tau0),events(b,tau1)
        for axis in ('speaker','item'):
            c=grouped[(a['split'],axis,a[axis])]
            for f in ('normal','errors'): c[f]+=ea[f]
            for f in ('normal_fp','error_exact','policy_fp','policy_exact','error_wrong','policy_wrong'):
                c[f]+=eb[f]-ea[f]
        c=preservation[a['split']]
        c['baseline_policy_exact']+=ea['policy_exact']
        c['retained_policy_exact']+=ea['policy_exact'] and eb['policy_exact']
        c['new_policy_exact']+=not ea['policy_exact'] and eb['policy_exact']
        c['lost_policy_exact']+=ea['policy_exact'] and not eb['policy_exact']
        c['normal_false_alarms_added']+=not ea['policy_fp'] and eb['policy_fp']
        c['normal_false_alarms_removed']+=ea['policy_fp'] and not eb['policy_fp']
    ci={}
    for split in ('valid','valid_unseen'):
        ci[split]={}
        for axis in ('speaker','item'):
            values=[v for (sp,ax,key),v in grouped.items() if sp==split and ax==axis]
            ci[split][axis]={f:paired_cluster_ci(values,f,den) for f,den in
                [('normal_fp','normal'),('error_exact','errors'),('policy_fp','normal'),
                 ('policy_exact','errors'),('policy_wrong','errors')]}
    return dict(paired_ci=ci,preservation=dict(preservation))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True); ap.add_argument('--experiments',required=True)
    a=ap.parse_args(); p=bind(a.protocol); root=Path(a.experiments)
    out=root/'comparison'; out.mkdir(exist_ok=False)
    reports={}; logs={}; rows=[]; exposures=[]
    for arm in ('B0','B1','B2','B3'):
        finished=json.loads((root/arm/'finished.json').read_text())
        assert not finished['smoke'] and finished['updates']==2000
        assert finished['protocol_sha256']==file_sha(a.protocol)
        exposures.append({k:finished['final_exposure'][k] for k in p['expected_final_exposure']})
        for step in (3000,4000):
            name='%s_eval_%d'%(arm,step); path=root/name
            r=json.loads((path/'report.json').read_text()); f=json.loads((path/'finished.json').read_text())
            assert r['protocol_sha256']==file_sha(a.protocol)
            for key,filename in [('report_sha256','report.json'),('sites_sha256','sites.jsonl'),
                                 ('sequences_sha256','sequences.jsonl')]:
                assert f[key]==file_sha(path/filename)
            reports[name]=r; logs[name]=ledger(path/'sites.jsonl')
            for split in ('valid','valid_unseen'):
                for policy in ('greedy','calibrated'):
                    rows.append(dict(condition=arm,step=step,split=split,policy=policy,
                                     **r['results'][split]['all'][policy]))
    assert all(x==p['expected_final_exposure'] for x in exposures)
    initial=json.loads((Path(p['historical_experiment'])/'w2vbert_eval_2000/report.json').read_text())
    assert initial['checkpoint_sha256']==p['bindings'][p['parent_checkpoint']]
    initial['protocol_sha256']=file_sha(a.protocol); initial['backbone']='B0_common_initial'
    trajectories=compare_trajectory([initial,reports['B0_eval_3000'],reports['B0_eval_4000']],
                                    [reports['%s_eval_%d'%(arm,step)]
                                     for arm in ('B1','B2','B3') for step in (3000,4000)])
    comparisons={}; b=reports['B0_eval_4000']; bledger=logs['B0_eval_4000']
    tau0=b['threshold'] if b['threshold'] is not None else float('-inf')
    for arm in ('B1','B2','B3'):
        r=reports[arm+'_eval_4000']; tau1=r['threshold'] if r['threshold'] is not None else float('-inf')
        comparison=paired(bledger,logs[arm+'_eval_4000'],tau0,tau1)
        regressions=[]
        for phone,v in r['phone_counts']['valid_unseen'].items():
            old=b['phone_counts']['valid_unseen'][phone]
            assert v['errors']==old['errors'] and v['normal']==old['normal']
            if v['errors']>=30 and v['policy_exact_recall']<old['policy_exact_recall']-.05:
                regressions.append(dict(phone=phone,errors=v['errors'],
                    baseline=old['policy_exact_recall'],candidate=v['policy_exact_recall']))
        gap0,gap1=b['sequences']['valid_unseen'],r['sequences']['valid_unseen']
        deletion0,deletion1=b['results']['valid_unseen']['deletions']['calibrated'],r['results']['valid_unseen']['deletions']['calibrated']
        deletion_ok=(deletion1['errors']<30 or deletion1['exact_recall']>=deletion0['exact_recall']-.05)
        insertion_ok=(gap1['true_insertion_gaps']<30 or gap1['insertion_exact_recall']>=gap0['insertion_exact_recall']-.05)
        traj=next(t for t in trajectories if t['backbone']==arm and t['step']==4000)
        checks=dict(seen_FPR_at_most_1pct=r['results']['valid']['all']['calibrated']['normal_fpr']<=.01,
            unseen_FPR_at_most_1pct=r['results']['valid_unseen']['all']['calibrated']['normal_fpr']<=.01,
            unseen_exact_recall_improved=r['results']['valid_unseen']['all']['calibrated']['exact_recall']>
                b['results']['valid_unseen']['all']['calibrated']['exact_recall'],
            matched_seen_trajectory_improved=all(t['candidate_minus_baseline']<=0 for t in traj['brackets'])
                if traj['brackets'] else None,
            phone_guardrail_pass=not regressions,deletion_guardrail_pass=deletion_ok,
            insertion_guardrail_pass=insertion_ok,
            false_insertion_guardrail_pass=gap1['false_insertion_gap_rate']<=gap0['false_insertion_gap_rate']+.002)
        comparison.update(checks=checks,phone_regressions=regressions,trajectory=traj,
            development_gate_pass=all(v is True for v in checks.values()),
            service_adoption=False,independent_confirmation=False,suri=r['suri'])
        comparisons[arm]=comparison
    checkpoint_changes={}
    for arm in ('B0','B1','B2','B3'):
        a3,a4=reports[arm+'_eval_3000'],reports[arm+'_eval_4000']
        checkpoint_changes[arm]=paired(logs[arm+'_eval_3000'],logs[arm+'_eval_4000'],
            a3['threshold'] if a3['threshold'] is not None else float('-inf'),
            a4['threshold'] if a4['threshold'] is not None else float('-inf'))
    tsv(out/'summary.tsv',rows)
    result=dict(status='P3_development_comparison_completed',protocol_sha256=file_sha(a.protocol),
        conditions=8,actual_optimizer_updates=8000,common_exposure_verified=True,
        primary='B3 4000 versus B0 4000',comparisons=comparisons,trajectories=trajectories,
        checkpoint_changes=checkpoint_changes,service_model_changed=False,
        missing_acceptance_evidence=['independent new sentences/speakers','full production API/VAD/G2P/display validation'],
        inference_limits=['single seed; repeated development prompts; multiple descriptive ablations',
                          'no phonetic relabeling or independent listening performed'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (root/'completion.json').write_text(json.dumps(dict(status=result['status'],
        report=str(out/'report.json'),report_sha256=file_sha(out/'report.json'),
        development_gate_pass={k:v['development_gate_pass'] for k,v in comparisons.items()},
        service_model_changed=False),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],development_gate_pass=
                         {k:v['development_gate_pass'] for k,v in comparisons.items()})),flush=True)


if __name__=='__main__': main()

"""Report certified frontier samples, feasibility witnesses and remaining scope."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_h0_p5_readiness import write
from prepare_h0_p5_balance_design import tsv


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();root=a.root;out=root/'summary';out.mkdir(exist_ok=False)
    r=json.loads((root/'matrices/report.json').read_text());rows=[];reports={}
    selected=json.loads((root/'solution_selection.json').read_text())
    verified=json.loads((root/'verification/report.json').read_text())
    assert verified['status']=='raw_TRAIN_recount_passed'
    independently_counted={x['name']:x for x in verified['summaries']}
    for selection in selected:
        p=Path(selection['report'])
        d=json.loads(p.read_text());proto=json.loads((p.parent/'protocol.json').read_text());name=p.parent.name;reports[name]=d;c=selection['canonical_certificate']
        row=dict(condition=name,lambda_value=proto['lambda_value'],cap=proto['cap'],legacy=proto['legacy_objective'],
            context=d['context_objective'],context_reduction=1-d['normalized_context'],
            normal_HHI=independently_counted[name]['prompt'][0]['HHI'],error_HHI=independently_counted[name]['prompt'][1]['HHI'],
            normal_excess_HHI_reduction=independently_counted[name]['prompt'][0]['reduction'],error_excess_HHI_reduction=independently_counted[name]['prompt'][1]['reduction'],
            phone_min=d['ratios']['phone']['min'],phone_max=d['ratios']['phone']['max'],
            speaker_min=d['ratios']['speaker']['min'],speaker_max=d['ratios']['speaker']['max'],L1_share_change=d['max_L1_share_change'],
            certified=c['certified_to_tolerance'],constraint_violation=c['max_constraint_violation'],
            objective=c['objective'],lower_bound=c['lower_bound'],objective_gap=c['upper_minus_lower'],
            feasible_witness=d['manipulation_gate_passed'] and c['numerically_feasible'],iterations=c['iterations'])
        # A lower bound above this upper bound on the target set is a separating
        # certificate for that defined feasible set (not for other losses/data).
        row['target_set_objective_ceiling']=None if proto['legacy_objective'] else .75+.5*proto['lambda_value']
        row['separates_target_set']=not proto['legacy_objective'] and c['lower_bound']>.75+.5*proto['lambda_value']+1e-4
        rows.append(row)
    assert len(rows)==14
    tsv(out/'frontier.tsv',rows)
    frontier=sorted([x for x in rows if x['condition'].startswith('lambda_')],key=lambda x:x['lambda_value'])
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    prompt=[1-(x['normal_excess_HHI_reduction']+x['error_excess_HHI_reduction'])/2 for x in frontier]
    axes[0].plot(prompt,[x['context']/r['base_context'] for x in frontier],'-o',markersize=4,label='Certified scalarization samples')
    axes[0].scatter([1],[1],marker='x',s=60,color='black',label='C0')
    axes[0].axhline(.75,color='gray',ls='--',lw=1)
    axes[0].axvline(.5,color='gray',ls='--',lw=1)
    for x,px in zip(frontier,prompt):
        if x['lambda_value'] in (0,.3,1,10,100):
            axes[0].annotate(str(x['lambda_value']),(px,x['context']/r['base_context']),fontsize=8,xytext=(4,5),textcoords='offset points')
    axes[0].set_xlabel('Mean normalized excess prompt HHI (C0 = 1)')
    axes[0].set_ylabel('Normalized context deviation (C0 = 1)')
    axes[0].set_title('Two objectives; lower is better')
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for ax,source,key,idx in zip(axes[1:],('normal-source prompts','error-source prompts'),('normal_HHI','error_HHI'),(0,1)):
        ax.plot([x[key] for x in frontier],[x['context'] for x in frontier],'-o',label='Source projection of the same samples',markersize=4)
        size=sum(k[0]==r['dimensions']['source'][idx] for k in r['prompt_keys'])
        base_hhi=1/size+r['prompt_excess'][idx]
        ax.scatter([base_hhi],[r['base_context']],marker='x',s=60,color='black',label='C0')
        ax.axhline(.75*r['base_context'],color='gray',ls='--',lw=1)
        ax.axvline(1/size+.5*r['prompt_excess'][idx],color='gray',ls='--',lw=1)
        for x in frontier:
            if x['lambda_value'] in (0,.3,1,10,100):ax.annotate(str(x['lambda_value']),(x[key],x['context']),fontsize=8,xytext=(4,5),textcoords='offset points')
        ax.set_xlabel(source+' HHI');ax.grid(alpha=.2)
        ax.set_title('Individual source projection')
        ax.set_ylabel('Context deviation')
    fig.suptitle('P6 TRAIN contribution feasibility — no speech-model training')
    fig.tight_layout();fig.savefig(out/'pareto_frontier.svg');fig.savefig(out/'pareto_frontier.png',dpi=160);plt.close(fig)
    witness=[x['condition'] for x in rows if x['feasible_witness']]
    # These intersections were not hard constraints; expose what marginals hide.
    joint={}
    limits={'phone_source':(.8,1.2),'speaker_source':(.5,2),'l1_source':(.5,2)}
    with (root/'verification/group_contributions.tsv').open() as f:
        for x in csv.DictReader(f,delimiter='\t'):
            dim=x['dimension']
            if dim not in limits:continue
            key=(x['condition'],dim)
            z=joint.setdefault(key,dict(groups=0,outside=0,base_mass=0.,outside_base_mass=0.))
            mass=float(x['base']);ratio=float(x['ratio']);lo,hi=limits[dim]
            z['groups']+=1;z['base_mass']+=mass
            if ratio<lo-1e-6 or ratio>hi+1e-6:
                z['outside']+=1;z['outside_base_mass']+=mass
    joint_rows=[dict(condition=k[0],dimension=k[1],outside_base_share=v['outside_base_mass']/v['base_mass'],**v) for k,v in sorted(joint.items())]
    tsv(out/'joint_support.tsv',joint_rows)
    write(out/'report.json',dict(status='TRAIN_feasibility_and_holdout_design_completed',conditions=rows,
        feasible_witnesses=witness,joint_support=joint_rows,certified_conditions=sum(x['certified'] for x in rows),
        frontier_scope='finite scalarization grid, with numerical global objective bounds; not every point of a continuous frontier',
        feasibility_scope='fixed P5 TRAIN stream and contribution proxies; not independent phone gradients or evidence of MDD performance',
        model_optimizer_updates=0,model_inference_count=0,training_allowed=False,
        next_gate='rebuild each proposed fold lineage; recompute treatment and support on each fold TRAIN only; predeclare methods, metrics, CAL and budget before any model fit',
        independent_final_evaluation=False))
    print(json.dumps(dict(feasible_witnesses=witness,certified_conditions=sum(x['certified'] for x in rows)),ensure_ascii=False))

if __name__=='__main__':main()

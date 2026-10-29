"""Choose numerical representatives of the same problems by certificate only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,json
from train_h0_backbone_control import file_sha
from audit_h0_p5_readiness import write
from solve_h0_p6_frontier import load_problem,constraints,objectives
from mdd.h0_p6_feasibility import certificate
import numpy as np


def select(root,require_all=True):
    r,z,m=load_problem(root/'matrices')
    names=['lambda_%g'%v for v in (0,.03,.1,.3,1,3,10,30,100)]+['cap_%g'%v for v in (2,1.5,1.2,1.1)]+['legacy_convergence']
    result=[]
    for name in names:
        paths=[root/family/name/'report.json' for family in ('frontier','refined','checkpointed','interior')]
        candidates=[]
        for path in paths:
            if not path.exists():continue
            data=json.loads(path.read_text());proto=json.loads((path.parent/'protocol.json').read_text())
            sol=np.load(path.parent/'solution.npz');w=sol['w'];y=sol['y'].copy()
            if proto['cap'] is not None and not proto.get('scaled_caps',False):
                num=np.bincount(z['prompt_source']);y[-len(z['prompt_source']):]/=num[z['prompt_source']]
            A,l,u=constraints(r,z,m,not proto['legacy_objective'],proto['cap'],True)
            Q,t=objectives(r,z,m,proto['lambda_value'],proto['legacy_objective'])
            c=certificate(Q,t,w,A,l,u,y)
            c.update({k:data['certificate'][k] for k in ('solver_status','iterations','primal_residual','dual_residual','solver_run_time','osqp_version')})
            if 'clarabel_version' in data['certificate']:c['clarabel_version']=data['certificate']['clarabel_version']
            candidates.append((path,c))
        if not candidates:
            if require_all:raise RuntimeError('missing result: '+name)
            continue
        best=min(candidates,key=lambda pc:(not pc[1]['certified_to_tolerance'],not pc[1]['numerically_feasible'],abs(pc[1]['upper_minus_lower'])))
        result.append(dict(condition=name,report=str(best[0]),report_sha256=file_sha(best[0]),
            canonical_certificate=best[1],
            selection_rule='certificate pass, then numerical feasibility, then absolute objective-bound gap; same mathematical problem',
            attempts=[dict(report=str(path),certificate=c) for path,c in candidates]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    selections=select(a.root);assert all(x['canonical_certificate']['certified_to_tolerance'] for x in selections)
    write(a.root/'solution_selection.json',selections);print('Certified representatives:',len(selections))

if __name__=='__main__':main()

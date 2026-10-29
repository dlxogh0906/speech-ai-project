"""Remove numerical feasibility residuals by source normalization and mixing.

Only uncapped hard-support problems are supported: C0 is then an interior
anchor for every inequality. This is a deterministic feasibility construction,
not another weighting objective or a speech-model training experiment.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,json
import numpy as np
from scipy import sparse
from solve_h0_p6_frontier import load_problem,constraints,objectives,metrics
from mdd.h0_p6_feasibility import certificate
from audit_h0_p5_readiness import write
from train_h0_backbone_control import file_sha
from prepare_h0_p5_balance_design import tsv


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--matrices',type=Path,required=True)
    ap.add_argument('--solution',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(exist_ok=False);r,z,m=load_problem(a.matrices)
    proto=json.loads((a.solution/'protocol.json').read_text());assert not proto['legacy_objective'] and proto['cap'] is None
    saved=np.load(a.solution/'solution.npz');raw=saved['w'];y=saved['y'];source=np.asarray(m['source'].argmax(axis=0)).ravel()
    w=raw/(m['source'].dot(raw))[source];anchor=np.ones(len(w))
    A,l,u=constraints(r,z,m,True,None,True);Q,t=objectives(r,z,m,proto['lambda_value'],False)
    B=sparse.vstack([A[m['source'].shape[0]:],sparse.eye(len(w))],format='csr')
    low=np.r_[l[m['source'].shape[0]:],np.full(len(w),.25)];high=np.r_[u[m['source'].shape[0]:],np.full(len(w),4.)]
    base=B.dot(anchor);point=B.dot(w);delta=point-base;alpha=1.
    assert np.all(base>low) and np.all(base<high)
    over=point>high;under=point<low
    if over.any():alpha=min(alpha,float(np.min((high[over]-base[over])/delta[over])))
    if under.any():alpha=min(alpha,float(np.min((low[under]-base[under])/delta[under])))
    alpha*=1-1e-9;w=anchor+alpha*(w-anchor)
    c=certificate(Q,t,w,A,l,u,y);assert c['max_constraint_violation']<1e-10 and c['certified_to_tolerance']
    result,groups=metrics(r,z,m,w);result['certificate']=c
    assert result['manipulation_gate_passed']
    result['numerical_construction']=dict(source_normalized=True,anchor='C0',mixture_fraction=alpha,
        max_weight_change=float(np.max(abs(w-raw))),parent=str(a.solution),parent_solution_sha256=file_sha(a.solution/'solution.npz'))
    np.save(a.out/'weights.npy',w);np.savez(a.out/'solution.npz',w=w,y=y);write(a.out/'protocol.json',proto)
    write(a.out/'report.json',result);tsv(a.out/'support.tsv',groups)
    write(a.out/'finished.json',dict(status='feasible_construction_verified',model_optimizer_updates=0))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

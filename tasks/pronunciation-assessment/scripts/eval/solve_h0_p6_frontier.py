"""Sparse convex frontier and convergence check; TRAIN only, never model fit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse,json,time
import numpy as np
from scipy import sparse
from mdd.h0_p6_feasibility import solve_qp,solve_interior
from audit_h0_p5_readiness import write
from prepare_h0_p5_balance_design import tsv


def load_problem(folder):
    r=json.loads((folder/'report.json').read_text());a=dict(np.load(folder/'arrays.npz'))
    m={name:sparse.load_npz(folder/(name+'.npz')) for name in
        ('Qc','Qp','prompt','phone','speaker','l1','source','context','tokens')}
    return r,a,m


def constraints(r,a,m,hard_support=True,cap=None,scaled_caps=False):
    matrices=[m['source']];lower=[np.ones(m['source'].shape[0])];upper=[lower[0].copy()]
    if hard_support:
        for dim,lo,hi in [('phone',.8,1.2),('speaker',.5,2.)]:
            matrices.append(m[dim]);lower.append(np.full(m[dim].shape[0],lo));upper.append(np.full(m[dim].shape[0],hi))
        mb=np.array(r['baselines']['l1']);delta=.01/(mb/mb.sum())
        matrices.append(m['l1']);lower.append(np.maximum(0,1-delta));upper.append(1+delta)
    if cap is not None:
        source=a['prompt_source'];num=np.bincount(source)
        factor=num[source] if scaled_caps else np.ones(len(source))
        matrices.append(sparse.diags(factor).dot(m['prompt']));lower.append(np.full(len(source),-np.inf));upper.append(factor*cap/num[source])
    return sparse.vstack(matrices,format='csr'),np.concatenate(lower),np.concatenate(upper)


def objectives(r,a,m,lam=1.,legacy=False):
    if legacy:
        qs=[m['Qc']];ts=[a['tc']]
        for dim,scale in [('phone',20),('l1',10),('speaker',1)]:
            c=np.sqrt(scale/m[dim].shape[0]);qs.append(c*m[dim]);ts.append(np.full(m[dim].shape[0],c))
    else:
        qs=[m['Qc']/np.sqrt(r['base_context'])];ts=[a['tc']/np.sqrt(r['base_context'])]
        if lam>0:qs.append(np.sqrt(lam)*m['Qp']);ts.append(np.sqrt(lam)*a['tp'])
    return sparse.vstack(qs,format='csr'),np.concatenate(ts)


def metrics(r,a,m,w):
    context=float(sum((m['Qc'].dot(w)-a['tc'])**2));mass=m['prompt'].dot(w);prompt=[]
    for s,source in enumerate(r['dimensions']['source']):
        x=mass[a['prompt_source']==s];hhi=float(x.dot(x));ideal=1/len(x)
        prompt.append(dict(source=source,HHI=hhi,max_to_mean=float(x.max()*len(x)),
            normalized_excess_HHI=(hhi-ideal)/r['prompt_excess'][s],
            reduction=1-(hhi-ideal)/r['prompt_excess'][s]))
    ratios={};groups=[]
    for dim in ('phone','speaker','l1'):
        z=m[dim].dot(w);base=np.array(r['baselines'][dim]);ratios[dim]=dict(min=float(z.min()),max=float(z.max()))
        for k,b,v in zip(r['dimensions'][dim],base,z):groups.append(dict(dimension=dim,key=k,baseline=b,weighted=b*v,ratio=v))
    mb=np.array(r['baselines']['l1']);mw=mb*m['l1'].dot(w);l1delta=float(max(abs(mw/mw.sum()-mb/mb.sum())))
    gate=dict(prompt=all(x['reduction']>=.5-2e-6 for x in prompt),context=context<=.75*r['base_context']+2e-6,
        phone=ratios['phone']['min']>=.8-2e-6 and ratios['phone']['max']<=1.2+2e-6,
        speaker=ratios['speaker']['min']>=.5-2e-6 and ratios['speaker']['max']<=2+2e-6,l1=l1delta<=.01+2e-6)
    return dict(context_objective=context,normalized_context=context/r['base_context'],prompt=prompt,
        mean_normalized_prompt_excess=float(np.mean([x['normalized_excess_HHI'] for x in prompt])),
        ratios=ratios,max_L1_share_change=l1delta,weight_min=float(w.min()),weight_max=float(w.max()),
        gates=gate,manipulation_gate_passed=all(gate.values()),training_allowed=False),groups


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--matrices',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--lambda-value',type=float,default=1.)
    ap.add_argument('--cap',type=float);ap.add_argument('--legacy',action='store_true')
    ap.add_argument('--initial',type=Path);ap.add_argument('--max-iter',type=int,default=100000)
    ap.add_argument('--rho',type=float,default=.1);ap.add_argument('--adaptive-interval',type=int,default=0)
    ap.add_argument('--scaled-caps',action='store_true')
    ap.add_argument('--checkpoint-every',type=int,default=0)
    ap.add_argument('--solver',choices=['osqp','clarabel'],default='osqp')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);r,z,m=load_problem(a.matrices)
    A,l,u=constraints(r,z,m,not a.legacy,a.cap,a.scaled_caps);Q,t=objectives(r,z,m,a.lambda_value,a.legacy)
    initial=np.load(a.initial) if a.initial else np.ones(Q.shape[1])
    write(a.out/'protocol.json',dict(matrices=str(a.matrices),lambda_value=a.lambda_value,cap=a.cap,
        legacy_objective=a.legacy,hard_support=not a.legacy,initial=str(a.initial),max_iter=a.max_iter,
        model_optimizer_updates=0,created=time.time(),rho=a.rho,adaptive_interval=a.adaptive_interval,scaled_caps=a.scaled_caps,
        checkpoint_every=a.checkpoint_every,solver=a.solver))
    def checkpoint(iteration,w,y,cert):
        np.savez(a.out/'iterate.npz',w=w,y=y)
        with (a.out/'convergence.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=iteration,**cert))+'\n')
        print('certificate',iteration,json.dumps(cert),flush=True)
    if a.solver=='clarabel':w,y,cert=solve_interior(Q,t,A,l,u)
    else:w,y,cert=solve_qp(Q,t,A,l,u,initial,a.max_iter,a.rho,a.adaptive_interval,a.checkpoint_every,checkpoint if a.checkpoint_every else None)
    np.savez(a.out/'solution.npz',w=w,y=y);np.save(a.out/'weights.npy',w)
    result,groups=metrics(r,z,m,w);result['certificate']=cert
    tsv(a.out/'support.tsv',groups);write(a.out/'report.json',result)
    write(a.out/'finished.json',dict(status='numerical_solve_finished',certified=cert['certified_to_tolerance'],model_optimizer_updates=0))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

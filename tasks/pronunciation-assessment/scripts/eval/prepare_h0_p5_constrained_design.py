"""Bounded TRAIN-only prompt and context manipulation; no speech-model updates."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,collections,hashlib,json,time
import numpy as np
from scipy import sparse
from train_h0_backbone_control import file_sha,stream
from train_w2vbert_ctc import read_split
from prepare_h0_p5_balance_design import contexts,tsv
from audit_h0_p5_readiness import write

LO,HI=.25,4.


def project(v,coeff,groups,target):
    """Weighted Euclidean projection onto box and exact prompt mass constraints."""
    size=len(target);left=np.full(size,float(v.min()-HI));right=np.full(size,float(v.max()-LO))
    for _ in range(45):
        mid=(left+right)/2
        mass=np.bincount(groups,weights=coeff*np.clip(v-mid[groups],LO,HI),minlength=size)
        low=mass>target;left[low]=mid[low];right[~low]=mid[~low]
    return np.clip(v-((left+right)/2)[groups],LO,HI)


def matrix(keys,rows,cols,vals,n):
    return sparse.coo_matrix((vals,(rows,cols)),shape=(len(keys),n)).tocsr()


def project_caps(v,coeff,groups,caps,source_group,source_target):
    """Box, per-prompt upper caps, and exact source totals, weighted projection."""
    left=np.full(len(caps),float(v.min()-HI));right=np.full(len(caps),float(v.max()-LO))
    for _ in range(42):
        mid=(left+right)/2
        mass=np.bincount(groups,weights=coeff*np.clip(v-mid[groups],LO,HI),minlength=len(caps))
        more=mass>caps;left[more]=mid[more];right[~more]=mid[~more]
    cap_shift=(left+right)/2
    lo=np.full(len(source_target),float(v.min()-HI));hi=np.full(len(source_target),float(v.max()-LO))
    for _ in range(42):
        mid=(lo+hi)/2
        mass=np.bincount(groups,weights=coeff*np.clip(v-mid[source_group[groups]],LO,HI),minlength=len(caps))
        total=np.bincount(source_group,weights=np.minimum(mass,caps),minlength=len(source_target))
        more=total>source_target;lo[more]=mid[more];hi[~more]=mid[~more]
    shifts=np.maximum(cap_shift,((lo+hi)/2)[source_group])
    return np.clip(v-shifts[groups],LO,HI)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--p3-protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);p=json.loads(a.p3_protocol.read_text());manifest=Path(p['manifest'])
    for ext in ('tsv','ltr','source'):
        f=manifest/('train.'+ext);assert file_sha(f)==p['bindings'][str(f)]
    write(out/'protocol.json',dict(status='precomputed_design_before_model_training',created=time.time(),
        source=str(Path(__file__).resolve()),source_sha256=file_sha(__file__),parent_protocol_sha256=file_sha(a.p3_protocol),
        TRAIN_bindings={str(manifest/('train.'+s)):file_sha(manifest/('train.'+s)) for s in ('tsv','ltr','source')},
        bounds=[LO,HI],native_weight=1,physical_exposure='exact original B0 continuation; unchanged all arms',
        loss_coefficient='per occurrence 1/microbatch size, with CTC NLL/target length retained',
        C1='minimum prompt HHI within each source subject to bounds and preserved source total',
        C2='prompt cap 1.2 times source mean, preserved source totals; bounded convex quadratic context balancing with phone/L1/speaker penalties',
        previous_design='v1 fixed minimum-HHI prompt masses; v2 relaxes equality to universal upper caps, using TRAIN diagnostics only',
        context_proxy='sum utterance loss coefficient * number of context occurrences / target length; not gradient contribution',
        supported_context='at least 30 token draws, 5 unique manifest rows, 3 speakers and 2 items; no phone exceptions',
        context_target='square root of baseline proxy within phone-position, preserving supported group total',
        objective_weights=dict(context=1,phone=20,l1=10,speaker=1),max_iterations=300,
        gates=dict(prompt_excess_HHI_reduction_min=.5,context_objective_reduction_min=.25,
            phone_mass_ratio=[.8,1.2],speaker_mass_ratio=[.5,2.],max_L1_share_change=.01),
        gates_scope='TRAIN manipulation feasibility only; not performance or authorization to train',
        evaluation_inputs_read=False,model_optimizer_updates=0))
    rows,labels=read_split(str(manifest),'train');sources=(manifest/'train.source').read_text().splitlines()
    batches=stream(rows,p['batch_samples'],p['seed']);h=hashlib.sha256();count=collections.Counter();co=collections.Counter();draws=samples=0;receipts=[]
    for micro in range(p['stop_step']*p['accum']):
        ids=next(batches);h.update((' '.join(map(str,ids))+'\n').encode());draws+=len(ids);samples+=sum(rows[i][1] for i in ids)
        if micro>=p['start_step']*p['accum']:
            count.update(ids)
            for i in ids:co[i]+=1/len(ids)
        if micro+1 in (p['start_step']*p['accum'],p['stop_step']*p['accum']):
            r=dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest())
            assert r==(p['parent_exposure'] if not receipts else p['expected_final_exposure']);receipts.append(r)
    ix=np.array(sorted(i for i in count if sources[i].startswith('foreign')),dtype=int);n=len(ix)
    coeff=np.array([co[i] for i in ix]);draw=np.array([count[i] for i in ix]);meta=[];cs=[];support=collections.defaultdict(lambda:dict(draws=0,rows=0,speakers=set(),items=set()))
    for i in ix:
        parts=Path(rows[i][0]).stem.split('-');m=dict(source=sources[i],item=parts[5],speaker=parts[0],l1=parts[3]);meta.append(m)
        c=collections.Counter(contexts(labels[i]));cs.append(c)
        for key,num in c.items():
            z=support[key];z['draws']+=int(count[i]*num);z['rows']+=1;z['speakers'].add(m['speaker']);z['items'].add(m['item'])
    promptkeys=sorted({(m['source'],m['item']) for m in meta});lookup={k:j for j,k in enumerate(promptkeys)}
    group=np.array([lookup[m['source'],m['item']] for m in meta]);base=np.bincount(group,weights=coeff);target=base.copy()
    for src in sorted({m['source'] for m in meta}):
        g=np.array([j for j,k in enumerate(promptkeys) if k[0]==src]);x=base[g];low,high=0.,float(x.sum())
        for _ in range(80):
            mid=(low+high)/2
            if np.clip(mid,LO*x,HI*x).sum()<x.sum():low=mid
            else:high=mid
        target[g]=np.clip((low+high)/2,LO*x,HI*x)
    c1=target[group]/base[group];np.testing.assert_allclose(project(c1,coeff,group,target),c1,atol=1e-10)
    source_keys=sorted({k[0] for k in promptkeys});source_group=np.array([source_keys.index(k[0]) for k in promptkeys])
    source_target=np.bincount(source_group,weights=base)
    caps=np.array([1.2*base[source_group==s].mean() for s in source_group])
    assert np.all(LO*base<=caps)
    assert np.all(np.bincount(source_group,weights=np.minimum(HI*base,caps))>=source_target)
    # Every context is retained in the report; only sufficiently supported cells drive the objective.
    keys=sorted(support);kl={k:j for j,k in enumerate(keys)};rr=[];cc=[];vv=[];tv=[]
    for j,i in enumerate(ix):
        for k,num in cs[j].items():rr.append(kl[k]);cc.append(j);vv.append(coeff[j]*num/len(labels[i]));tv.append(draw[j]*num)
    A=matrix(keys,rr,cc,vv,n);tokens=matrix(keys,rr,cc,tv,n);ab=np.asarray(A.sum(1)).ravel()
    supported=np.array([support[k]['draws']>=30 and support[k]['rows']>=5 and len(support[k]['speakers'])>=3 and len(support[k]['items'])>=2 for k in keys])
    sk=np.flatnonzero(supported);sg=collections.defaultdict(list)
    for j in sk:sg[keys[j][:2]].append(j)
    total={g:sum(ab[j] for j in js) for g,js in sg.items()};tar=ab.copy()
    for g,js in sg.items():tar[js]=total[g]*np.sqrt(ab[js])/np.sqrt(ab[js]).sum()
    norm=np.array([total[keys[j][:2]] for j in sk]);B=sparse.diags(1/norm).dot(A[sk]);t=tar[sk]/norm
    # Average Pearson distance across phone-position groups; no rare zero-support cells invented.
    context_factor=1/(len(sg)*t)
    penalties=[];matrices={}
    for dim,weight in [('phone',20.),('l1',10.),('speaker',1.)]:
        dk=sorted({k[0] for k in keys}) if dim=='phone' else sorted({m[dim] for m in meta});dl={k:j for j,k in enumerate(dk)}
        if dim=='phone':
            merge=sparse.coo_matrix((np.ones(len(keys)),([dl[k[0]] for k in keys],np.arange(len(keys)))),shape=(len(dk),len(keys))).tocsr();M=merge.dot(A)
        else:M=sparse.coo_matrix((coeff,([dl[m[dim]] for m in meta],np.arange(n))),shape=(len(dk),n)).tocsr()
        mb=np.asarray(M.sum(1)).ravel();D=sparse.diags(1/mb).dot(M);penalties.append((D,weight/len(dk)));matrices[dim]=(dk,M,mb)
    def objective(w,grad=True):
        d=B.dot(w)-t;context=float(np.dot(context_factor,d*d));val=context;g=2*B.T.dot(context_factor*d) if grad else None
        for D,lam in penalties:
            e=D.dot(w)-1;val+=lam*float(np.dot(e,e))
            if grad:g+=2*lam*D.T.dot(e)
        return val,g,context
    w=project_caps(np.ones(n),coeff,group,caps,source_group,source_target);value,gradient,context=objective(w);step=10.;history=[]
    for it in range(300):
        for trial in range(30):
            candidate=project_caps(w-step*gradient/coeff,coeff,group,caps,source_group,source_target)
            new,_,ctx=objective(candidate,False);delta=candidate-w
            if new<=value+float(np.dot(gradient,delta))+.5/step*float(np.dot(coeff,delta*delta))+1e-12:break
            step*=.5
        change=float(np.max(np.abs(delta)));w=candidate;value,gradient,context=objective(w);step*=1.1
        history.append(dict(iteration=it+1,objective=value,context_objective=context,max_weight_change=change))
        if (it+1)%25==0:print(json.dumps(history[-1]),flush=True);write(out/'live_status.json',history[-1])
        if change<1e-7:break
    arms={'C0':np.ones(n),'C1':c1,'C2':w};summaries=[];contexts_out=[];group_out=[]
    c0context=objective(arms['C0'],False)[2]
    for arm,ww in arms.items():
        full=np.ones(len(rows));full[ix]=ww;np.save(out/(arm+'_weights.npy'),full)
        mass=np.bincount(group,weights=coeff*ww)
        if arm!='C2':np.testing.assert_allclose(mass,target if arm=='C1' else base,rtol=1e-9,atol=1e-9)
        else:
            assert np.all(mass<=caps+1e-8)
            np.testing.assert_allclose(np.bincount(source_group,weights=mass),source_target,rtol=1e-9,atol=1e-9)
        assert min(ww)>=LO-1e-10 and max(ww)<=HI+1e-10
        prompt=[]
        for src in sorted({m['source'] for m in meta}):
            js=[j for j,k in enumerate(promptkeys) if k[0]==src];x=mass[js];b=base[js];ideal=1/len(js);hhi=float(sum((x/x.sum())**2));bh=float(sum((b/b.sum())**2))
            prompt.append(dict(source=src,prompts=len(js),HHI=hhi,max_to_mean=float(x.max()/x.mean()),excess_HHI_reduction=(bh-hhi)/(bh-ideal)))
        ratios={};l1delta=0
        for dim,(dk,M,mb) in matrices.items():
            after=M.dot(ww);ratio=after/mb;ratios[dim]=dict(min=float(ratio.min()),max=float(ratio.max()))
            if dim=='l1':l1delta=float(max(abs(after/after.sum()-mb/mb.sum())))
            for k,b,z in zip(dk,mb,after):group_out.append(dict(arm=arm,dimension=dim,key=k,base=float(b),weighted=float(z)))
        after=A.dot(ww);token_after=tokens.dot(ww);ctx=objective(ww,False)[2]
        special={v:float(sum(token_after[j] for j,k in enumerate(keys) if (k[0],k[2],k[3])==(v,'ㅅ','ㄹ'))) for v in ('ㅜ','ㅗ')}
        gates=dict(prompt_concentration=all(z['excess_HHI_reduction']>=.5-1e-9 for z in prompt),context_manipulation=(c0context-ctx)/c0context>=.25,
            phone_support=ratios['phone']['min']>=.8 and ratios['phone']['max']<=1.2,
            speaker_support=ratios['speaker']['min']>=.5 and ratios['speaker']['max']<=2.,L1_support=l1delta<=.01)
        summaries.append(dict(arm=arm,weight_min=float(ww.min()),weight_max=float(ww.max()),at_lower=int(sum(ww<=LO+1e-8)),at_upper=int(sum(ww>=HI-1e-8)),
            prompt=prompt,context_objective=ctx,context_objective_reduction=(c0context-ctx)/c0context,mass_ratios=ratios,max_L1_share_change=l1delta,
            s_u_l_weighted_tokens=special['ㅜ'],s_o_l_weighted_tokens=special['ㅗ'],s_o_to_u_ratio=special['ㅗ']/special['ㅜ'],gates=gates,
            manipulation_gate_passed=all(gates.values()),training_allowed=False))
        for j,k in enumerate(keys):contexts_out.append(dict(arm=arm,phone=k[0],position=k[1],left=k[2],right=k[3],supported=bool(supported[j]),base_proxy=float(ab[j]),weighted_proxy=float(after[j]),weighted_token_draws=float(token_after[j])))
    tsv(out/'contexts.tsv',contexts_out);tsv(out/'group_contributions.tsv',group_out)
    tsv(out/'prompt_contributions.tsv',[dict(source=k[0],item=k[1],C0=float(base[j]),C1=float(target[j]),C2=float(np.bincount(group,weights=coeff*w)[j])) for j,k in enumerate(promptkeys)])
    write(out/'optimization_trace.json',history)
    write(out/'report.json',dict(status='bounded_TRAIN_design_measured_not_trained',receipts=receipts,continuation_rows=sum(count.values()),continuation_samples=sum(count[i]*rows[i][1] for i in count),
        active_foreign_rows=n,context_cells=len(keys),supported_context_cells=int(supported.sum()),phone_position_groups=len(sg),arms=summaries,
        native_weight=1,physical_exposure_unchanged=True,model_optimizer_updates=0,numerical_weight_design_iterations=len(history),
        limitations=['utterance-level CTC coefficients, not independent token gradients','no missing contexts manufactured','TRAIN manipulation gates only; no model effectiveness test','bounded design failure does not disprove imbalance hypothesis']))
    write(out/'finished.json',dict(status='completed_no_training',report_sha256=file_sha(out/'report.json'),protocol_sha256=file_sha(out/'protocol.json'),model_optimizer_updates=0))

if __name__=='__main__':main()

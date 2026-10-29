"""Freeze the exact P5 TRAIN problem, compressing only proportional columns."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,collections,gzip,hashlib,json,time
import numpy as np
from scipy import sparse
from train_h0_backbone_control import file_sha,stream
from train_w2vbert_ctc import read_split
from prepare_h0_p5_balance_design import contexts
from audit_h0_p5_readiness import write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p3-protocol',type=Path,required=True)
    ap.add_argument('--previous-design',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    p=json.loads(a.p3_protocol.read_text());manifest=Path(p['manifest'])
    bindings={str(a.p3_protocol):file_sha(a.p3_protocol)}
    for ext in ('tsv','ltr','source'):
        f=manifest/('train.'+ext);assert file_sha(f)==p['bindings'][str(f)]
        bindings[str(f)]=file_sha(f)
    write(out/'protocol.json',dict(created=time.time(),bindings=bindings,
        source_sha256=file_sha(__file__),evaluation_inputs_read=False,model_optimizer_updates=0,
        target='identical P5 v2 context target and supported cells',
        compression='same source/item/speaker/L1/actual target: coefficient-weighted average; exact feasible objective image under common box',
        grid=[0,.03,.1,.3,1,3,10,30,100],
        objective='context/C0_context + lambda * mean source normalized excess prompt HHI',
        constraints=dict(weight=[.25,4],source='exact',phone=[.8,1.2],speaker=[.5,2],L1_share_absolute_change=.01),
        additional_cap_sensitivity=[None,2.,1.5,1.2,1.1],
        certificate=dict(feasibility_tolerance=2e-6,gap_tolerance=2e-5),
        no_train_gate='P6 is feasibility only; model training remains paused'))
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
    ix=np.array(sorted(i for i in count if sources[i].startswith('foreign')),dtype=int)
    lookup={};groups=[];meta=[];cs=[];coeff=[];draw=[];support=collections.defaultdict(lambda:dict(draws=0,rows=0,speakers=set(),items=set()))
    for i in ix:
        parts=Path(rows[i][0]).stem.split('-');m=dict(source=sources[i],item=parts[5],speaker=parts[0],l1=parts[3])
        key=(m['source'],m['item'],m['speaker'],m['l1'],tuple(labels[i]))
        if key not in lookup:
            lookup[key]=len(meta);meta.append(m);cs.append(collections.Counter(contexts(labels[i])));coeff.append(0.);draw.append(0)
        j=lookup[key];groups.append(j);coeff[j]+=co[i];draw[j]+=count[i]
        for k,num in cs[j].items():
            z=support[k];z['draws']+=int(count[i]*num);z['rows']+=1;z['speakers'].add(m['speaker']);z['items'].add(m['item'])
    groups=np.array(groups);coeff=np.array(coeff);draw=np.array(draw);n=len(meta)
    keys=sorted(support);kl={k:j for j,k in enumerate(keys)};rr=[];cc=[];vv=[];tv=[]
    for j,c in enumerate(cs):
        length=sum(c.values())
        for k,num in c.items():rr.append(kl[k]);cc.append(j);vv.append(coeff[j]*num/length);tv.append(draw[j]*num)
    A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(keys),n)).tocsr()
    tokens=sparse.coo_matrix((tv,(rr,cc)),shape=A.shape).tocsr();ab=np.asarray(A.sum(1)).ravel()
    supported=np.array([support[k]['draws']>=30 and support[k]['rows']>=5 and len(support[k]['speakers'])>=3 and len(support[k]['items'])>=2 for k in keys])
    sk=np.flatnonzero(supported);sg=collections.defaultdict(list)
    for j in sk:sg[keys[j][:2]].append(j)
    totals={g:sum(ab[j] for j in js) for g,js in sg.items()};tar=ab.copy()
    for g,js in sg.items():tar[js]=totals[g]*np.sqrt(ab[js])/np.sqrt(ab[js]).sum()
    norm=np.array([totals[keys[j][:2]] for j in sk]);t=tar[sk]/norm
    Qc=sparse.diags(1/np.sqrt(len(sg)*t)/norm).dot(A[sk]).tocsr();tc=t/np.sqrt(len(sg)*t)
    base_context=float(np.sum((Qc.dot(np.ones(n))-tc)**2))
    pk=sorted({(m['source'],m['item']) for m in meta});pl={k:j for j,k in enumerate(pk)};pg=np.array([pl[m['source'],m['item']] for m in meta])
    Mp=sparse.coo_matrix((coeff,(pg,np.arange(n))),shape=(len(pk),n)).tocsr();bp=np.asarray(Mp.sum(1)).ravel()
    src=sorted({m['source'] for m in meta});sp=np.array([src.index(k[0]) for k in pk]);st=np.bincount(sp,weights=bp)
    P=sparse.diags(1/st[sp]).dot(Mp).tocsr();excess=[];prompt_scale=np.zeros(len(pk));pt=np.zeros(len(pk))
    for s in range(len(src)):
        z=sp==s;num=sum(z);e=float(sum((bp[z]/st[s])**2)-1/num);excess.append(e)
        prompt_scale[z]=1/np.sqrt(len(src)*e);pt[z]=1/num
    Qp=sparse.diags(prompt_scale).dot(P).tocsr();tp=prompt_scale*pt
    matrices={'context':A,'tokens':tokens,'Qc':Qc,'Qp':Qp,'prompt':P}
    dims={};baselines={}
    for dim in ('phone','speaker','l1','source'):
        dk=sorted({k[0] for k in keys}) if dim=='phone' else sorted({m[dim] for m in meta});dl={k:j for j,k in enumerate(dk)}
        if dim=='phone':
            merge=sparse.coo_matrix((np.ones(len(keys)),([dl[k[0]] for k in keys],np.arange(len(keys)))),shape=(len(dk),len(keys))).tocsr();M=merge.dot(A)
        else:M=sparse.coo_matrix((coeff,([dl[m[dim]] for m in meta],np.arange(n))),shape=(len(dk),n)).tocsr()
        mb=np.asarray(M.sum(1)).ravel();D=sparse.diags(1/mb).dot(M).tocsr()
        matrices[dim]=D;dims[dim]=dk;baselines[dim]=mb.tolist()
    for name,M in matrices.items():sparse.save_npz(out/(name+'.npz'),M)
    np.savez(out/'arrays.npz',ix=ix,row_to_variable=groups,coeff=coeff,original_coeff=np.array([co[i] for i in ix]),
        tc=tc,tp=tp,supported=supported,prompt_source=sp,prompt_base=bp,source_total=st,context_base=ab)
    with gzip.open(out/'variables.jsonl.gz','wt') as f:
        for j,m in enumerate(meta):f.write(json.dumps(dict(variable=j,**m),ensure_ascii=False)+'\n')
    with gzip.open(out/'contexts.jsonl.gz','wt') as f:
        for k in keys:
            z=support[k];f.write(json.dumps(dict(key=k,draws=z['draws'],rows=z['rows'],speakers=len(z['speakers']),items=len(z['items'])),ensure_ascii=False)+'\n')
    # Verify the quotient against the earlier independently verified P5 results.
    old=json.loads((a.previous_design/'report.json').read_text());checks=[]
    for arm in old['arms']:
        name=arm['arm'];raw=np.load(a.previous_design/(name+'_weights.npy'))[ix]
        w=np.bincount(groups,weights=np.array([co[i] for i in ix])*raw,minlength=n)/coeff
        ctx=float(sum((Qc.dot(w)-tc)**2));np.testing.assert_allclose(ctx,arm['context_objective'],rtol=1e-10,atol=1e-12)
        np.save(out/('previous_'+name+'.npy'),w);checks.append(dict(arm=name,context_objective=ctx))
    report=dict(status='TRAIN_matrices_frozen',receipts=receipts,manifest_rows=len(rows),active_foreign_rows=len(ix),
        variables=n,compression='exact proportional-column quotient for these objectives and constraints',
        context_cells=len(keys),supported_cells=int(supported.sum()),context_groups=len(sg),
        base_context=base_context,prompt_excess=excess,prompt_keys=pk,dimensions=dims,baselines=baselines,
        previous_reproduction=checks,continuation_rows=sum(count.values()),continuation_samples=sum(count[i]*rows[i][1] for i in count),
        model_optimizer_updates=0)
    write(out/'report.json',report)
    write(out/'finished.json',dict(report_sha256=file_sha(out/'report.json'),files={x.name:file_sha(x) for x in out.iterdir() if x.is_file()}))
    print(json.dumps({k:report[k] for k in ('variables','active_foreign_rows','base_context','previous_reproduction')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

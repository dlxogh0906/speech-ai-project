"""Recount every saved solution against original TRAIN rows and batch replay."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,collections,gzip,hashlib,json
import numpy as np
from train_h0_backbone_control import file_sha,stream
from train_w2vbert_ctc import read_split
from prepare_h0_p5_balance_design import contexts,tsv
from audit_h0_p5_readiness import write


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--p3-protocol',type=Path,required=True);ap.add_argument('--selection',type=Path)
    ap.add_argument('--out',type=Path);ap.add_argument('--expected',type=int,default=14)
    a=ap.parse_args();out=a.out or a.root/'verification';out.mkdir(exist_ok=False)
    selected=json.loads((a.selection or a.root/'solution_selection.json').read_text())
    paths=[Path(x['report']) for x in selected];assert len(paths)==a.expected,len(paths)
    solutions=[np.load(x.parent/'weights.npy') for x in paths];names=[x.parent.name for x in paths]
    z=dict(np.load(a.root/'matrices/arrays.npz'));W=np.column_stack([np.ones(len(z['coeff']))]+solutions)
    names=['C0']+names;N=len(names);expanded=W[z['row_to_variable']]
    mr=json.loads((a.root/'matrices/report.json').read_text());p=json.loads(a.p3_protocol.read_text());manifest=Path(p['manifest'])
    rows,labels=read_split(str(manifest),'train');sources=(manifest/'train.source').read_text().splitlines()
    batches=stream(rows,p['batch_samples'],p['seed']);h=hashlib.sha256();count=collections.Counter();co=collections.Counter();draws=samples=0;receipts=[]
    for step in range(p['stop_step']*p['accum']):
        ids=next(batches);h.update((' '.join(map(str,ids))+'\n').encode());draws+=len(ids);samples+=sum(rows[i][1] for i in ids)
        if step>=p['start_step']*p['accum']:
            count.update(ids)
            for i in ids:co[i]+=1/len(ids)
        if step+1 in (p['start_step']*p['accum'],p['stop_step']*p['accum']):
            r=dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest());assert r==(p['parent_exposure'] if not receipts else p['expected_final_exposure']);receipts.append(r)
    ix=sorted(i for i in count if sources[i].startswith('foreign'));np.testing.assert_array_equal(ix,z['ix'])
    np.testing.assert_allclose([co[i] for i in ix],z['original_coeff'],rtol=0,atol=1e-15)
    context=collections.defaultdict(lambda:np.zeros(N));tokens=collections.defaultdict(lambda:np.zeros(N));agg=collections.defaultdict(lambda:np.zeros(N))
    support=collections.defaultdict(lambda:dict(draws=0,rows=0,speakers=set(),items=set()))
    for j,i in enumerate(ix):
        parts=Path(rows[i][0]).stem.split('-');src=sources[i];item=parts[5];speaker=parts[0];l1=parts[3];v=co[i]*expanded[j]
        for dim,key in [('source',src),('prompt',src+':'+item),('speaker',speaker),('l1',l1),('speaker_source',speaker+':'+src),('l1_source',l1+':'+src)]:agg[dim,key]+=v
        for key,num in collections.Counter(contexts(labels[i])).items():
            mass=v*num/len(labels[i]);context[key]+=mass;tokens[key]+=count[i]*expanded[j]*num
            agg['phone',key[0]]+=mass;agg['phone_source',key[0]+':'+src]+=mass
            s=support[key];s['draws']+=count[i]*num;s['rows']+=1;s['speakers'].add(speaker);s['items'].add(item)
    groups=collections.defaultdict(list)
    for k,s in support.items():
        if s['draws']>=30 and s['rows']>=5 and len(s['speakers'])>=3 and len(s['items'])>=2:groups[k[:2]].append(k)
    objective=np.zeros(N);dispersion=np.zeros(N)
    for group,keys in groups.items():
        base=np.array([context[k][0] for k in keys]);total=base.sum();target=np.sqrt(base)/np.sqrt(base).sum()
        after=np.array([context[k] for k in keys]);objective+=np.sum(((after/total)-target[:,None])**2/target[:,None],axis=0)/len(groups)
        dispersion+=np.sum((after/after.sum(axis=0))**2,axis=0)/len(groups)-1/(len(keys)*len(groups))
    summaries=[];table=[]
    for c,name in enumerate(names):
        prompt=[]
        for src in ('foreign_clean','foreign_mis'):
            keys=sorted(k for dim,k in agg if dim=='prompt' and k.startswith(src+':'))
            values=np.array([agg['prompt',k][c] for k in keys]);base=np.array([agg['prompt',k][0] for k in keys]);hh=float(np.sum((values/values.sum())**2));bh=float(np.sum((base/base.sum())**2))
            prompt.append(dict(source=src,HHI=hh,fixed_source_HHI=float(np.sum((values/base.sum())**2)),
                source_total_ratio=float(values.sum()/base.sum()),reduction=(bh-hh)/(bh-1/len(keys))))
        support_ranges={}
        for dim in ('phone','speaker','l1','phone_source','speaker_source','l1_source'):
            vs=[v[c]/v[0] for (d,k),v in agg.items() if d==dim]
            support_ranges[dim]=dict(min=float(min(vs)),max=float(max(vs)))
        if c:
            rr=json.loads(paths[c-1].read_text());np.testing.assert_allclose(objective[c],rr['context_objective'],rtol=1e-9,atol=1e-11)
            for got,expected in zip(prompt,rr['prompt']):np.testing.assert_allclose(got['fixed_source_HHI'],expected['HHI'],rtol=1e-9)
            for dim in ('phone','speaker','l1'):
                np.testing.assert_allclose(list(support_ranges[dim].values()),list(rr['ratios'][dim].values()),rtol=1e-8,atol=1e-10)
            for src in ('foreign_clean','foreign_mis'):np.testing.assert_allclose(agg['source',src][c],agg['source',src][0],rtol=2e-6,atol=2e-6)
        summaries.append(dict(name=name,context_objective=float(objective[c]),mean_actual_context_excess_HHI=float(dispersion[c]),
            prompt=prompt,support_ranges=support_ranges,context_mass_min_ratio=float(min(v[c]/v[0] for v in context.values())),
            context_mass_max_ratio=float(max(v[c]/v[0] for v in context.values())),
            s_u_l=float(sum(v[c] for k,v in tokens.items() if (k[0],k[2],k[3])==('ㅜ','ㅅ','ㄹ'))),
            s_o_l=float(sum(v[c] for k,v in tokens.items() if (k[0],k[2],k[3])==('ㅗ','ㅅ','ㄹ')))))
        for (dim,k),v in sorted(agg.items()):table.append(dict(condition=name,dimension=dim,key=k,base=float(v[0]),weighted=float(v[c]),ratio=float(v[c]/v[0])))
    tsv(out/'group_contributions.tsv',table)
    tsv(out/'context_contributions.tsv',[dict(condition=name,phone=k[0],position=k[1],left=k[2],right=k[3],base=float(v[0]),weighted=float(v[c]),ratio=float(v[c]/v[0])) for c,name in enumerate(names) for k,v in sorted(context.items())])
    write(out/'report.json',dict(status='raw_TRAIN_recount_passed',summaries=summaries,receipts=receipts,
        active_rows=len(ix),supported_contexts=sum(map(len,groups.values())),source_totals_preserved=True,
        full_foreign_phone_support=len({k[0] for k in context}),model_optimizer_updates=0,
        hashes={str(x):file_sha(x) for x in paths}))
    print('raw TRAIN contribution verification passed for',len(paths),'solutions',flush=True)

if __name__=='__main__':main()

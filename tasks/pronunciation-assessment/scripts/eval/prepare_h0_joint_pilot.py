"""Prepare the fixed lambda=.3 fold pilot, with hard joint support constraints."""
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
    ap.add_argument('--backbone-protocol',type=Path,required=True)
    ap.add_argument('--holdouts',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fold',type=int,default=0,choices=range(5))
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    bp=json.loads(a.backbone_protocol.read_text());hp=json.loads((a.holdouts/'protocol.json').read_text())
    hr=json.loads((a.holdouts/'report.json').read_text())
    assert json.loads((a.holdouts/'verification.json').read_text())['status']=='passed'
    indices_path=a.holdouts/('fold_%d'%a.fold)/'row_indices.npz'
    bindings=dict(hr['bindings']);bindings[str(indices_path)]=file_sha(indices_path)
    bindings[str(a.holdouts/'protocol.json')]=file_sha(a.holdouts/'protocol.json')
    bindings[str(a.holdouts/'verification.json')]=file_sha(a.holdouts/'verification.json')
    for path,digest in bindings.items():
        assert file_sha(path)==digest, path
    manifest=Path(bp['stage2']['manifest'])
    for suffix in ('tsv','ltr','source'):
        path=manifest/('train.'+suffix);bindings[str(path)]=file_sha(path)
    p=dict(manifest=str(manifest),batch_samples=bp['batch_samples'],seed=bp['seed'],accum=bp['accum'],start_step=2000,stop_step=4000)
    design=dict(version='h0_p5_joint_single_candidate_v1',created=time.time(),fold=a.fold,
        candidate=dict(lambda_value=.3,objective='P6 context/C0 + .3 * mean source normalized excess prompt HHI',
            context_support=dict(token_draws=30,manifest_rows=5,speakers=3,items=2),
            context_target='sqrt baseline context mass normalized within supported phone-position group',
            no_lambda_or_weight_family_search=True),
        hard_constraints=dict(weight=[.25,4],source_totals='exact nominal loss coefficients',native_weight=1,
            phone=[.8,1.2],speaker=[.5,2],phone_source=[.8,1.2],speaker_source=[.5,2],L1_share_absolute_change=.01,
            rare_groups='same bounds for every positive-support group; zero support listed, never synthesized'),
        treatment_gate=dict(each_source_excess_HHI_reduction_min=.5,context_deviation_reduction_min=.25,
            max_constraint_violation=2e-6,objective_gap_relative_to_max_1=2e-5,
            fail='do not start speech training; do not change lambda, support limits, fold, or target after seeing this result'),
        pilot_order=[0,1,2,3,4],fold_plan=hp['folds'],holdout_design=str(a.holdouts),
        lineage='public w2v-BERT SSL -> fold-filtered Stage1 10k -> fold-filtered common Stage2 2k -> paired C0 / lambda=.3 to 4k',
        old_H0_checkpoint_allowed=False,
        training=dict(stage1=bp['stage1'],stage2=bp['stage2'],seed=42,batch_samples=1600000,accum=4,
            fork='same newly trained fold 2k model/RNG/AMP; both AdamW reset, LR continues 40k schedule',
            loss='mean(weight_i * CTC_NLL_i / target_length_i); never divide by weight sum',
            checkpoints='Stage1 10k, common Stage2 2k, both arms 4k; no posthoc checkpoint selection',
            controls=bp['controls'],manifest_override='use only corresponding fold row selections before training or sampling'),
        evaluation=dict(calibration='fold CAL normal observed sites, one 1% budget threshold per fixed arm; freeze before inner/outer',
            roles='fixed existing train/CAL/inner/outer speaker and prompt roles; no heldout data in weights or preprocessing',
            primary=['exact diagnosis over all annotation-identifiable errors','observed normal FPR','normal/error Coverage separately',
                'unobserved-case conservative bounds','deletion and insertion exact match and coverage'],
            pilot_gate='inner validation: candidate observed normal FPR <=1%, full-error exact recall > C0, normal/error coverage each >= C0; existing error-support>=30 phone/deletion/insertion recall drop <=5pp; normal gap false insertion rise <=.2pp',
            pilot_gate_scope='computational continuation rule; pilot failure is not population impossibility',
            outer='no outer-policy fitting; report every fixed fold and pooled out-of-fold counts if all 5 run; never claim success from pilot alone',
            complete_rule='all five folds, same fixed method and CAL policy; report paired speaker and prompt uncertainty, rare support, RS/RW and L1; positive pilot is insufficient',
            negative_result='terminate this fixed weighting comparison; no same-family rescue tuning',
            independent_final=False,production_API=False),
        unresolved_before_model_training=['independent fold TRAIN recount','audio/label/alias preflight','persistent checkpoint capacity','weighted training and full-denominator evaluation runtime checks'],
        source_sha256=file_sha(__file__),bindings=bindings,public_SSL=bp['backbones']['w2vbert'],
        stage2_plan=p,model_optimizer_updates=0,model_inference_count=0)
    write(a.out/'protocol.json',design)
    rows,labels=read_split(str(manifest),'train');sources=(manifest/'train.source').read_text().splitlines()
    selected=np.load(indices_path)['stage2_train']
    rows=[rows[i] for i in selected];labels=[labels[i] for i in selected];sources=[sources[i] for i in selected]
    out=a.out/'matrices';out.mkdir();np.save(out/'manifest_row_indices.npy',selected)
    batches=stream(rows,p['batch_samples'],p['seed']);h=hashlib.sha256();count=collections.Counter();co=collections.Counter()
    draws=samples=0;receipts=[];stage1_receipt=None
    for micro in range(p['stop_step']*p['accum']):
        ids=next(batches);h.update((' '.join(map(str,ids))+'\n').encode());draws+=len(ids);samples+=sum(rows[i][1] for i in ids)
        if micro>=p['start_step']*p['accum']:
            count.update(ids)
            for i in ids:co[i]+=1/len(ids)
        if micro+1 in (p['start_step']*p['accum'],p['stop_step']*p['accum']):
            receipts.append(dict(row_draws=draws,audio_samples=samples,batch_order_sha256=h.hexdigest()))
    print('fold stream replayed',receipts,flush=True)
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
    for dim in ('phone', 'speaker'):
        raw=sparse.diags(baselines[dim]).dot(matrices[dim])
        pieces=[];joint_keys=[];joint_base=[]
        for source in src:
            mask=np.array([m['source']==source for m in meta],dtype=float)
            part=raw.dot(sparse.diags(mask)).tocsr()
            base=np.asarray(part.sum(axis=1)).ravel();keep=np.flatnonzero(base>0)
            pieces.append(sparse.diags(1/base[keep]).dot(part[keep]))
            joint_keys.extend(k+':'+source for k in np.array(dims[dim])[keep])
            joint_base.extend(base[keep].tolist())
        matrices[dim+'_source']=sparse.vstack(pieces,format='csr')
        dims[dim+'_source']=joint_keys;baselines[dim+'_source']=joint_base
    for name,M in matrices.items():sparse.save_npz(out/(name+'.npz'),M)
    np.savez(out/'arrays.npz',ix=ix,row_to_variable=groups,coeff=coeff,original_coeff=np.array([co[i] for i in ix]),
        tc=tc,tp=tp,supported=supported,prompt_source=sp,prompt_base=bp,source_total=st,context_base=ab)
    with gzip.open(out/'variables.jsonl.gz','wt') as f:
        for j,m in enumerate(meta):f.write(json.dumps(dict(variable=j,**m),ensure_ascii=False)+'\n')
    with gzip.open(out/'contexts.jsonl.gz','wt') as f:
        for k in keys:
            z=support[k];f.write(json.dumps(dict(key=k,draws=z['draws'],rows=z['rows'],speakers=len(z['speakers']),items=len(z['items'])),ensure_ascii=False)+'\n')
    report=dict(status='fold_TRAIN_joint_matrices_frozen',receipts=receipts,manifest_rows=len(rows),active_foreign_rows=len(ix),
        variables=n,context_cells=len(keys),supported_cells=int(supported.sum()),context_groups=len(sg),
        base_context=base_context,prompt_excess=excess,prompt_keys=pk,dimensions=dims,baselines=baselines,
        continuation_rows=sum(count.values()),continuation_samples=sum(count[i]*rows[i][1] for i in count),
        continuation_source_draws=dict(collections.Counter({s:sum(count[i] for i in count if sources[i]==s) for s in set(sources)})),
        protocol_sha256=file_sha(a.out/'protocol.json'),model_optimizer_updates=0)
    write(out/'report.json',report)
    write(out/'finished.json',dict(report_sha256=file_sha(out/'report.json'),files={x.name:file_sha(x) for x in out.iterdir() if x.is_file()}))
    print(json.dumps({k:report[k] for k in ('variables','active_foreign_rows','base_context','continuation_source_draws')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

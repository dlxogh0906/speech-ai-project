"""P4 frozen-readout comparison on full development denominators, no selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from diagnose_h0_p4 import MODES,read_jsonl,write_json
from fit_h0_representation_probes import counts,threshold,table
from eval_h0_intermediate import evaluate_ledgers
from mdd.h0_phone_audit import paired_cluster_ci
from train_h0_backbone_control import file_sha


def matched_ids(all_sites, ids, ledger):
    by_key={(s['uid'],s['q']):s for s in ledger}
    result=[by_key[all_sites[i]['uid'],all_sites[i]['q']] for i in ids]
    for i,s in zip(ids,result):
        if (s['expected'],s['actual'])!=(all_sites[i]['expected'],all_sites[i]['actual']):
            raise ValueError('site truth mismatch')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    root=Path(a.root);p=json.loads((root/'protocol.json').read_text());p1=Path(p['p1']);p3=Path(p['p3'])
    dest=root/'comparison';dest.mkdir(exist_ok=False)
    finished={arm:json.loads((root/arm/'finished.json').read_text()) for arm in p['arms']}
    if finished['B0']['audio_order_and_PCM_sha256']!=finished['B3']['audio_order_and_PCM_sha256']:raise ValueError('audio changed')
    ids=np.load(root/'B0/site_ids.npy')
    if not np.array_equal(ids,np.load(root/'B3/site_ids.npy')):raise ValueError('site order differs')
    all_sites=read_jsonl(p1/'sites.jsonl');ss=[all_sites[i] for i in ids]
    fit=json.loads((p1/'probes/fit_protocol.json').read_text());classes=fit['classes'];classid={v:i for i,v in enumerate(classes)}
    expected=np.array([classid[s['expected']] for s in ss]);actual=np.array([classid[s['actual'] or '<DEL>'] for s in ss])
    cal=np.array([s['role']=='calibration' for s in ss]);selected=~cal
    seen=np.array([s['split']=='valid' for s in ss]);normal=expected==actual
    suri=np.array([s['item']=='RW054' and s['whole_tag_normal'] and s['expected']=='ㅜ' for s in ss])
    if not a.smoke and int(suri.sum())!=475:raise ValueError('Suri cohort changed')
    groups={}
    for split,sm in [('valid',seen),('valid_unseen',~seen)]:
        groups[split,'all','all']=np.flatnonzero(sm&selected)
        for dim in ('expected','position','item'):
            values=classes[:-1] if dim=='expected' else sorted({s[dim] for s in ss})
            for val in values:
                groups[split,dim,val]=np.flatnonzero(sm&selected&np.array([s[dim]==val for s in ss]))
        groups[split,'actual','<DEL>']=np.flatnonzero(sm&selected&(actual==classid['<DEL>']))
    rows=[];main_rows=[];predictions={};taus={};ctc={};auxiliary_reports={};threeway=[];pairs=[];surirows=[];confusions=[]
    for arm in p['arms']:
        f=finished[arm]
        if f['smoke']!=a.smoke or f['sites']!=len(ss) or f['final_reproduced_sites']!=len(ss):raise ValueError('incomplete extraction')
        avail=np.load(root/arm/'available.npy');aux=matched_ids(all_sites,ids,read_jsonl(root/arm/'aux_sites.jsonl'))
        final=matched_ids(all_sites,ids,read_jsonl(p3/(arm+'_eval_4000')/'sites.jsonl'))
        ctc[arm]={kind:np.array([classid[s['predicted'] or '<DEL>'] for s in data]) for kind,data in [('aux',aux),('final',final)]}
        auxdest=root/arm/'aux_evaluation';auxdest.mkdir(exist_ok=False)
        os.link(root/arm/'aux_sites.jsonl',auxdest/'sites.jsonl');os.link(root/arm/'aux_sequences.jsonl',auxdest/'sequences.jsonl')
        prior=json.loads((p3/'design_v2/protocol.json').read_text());prior['_path']=str(root/'protocol.json')
        auxiliary_reports[arm]=evaluate_ledgers(auxdest,prior,arm+'_B3_aux_readout',4000,p['bindings'][str(p3/arm/'step_4000.pt')])
        for k,mode in enumerate(MODES):
            logits=np.load(root/arm/(mode+'.npy'));pred=logits.argmax(-1);score=logits.max(-1)-logits[np.arange(len(ss)),expected]
            ok=avail[:,k];tau=threshold(score,cal&normal,pred!=expected,ok)
            readout='tag' if mode=='tag_matched_readout' else 'auto'
            saved=json.loads((p1/'probes'/('w2vbert_2000_L12_'+readout)/'calibration.json').read_text())
            oldtau=-np.inf if saved['negative_infinity'] else saved['threshold']
            calibration=counts(np.flatnonzero(cal),expected,actual,pred,score,ok,tau)
            if calibration['normal_fp']>int(calibration['normal']*.01):raise ValueError('CAL budget')
            taus[arm,mode]=tau;predictions[arm,mode]=dict(pred=pred,score=score,ok=ok)
            for policy,t in [('seen_CAL_recalibrated',tau),('parent_CAL_transfer',oldtau)]:
                for (split,dim,key),ix in groups.items():
                    row=dict(arm=arm,mode=mode,policy=policy,threshold=float(t) if np.isfinite(t) else None,
                        split=split,dimension=dim,key=key,items=len({ss[i]['item'] for i in ix}),
                        **counts(ix,expected,actual,pred,score,ok,t))
                    rows.append(row)
                    if dim=='all':main_rows.append(row)
            six=np.flatnonzero(suri);pc=pred[six]
            surirows.append(dict(arm=arm,mode=mode,total=len(six),available=int(ok[six].sum()),
                u_correct=int(((pc==classid['ㅜ'])&ok[six]).sum()),o_predicted=int(((pc==classid['ㅗ'])&ok[six]).sum()),
                u_minus_o_quantiles=np.quantile(logits[six,classid['ㅜ']]-logits[six,classid['ㅗ']],[.1,.5,.9]).tolist(),
                aux_u=int((ctc[arm]['aux'][six]==classid['ㅜ']).sum()),final_u=int((ctc[arm]['final'][six]==classid['ㅜ']).sum())))
            # Actual-o checks distinguish false-u bias from recovering normal-u.
            for split,sm in [('valid',seen),('valid_unseen',~seen)]:
                for cohort,cm in [('all_actual_u_o',np.ones(len(ss),bool)),('expected_u',expected==classid['ㅜ']),
                                  ('expected_o',expected==classid['ㅗ'])]:
                    m=selected&sm&cm&ok&np.isin(actual,[classid['ㅜ'],classid['ㅗ']]);ix=np.flatnonzero(m)
                    y=actual[ix]==classid['ㅗ'];diff=logits[ix,classid['ㅗ']]-logits[ix,classid['ㅜ']]
                    pairs.append(dict(arm=arm,mode=mode,split=split,cohort=cohort,
                        actual_u=int((~y).sum()),actual_o=int(y.sum()),items=len({ss[i]['item'] for i in ix}),
                        o_to_u=int((y&(pred[ix]==classid['ㅜ'])).sum()),u_to_o=int(((~y)&(pred[ix]==classid['ㅗ'])).sum()),
                        o_correct=int((y&(pred[ix]==classid['ㅗ'])).sum()),u_correct=int(((~y)&(pred[ix]==classid['ㅜ'])).sum()),
                        auc_o_vs_u=float(roc_auc_score(y,diff)) if len(np.unique(y))==2 else None))
            for split,sm in [('valid',seen),('valid_unseen',~seen)]:
                c=collections.Counter((ss[i]['expected'],ss[i]['actual'] or '<DEL>',classes[pred[i]],ss[i]['position'],bool(ok[i]))
                                      for i in np.flatnonzero(selected&sm))
                confusions.extend(dict(arm=arm,mode=mode,split=split,expected=t[0],actual=t[1],predicted=t[2],position=t[3],available=t[4],n=n)
                                  for t,n in c.items())
        pp=predictions[arm,'parent_auto'];aok=ctc[arm]['aux']==actual;fok=ctc[arm]['final']==actual
        pok=(pp['pred']==actual)&pp['ok']
        for label,m in [('suri',suri),('unseen_all',selected&~seen),('seen_all',selected&seen),
                        ('unseen_errors',selected&~seen&~normal),('unseen_normals',selected&~seen&normal)]:
            c=collections.Counter((bool(pok[i]),bool(aok[i]),bool(fok[i]),bool(pp['ok'][i])) for i in np.flatnonzero(m))
            threeway.extend(dict(arm=arm,cohort=label,probe_correct=k[0],aux_correct=k[1],final_correct=k[2],probe_available=k[3],n=n) for k,n in c.items())
    contrasts=[]
    for mode in MODES:
        x=predictions['B0',mode];y=predictions['B3',mode]
        wx=(x['pred']!=expected)&x['ok']&(x['score'].astype(np.float64)>=taus['B0',mode])
        wy=(y['pred']!=expected)&y['ok']&(y['score'].astype(np.float64)>=taus['B3',mode])
        deltas=dict(normal_fp=(wy&normal).astype(int)-(wx&normal),
                    exact=((y['pred']==actual)&wy&~normal).astype(int)-((x['pred']==actual)&wx&~normal),
                    phone_correct=((y['pred']==actual)&y['ok']).astype(int)-((x['pred']==actual)&x['ok']))
        for axis in ('speaker','item'):
            buckets=collections.defaultdict(collections.Counter)
            for i in np.flatnonzero(selected&~seen):
                c=buckets[ss[i][axis]];c.update(normal=int(normal[i]),errors=int(not normal[i]),total=1)
                for name,v in deltas.items():c[name]+=int(v[i])
            for name,den in [('normal_fp','normal'),('exact','errors'),('phone_correct','total')]:
                contrasts.append(dict(mode=mode,axis=axis,metric=name,**paired_cluster_ci(list(buckets.values()),name,den)))
    for name,data in [('all_groups',rows),('summary',main_rows),('suri',surirows),('three_way',threeway),('actual_u_o',pairs),('confusions',confusions),('paired',contrasts)]:table(dest/(name+'.tsv'),data)
    report=dict(status='completed',smoke=a.smoke,arms=p['arms'],primary='parent_auto fixed 2k TRAIN readout and fixed 2k bounds',
        summary=main_rows,suri=surirows,three_way=threeway,actual_u_o=pairs,paired=contrasts,
        extraction=finished,auxiliary_suri={k:v['suri'] for k,v in auxiliary_reports.items()},
        auxiliary_metrics={k:v['results'] for k,v in auxiliary_reports.items()},
        auxiliary_gaps={k:v['sequences'] for k,v in auxiliary_reports.items()},
        limitations=p['limitations'],source_sha256=file_sha(__file__),protocol_sha256=file_sha(root/'protocol.json'))
    write_json(dest/'report.json',report)
    write_json(dest/'finished.json',dict(status='completed',encoder_updates=0,probe_updates=0,sites_per_arm=len(ss)))


if __name__=='__main__':main()

"""Paired P1 contrasts, CTC recovery, window drift, and a shareable figure."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import csv
import json
import numpy as np
from diagnose_h0_representations import LAYERS,WINDOWS,write_json
from fit_h0_representation_probes import table


def paired(idx, cluster, lhs, rhs, normal, errors):
    keys,inv=np.unique(cluster[idx],return_inverse=True)
    data=np.stack([rhs['fp']-lhs['fp'],normal,rhs['exact']-lhs['exact'],errors],1)
    sums=np.zeros((len(keys),4));np.add.at(sums,inv,data[idx])
    res=dict(clusters=len(keys),normal=int(normal[idx].sum()),errors=int(errors[idx].sum()))
    if len(keys)>1:
        rng=np.random.RandomState(20261027);z=sums[rng.randint(len(keys),size=(2000,len(keys)))].sum(1)
    for name,num,den in [('fpr',0,1),('exact_recall',2,3)]:
        res[name+'_difference']=float(sums[:,num].sum()/sums[:,den].sum()) if sums[:,den].sum() else None
        if len(keys)>1:
            v=z[z[:,den]>0,num]/z[z[:,den]>0,den]
            res[name+'_ci95']=np.quantile(v,[.025,.975]).tolist() if len(v) else None
    return res


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--plot-only',action='store_true');a=ap.parse_args()
    root=Path(a.root);out=root/'interpretation'
    if a.plot_only:
        plot(root,out);return
    out.mkdir(exist_ok=False)
    p=json.loads((root/'protocol.json').read_text());classes=p['probe']['classes'];ids={c:i for i,c in enumerate(classes)}
    sites=[json.loads(s) for s in (root/'sites.jsonl').open()];n=len(sites)
    expected=np.array([ids[s['expected']] for s in sites]);actual=np.array([ids[s['actual'] or '<DEL>'] for s in sites])
    normal=(expected==actual);errors=~normal
    split=np.array([s['split'] for s in sites]);role=np.array([s['role'] for s in sites]);item=np.array([s['item'] for s in sites]);speaker=np.array([s['speaker'] for s in sites])
    conditions=['%s_%d'%(k,t) for k in ('w2vbert','xlsr300m') for t in (2000,4000)]
    common=np.zeros(n,dtype=bool);common[np.load(root/'probes/common_window_ids.npy')]=True
    events={};windows=[];recovery=[];contrasts=[];pair_rows=[]
    for name in conditions:
        kind,step=name.rsplit('_',1)
        ledger={}
        for line in (Path(p['experiments'])/('%s_eval_%s'%(kind,step))/'sites.jsonl').open():
            s=json.loads(line);ledger[s['uid'],s['q']]=ids[s['predicted'] or '<DEL>']
        ctc=np.array([ledger.get((s['uid'],s['q']),-1) for s in sites])
        available=np.load(root/name/'available.npy');bounds=np.load(root/name/'window_bounds.npy')
        for sp in ('valid','valid_unseen'):
            base=(split==sp)&(role=='selection')
            for group,g in [('all',np.ones(n,bool)),('suri_u', (item=='RW054')&(expected==ids['ㅜ'])&np.array([s['whole_tag_normal'] for s in sites]))]:
                ix=np.flatnonzero(base & g & available[:,0]&available[:,2])
                if not len(ix):continue
                b=bounds[ix]
                inter=np.maximum(0,np.minimum(b[:,1],b[:,5])-np.maximum(b[:,0],b[:,4]))
                union=np.maximum(b[:,1],b[:,5])-np.minimum(b[:,0],b[:,4])
                distance=abs((b[:,0]+b[:,1])-(b[:,4]+b[:,5]))*.01
                windows.append(dict(condition=name,split=sp,group=group,sites=len(ix),
                    median_auto_tag_center_difference_ms=float(np.median(distance)*1000),
                    p90_auto_tag_center_difference_ms=float(np.quantile(distance,.9)*1000),
                    median_auto_tag_iou=float(np.median(inter/union)),no_overlap=int((inter==0).sum()),
                    median_auto_frames=float(np.median(b[:,1]-b[:,0])),median_tag_frames=float(np.median(b[:,5]-b[:,4]))))
        for layer in LAYERS:
            for wi,window in enumerate(WINDOWS):
                key='%s_L%d_%s'%(name,layer,window);dest=root/'probes'/key
                logits=np.load(dest/'logits.npy',mmap_mode='r');pred=logits.argmax(1)
                score=(logits.max(1)-logits[np.arange(n),expected]).astype(float)
                c=json.loads((dest/'calibration.json').read_text());tau=-np.inf if c['negative_infinity'] else c['threshold']
                av=available[:,wi];warn=(pred!=expected)&av&(score>=tau)
                events[key]=dict(fp=(normal&warn).astype(int),exact=(errors&warn&(pred==actual)).astype(int))
                for sp in ('valid','valid_unseen'):
                    ix=np.flatnonzero((split==sp)&(role=='selection'))
                    ctc_wrong=ctc!=actual
                    for group,m in [('all',np.ones(n,bool)),('common_windows',common),
                                    ('suri_475_u',(item=='RW054')&(expected==ids['ㅜ'])&np.array([s['whole_tag_normal'] for s in sites]))]:
                        jj=ix[m[ix]]
                        if not len(jj):continue
                        cw=ctc_wrong[jj];correct=(pred[jj]==actual[jj])&av[jj]
                        recovery.append(dict(condition=name,layer=layer,window=window,split=sp,group=group,
                            normal=int(normal[jj].sum()),errors=int(errors[jj].sum()),
                            ctc_wrong_normal=int((cw&normal[jj]).sum()),
                            probe_recovers_ctc_wrong_normal=int((cw&normal[jj]&correct).sum()),
                            probe_breaks_ctc_correct_normal=int((~cw&normal[jj]&~correct&av[jj]).sum()),
                            ctc_wrong_errors=int((cw&errors[jj]).sum()),
                            probe_recovers_ctc_wrong_errors=int((cw&errors[jj]&correct).sum()),
                            calibrated_recovered_errors=int((cw&errors[jj]&correct&warn[jj]).sum())))
        # Every contrast was specified by structure, not by highest observed AUC.
        comparisons=[('L24_auto','L6_auto'),('L24_auto','L12_auto')]
        comparisons += [('L%d_auto'%l,'L%d_%s'%(l,w)) for l in LAYERS for w in ('multi','tag')]
        for left,right in comparisons:
            for sp in ('valid','valid_unseen'):
                ix=np.flatnonzero((split==sp)&(role=='selection')&common)
                for cluster,g in [('speaker',speaker),('item',item)]:
                    contrasts.append(dict(condition=name,left=left,right=right,split=sp,cluster=cluster,
                        cohort='common windows across all checkpoints',
                        **paired(ix,g,events[name+'_'+left],events[name+'_'+right],normal,errors)))
    for k in ('w2vbert','xlsr300m'):
        for l in LAYERS:
            for w in WINDOWS:
                for sp in ('valid','valid_unseen'):
                    ix=np.flatnonzero((split==sp)&(role=='selection')&common)
                    for cluster,g in [('speaker',speaker),('item',item)]:
                        contrasts.append(dict(condition=k,left='2000_L%d_%s'%(l,w),right='4000_L%d_%s'%(l,w),
                            split=sp,cluster=cluster,cohort='common windows across all checkpoints',
                            **paired(ix,g,events['%s_2000_L%d_%s'%(k,l,w)],events['%s_4000_L%d_%s'%(k,l,w)],normal,errors)))
    table(out/'paired_contrasts.tsv',contrasts);table(out/'ctc_recovery.tsv',recovery);table(out/'windows.tsv',windows)
    # A shared output capacity can have unsupported classes; explicitly list them.
    fit=json.loads((root/'probes/fit_protocol.json').read_text())
    write_json(out/'report.json',dict(windows=windows,recovery=recovery,
        unsupported_train_classes=[c for c,n in fit['class_support'].items() if not n],
        sparse_train_classes={c:n for c,n in fit['class_support'].items() if 0<n<30},
        limits=['Reference-conditioned windows remain text dependent; classifier itself has no reference feature.',
                'TAG-window success is diagnostic and has no direct runtime implementation.',
                'Fixed linear probe failure is not absence of all information.',
                'Paired CIs condition on fixed heads/thresholds; no CAL or training-seed refit.',
                '9 development items; no independent final selection or deployment.']))
    plot(root,out)
    write_json(out/'finished.json',dict(status='completed',paired_contrasts=len(contrasts)))


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suri=list(csv.DictReader((root/'probes/suri.tsv').open(),delimiter='\t'))
    summary=list(csv.DictReader((root/'probes/summary.tsv').open(),delimiter='\t'))
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors={'auto':'#247ba0','multi':'#e49b25','tag':'#8756a8'}
    for col,k in enumerate(('w2vbert','xlsr300m')):
        for step in (2000,4000):
            for w in WINDOWS:
                ss=sorted([s for s in suri if s['condition']=='%s_%d'%(k,step) and s['window']==w],key=lambda s:int(s['layer']))
                yy=[float(s['predicts_u'])/475*100 for s in ss]
                axes[0,col].plot(LAYERS,yy,marker='o',color=colors[w],ls='-' if step==2000 else '--',label='%s %dk'%(w,step//1000))
                ss=sorted([s for s in summary if s['condition']=='%s_%d'%(k,step) and s['window']==w and s['split']=='valid_unseen' and s['group']=='all' and s['policy']=='calibrated'],key=lambda s:int(s['layer']))
                axes[1,col].plot(LAYERS,[float(s['normal_fpr'])*100 for s in ss],marker='o',color=colors[w],ls='-' if step==2000 else '--')
        axes[0,col].set_title(k);axes[0,col].set_ylim(0,101)
        axes[1,col].set_ylim(0,7)
        axes[1,col].axhline(1,color='black',ls=':',lw=1,label='1% target')
        for ax in axes[:,col]:ax.set_xticks(LAYERS);ax.set_xlabel('Encoder layer');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Suri: predicted /u/ (% of fixed 475)')
    axes[1,0].set_ylabel('Unseen normal FPR (%)\nFrozen seen-CAL threshold')
    axes[0,1].legend(ncol=2,fontsize=8)
    fig.suptitle('Frozen layer/window diagnostics: representation and calibration are separate')
    fig.savefig(out/'P1_layers_windows.svg');fig.savefig(out/'P1_layers_windows.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

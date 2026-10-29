"""Evaluate fixed fold C0/lambda=.3 with observed and full known denominators."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import gzip
import json
import time
import numpy as np
import torch
from mdd.h0_joint_evaluation import score_record, calibrate, aggregate, site_counts, gap_counts, pilot_gate
from train_h0_backbone_control import Backbone, file_sha
from train_w2vbert_ctc import read_split,load_vocab
from eval_h0_rp_stress import read_audio,forward
from audit_h0_p5_readiness import write


def records(path):
    with gzip.open(str(path),'rt') as f:
        return [json.loads(line) for line in f]


def evaluate(root,arm):
    p=json.loads((root/'protocol.json').read_text())
    runtime=json.loads((root/'runtime_check.json').read_text())
    assert runtime['status']=='passed'
    for path,h in runtime['source_bindings'].items(): assert file_sha(path)==h,path
    out=root/'evaluation'/arm;out.mkdir(parents=True,exist_ok=False)
    ckpath=root/'training'/arm/'step_4000.pt'
    receipt=json.loads((ckpath.parent/'finished.json').read_text())
    digest=file_sha(ckpath); assert digest==receipt['checkpoint_sha256']
    ck=torch.load(ckpath,map_location='cpu')
    assert ck['phase']==arm and ck['step']==4000 and ck['protocol_sha256']==file_sha(root/'protocol.json')
    vocab=ck['vocab'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Backbone('w2vbert',p['public_SSL']['path'],max(vocab.values())+1,device)
    model.load_state_dict(ck['model'],strict=True);model.eval();del ck
    binding=json.loads((root/'manifests/report.json').read_text())['bindings']
    audio={}
    with gzip.open(str(root/'audio_annotation_preflight/audio_bindings.jsonl.gz'),'rt') as f:
        for line in f:
            x=json.loads(line)
            if any(role!='train' for role in x['roles']):audio[x['path']]=x
    reports={}; threshold=None; start=time.time()
    for role in ('cal','inner','outer'):
        manifest=root/'manifests/stage2';path=manifest/(role+'.tsv')
        assert file_sha(path)==binding[str(path)]['sha256']
        annpath=root/'audio_annotation_preflight'/(role+'_annotations.jsonl.gz')
        assert file_sha(annpath)==runtime['input_bindings'][str(annpath)]
        annotations=records(annpath);rows,_=read_split(str(manifest),role)
        assert len(rows)==len(annotations)
        result=[];lpdir=out/(role+'_log_probs');lpdir.mkdir()
        with gzip.open(str(out/(role+'_records.jsonl.gz')),'xt') as dest:
            for i,(r,(path,n)) in enumerate(zip(annotations,rows)):
                assert r['row']==i and r['uid']==Path(path).stem
                x,h=read_audio(path);assert len(x)==n==audio[path]['samples'] and h==audio[path]['pcm_sha256']
                lp=forward(model,x)
                np.savez_compressed(lpdir/(r['uid']+'.npz'),lp=lp)
                value=score_record(r,lp,vocab);value.update(pcm_sha256=h,frames=len(lp))
                dest.write(json.dumps(value,ensure_ascii=False)+'\n');result.append(value)
                if (i+1)%500==0:
                    status=dict(status='evaluating',arm=arm,role=role,completed=i+1,total=len(rows),elapsed_seconds=time.time()-start)
                    write(out/'live_status.json',status);print(json.dumps(status),flush=True)
        if role=='cal':
            calibration=calibrate(result);threshold=calibration['threshold']
            write(out/'calibration.json',dict(calibration,checkpoint_sha256=digest))
        reports[role]=aggregate(result,threshold,vocab)
        write(out/(role+'_report.json'),reports[role])
    write(out/'finished.json',dict(status='fixed_arm_evaluated',arm=arm,checkpoint_sha256=digest,
        calibration=calibration,recordings=sum(x['recordings'] for x in reports.values()),
        elapsed_seconds=time.time()-start,model_optimizer_updates=0,
        source_bindings=runtime['source_bindings'],vocab=vocab,
        output_hashes={str(f):file_sha(f) for f in out.glob('*.json*')}))


def paired(a,b,ta,tb,axis):
    keys=['normal','error','observed_normal','observed_error','fp','exact']
    tables=[collections.defaultdict(collections.Counter) for _ in range(2)]
    for r,s in zip(a,b):
        assert (r['uid'],r['speaker'],r['item'])==(s['uid'],s['speaker'],s['item'])
        assert [(x['status'],x['actual']) for x in r['sites']]==[(x['status'],x['actual']) for x in s['sites']]
        for table,row,t in zip(tables,(r,s),(ta,tb)):table[row[axis]].update(site_counts(row['sites'],t))
    units=sorted(tables[0]); assert units==sorted(tables[1])
    values=[np.array([[tab[u][k] for k in keys] for u in units],float) for tab in tables]
    rng=np.random.RandomState(20261029);deltas=collections.defaultdict(list)
    if len(units)<2:return dict(units=len(units),ci95=None)
    for _ in range(2000):
        ix=rng.randint(len(units),size=len(units));x,y=[v[ix].sum(0) for v in values]
        for name,num,den in [('FPR',4,2),('full_error_exact',5,1),('normal_coverage',2,0),('error_coverage',3,1)]:
            if x[den] and y[den]:deltas[name].append(y[num]/y[den]-x[num]/x[den])
    return dict(units=len(units),replicates=2000,ci95={k:np.quantile(v,[.025,.975]).tolist() for k,v in deltas.items()},
                scope='fixed CAL/models; separate speaker or prompt clusters, no training/CAL resampling')


def compare(root):
    out=root/'comparison';out.mkdir(exist_ok=False)
    folders=[root/'evaluation'/arm for arm in ('C0','weighted')]
    thresholds=[json.loads((d/'calibration.json').read_text())['threshold'] for d in folders]
    checks={};reports={};uncertainty={}
    for role in ('cal','inner','outer'):
        outputs=[records(d/(role+'_records.jsonl.gz')) for d in folders]
        rebuilt=[]
        for d,rs,t in zip(folders,outputs,thresholds):
            vocab=json.loads((d/'finished.json').read_text())['vocab']
            report=aggregate(rs,t,vocab);assert report==json.loads((d/(role+'_report.json')).read_text())
            rebuilt.append(report)
        # Independently check the outer denominator against pre-inference annotation support.
        support=json.loads((root/'audio_annotation_preflight/annotation_support.json').read_text())[role]
        for report in rebuilt:
            c=report['sites']['all']['counts'];g=report['gaps']['all']['counts']
            assert c['normal']==support['normal'] and c['error']==support['substitution']+support['deletion']
            assert g['normal']==support['normal_gap'] and g['error']==support['insertion_gap']
        checks[role]='saved records recounted and pre-inference denominators preserved'
        reports[role]=dict(zip(('C0','weighted'),rebuilt))
        if role!='cal':uncertainty[role]={axis:paired(*outputs,*thresholds,axis) for axis in ('speaker','item')}
    gate=pilot_gate(reports['inner']['C0'],reports['inner']['weighted'])
    write(out/'report.json',dict(status='pilot_compared',pilot_gate=gate,roles=reports,paired=uncertainty,
        verification=checks,independent_final=False,all_five_folds_completed=False,
        conclusion='continue fixed folds' if gate['continue_folds'] else 'stop fixed weighting family under pilot compute rule'))
    write(out/'finished.json',dict(status='pilot_comparison_complete',report_sha256=file_sha(out/'report.json'),
        continue_folds=gate['continue_folds']))
    print(json.dumps(gate),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--arm',choices=['C0','weighted','compare'],required=True);a=ap.parse_args()
    if a.arm=='compare':compare(a.root)
    else:evaluate(a.root,a.arm)


if __name__=='__main__':main()

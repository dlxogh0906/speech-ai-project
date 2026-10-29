"""Diagnose learned-head fit versus sampling coverage, without new fitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import eval_sibilant_directional as service
from eval_sibilant_context_rank import model
from train_sibilant_cv_support import score


def directions(labels,features,probabilities,seen=None):
    byindex=collections.defaultdict(list)
    for r in labels:
        if r.get('feature_index') is not None:byindex[r['feature_index']].append(r)
    result={}
    for expected in ('ㅅ','ㅆ'):
        rows=[]
        for i,ls in byindex.items():
            if len(ls)!=1:continue
            r=ls[0]
            if r['expected']!=expected or features[i]['canonical']!=expected or r['alignment_ambiguous'] or r['label_kind'] not in ('correct','pair_error'):continue
            rows.append((i,int(r['label_kind']=='pair_error'),r.get('following')))
        groups={'all':rows}
        if seen is not None:
            groups.update(seen=[v for v in rows if v[0] in seen],unseen=[v for v in rows if v[0] not in seen])
        for vowel in ('ㅣ','ㅡ','ㅜ'):groups[vowel]=[v for v in rows if v[2]==vowel]
        result[expected]={}
        for k,rs in groups.items():
            ys=[v[1] for v in rs];ci=1 if expected=='ㅅ' else 0
            scores=[float(probabilities[v[0],ci]) for v in rs]
            result[expected][k]={'all':len(rs),'normal':ys.count(0),'pair_error':ys.count(1),
                'AUC':float(roc_auc_score(ys,scores)) if len(set(ys))==2 else None}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    data=dict(np.load(p['prefix']+'.npz'));features=service.read_jsonl(p['prefix']+'.features.jsonl');labels=service.read_jsonl(p['prefix']+'.labels.jsonl')
    plan=np.load(p['batch_plan']);seen=set(int(i) for i in plan.ravel())
    if len(data['x'])!=len(features) or max(seen)>=len(features):raise ValueError('batch-plan/input index mismatch')
    torch.set_num_threads(2);artifact=torch.load(p['artifact'],map_location='cpu');encoder,head=model(p,artifact)
    ps,av,seconds=score(encoder,head,{k:data[k] for k in ('x','lengths','relative')})
    np.savez(out/'train_scores.npz',probabilities=ps,available=av)
    groups={}
    for name,ix in [('all',np.arange(len(ps))),('seen',np.array(sorted(seen))),('unseen',np.array([i for i in range(len(ps)) if i not in seen]))]:
        ys=data['y'][ix];known=ys>=0;pred=ps[ix].argmax(-1);groups[name]={'all':len(ix),'known_classes':int(known.sum()),
            'correct_top1':int((pred[known]==ys[known]).sum()),'accuracy':float((pred[known]==ys[known]).mean()),
            'class_counts':{str(c):int((ys==c).sum()) for c in range(4)},'support_available':int(av[ix].sum())}
    calfeatures=service.read_jsonl(p['cal_prefix']+'.features.jsonl');callabels=service.read_jsonl(p['cal_prefix']+'.labels.jsonl');cal=np.load(p['cal_scores'])
    if len(calfeatures)!=len(cal['probabilities']):raise ValueError('cal input-score count differs')
    result={'scope':p['scope'],'groups':groups,'train_pair_directions':directions(labels,features,ps,seen),
            'cal_pair_directions':directions(callabels,calfeatures,cal['probabilities']),
            'seconds':seconds,'inference_batch':64,'new_training':False,'new_service_improvement_proven':False,
            'artifact_sha256':service.sha(p['artifact'])}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

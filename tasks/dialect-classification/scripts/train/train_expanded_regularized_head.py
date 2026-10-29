"""Small validation-only regularization sweep on the expanded frozen EAT cache."""
import csv
import json
import os
from pathlib import Path
import time
import warnings
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from train_standard_six_class import metrics, load_rows, write_json, LABELS, KO

ROOT=Path('/data/speech-ai/out/dialect')
RUN=ROOT/'expanded_six_class_20261013'
OUT=RUN/'regularized_head'
BASE=ROOT/'standard_six_class_20261013/eat_large'
REGIONAL=ROOT/'finetune_speakerid_20261012/eat_large/best_adapted.pt'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.time();torch.set_num_threads(2)
    regional=torch.load(REGIONAL,map_location='cpu')
    baseline=torch.load(BASE/'best_adapted.pt',map_location='cpu')
    for name,value in regional['encoder_delta'].items():
        if not torch.equal(value,baseline['encoder_delta'][name]):
            raise RuntimeError('Baseline test cache uses a different encoder')
    for split in ['valid','test']:
        assert (RUN/'manifests'/(split+'.csv')).read_bytes()==(BASE.parent/'manifests'/(split+'.csv')).read_bytes()
    rows={s:load_rows(RUN/'manifests'/(s+'.csv')) for s in ['train','valid','test']}
    cache=RUN/'eat_large/features'
    identity=json.loads((cache/'identity.json').read_text())
    indices=[identity['features'].index(n) for n in regional['feature_names']]
    x={}
    for split in ['train','valid']:
        if not np.load(cache/(split+'.ok.npy')).all():raise RuntimeError('Feature cache incomplete')
        arr=np.load(cache/(split+'.npy'),mmap_mode='r')
        x[split]=(np.asarray(arr[:,indices,:],np.float32).reshape(len(arr),-1)-regional['mu'])/regional['sd']
    y=np.array([int(r['label_id']) for r in rows['train']])
    counts=Counter(r['person_id'] for r in rows['train'])
    people={i:{r['person_id'] for r in rows['train'] if int(r['label_id'])==i} for i in range(6)}
    weights=np.array([1/counts[r['person_id']]/len(people[int(r['label_id'])]) for r in rows['train']]);weights/=weights.mean()
    history=[];best=None
    for c in [.003,.03,.3]:
        write_json(OUT/'progress.json',{'status':'running','C':c,'stage':'fit','pid':os.getpid()})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model=LogisticRegression(C=c,solver='lbfgs',max_iter=400,tol=1e-4,multi_class='multinomial').fit(x['train'],y,sample_weight=weights)
        logits=model.decision_function(x['valid']).astype(np.float32)
        valid=metrics(rows['valid'],logits)
        item={'C':c,'validation_top1':valid['top1'],'validation_macro_f1':valid['macro_f1'],
              'iterations':model.n_iter_.tolist(),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in caught),
              'elapsed_seconds':time.time()-started}
        history.append(item);print(json.dumps(item),flush=True);write_json(OUT/'history.json',history)
        if best is None or (valid['top1'],valid['macro_f1'])>(best['valid']['top1'],best['valid']['macro_f1']):
            best={'model':model,'C':c,'valid':valid,'logits':logits,'converged':item['converged']}
    model=best['model'];name='regularized_frozen:C='+str(best['C'])
    summary=json.loads((RUN/'manifests/summary.json').read_text())
    checkpoint={'labels':LABELS,'labels_ko':KO,'model':'eat_large_standard_six',
                'encoder_delta':regional['encoder_delta'],
                'head':{'weight':torch.from_numpy(model.coef_.astype(np.float32)),'bias':torch.from_numpy(model.intercept_.astype(np.float32))},
                'mu':regional['mu'],'sd':regional['sd'],'feature_names':regional['feature_names'],
                'data_summary':summary,'selected_candidate':name,'valid':best['valid']}
    torch.save(checkpoint,OUT/'best_adapted.pt')
    write_json(OUT/'selection.json',{'C':best['C'],'rule':'validation Top1 then macro F1','converged':best['converged']})
    # Test is read only after choosing C using validation.
    arr=np.load(ROOT/'seven_class_north_pilot_20261013/test_extract/features/test.npy',mmap_mode='r')
    test_x=(np.asarray(arr[:,indices,:],np.float32).reshape(len(arr),-1)-regional['mu'])/regional['sd']
    old=np.load(BASE/'test_predictions.npz')
    assert np.array_equal(old['row_id'],np.array([r['row_id'] for r in rows['test']]))
    reproduced=test_x@baseline['head']['weight'].numpy().T+baseline['head']['bias'].numpy()
    if np.mean(reproduced.argmax(1)==old['logits'].argmax(1))<.999:raise RuntimeError('Pinned test cache does not reproduce baseline')
    test_logits=model.decision_function(test_x).astype(np.float32);test=metrics(rows['test'],test_logits)
    for split,logits in [('valid',best['logits']),('test',test_logits)]:
        np.savez_compressed(OUT/(split+'_predictions.npz'),logits=logits,labels=np.array([int(r['label_id']) for r in rows[split]]),row_id=np.array([r['row_id'] for r in rows[split]]))
    report={'model':'regularized expanded frozen EAT head','selected_candidate':name,'best_frozen_candidate':name,
            'valid':best['valid'],'test':test,'data_summary':summary,'history':history,'converged':best['converged'],
            'selection':'validation Top1 then macro F1','elapsed_seconds':time.time()-started}
    write_json(OUT/'report.json',report)
    write_json(OUT/'progress.json',{'status':'complete','test_top1':test['top1'],'valid_top1':best['valid']['top1'],'selected_C':best['C']})
    print(json.dumps({'complete':True,'test_top1':test['top1'],'valid_top1':best['valid']['top1']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'progress.json',{'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise

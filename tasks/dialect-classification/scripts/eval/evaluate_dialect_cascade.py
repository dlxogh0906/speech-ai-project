"""Evaluate a fixed EAT standard gate followed by the trained XLS-R regional model.

No gate threshold is tuned on test. Cached logits also permit applying the same
fixed routing rule to the validation-selected expanded EAT model later.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'tasks/dialect-classification/scripts/train'))
import train_dialect_comparison as enc
from train_standard_six_class import metrics,load_rows,write_json
ROOT=Path('/data/speech-ai/out/dialect');RUN=ROOT/'expanded_six_class_20261013';OUT=RUN/'xlsr_cascade'
BASE=ROOT/'standard_six_class_20261013/eat_large'
CKPT=ROOT/'finetune_speakerid_20261012/xlsr2b/best_adapted.pt'


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.30,device=0)
    device=torch.device('cuda:0');started=time.time()
    def progress(stage,**extra):
        value=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-started,**extra)
        value.setdefault('status','running')
        write_json(OUT/'progress.json',value);print(json.dumps(value),flush=True)
    progress('load_xlsr')
    state=torch.load(CKPT,map_location='cpu')
    summary=json.loads((RUN/'manifests/summary.json').read_text())
    if state['data_summary']['manifest_sha256']!=summary['original_regional_manifest_sha256']:
        raise RuntimeError('XLSR was trained using another speaker split')
    encoder=enc.Encoder('xlsr2b',device)
    from fairseq.modules.checkpoint_activations import unwrap_checkpoint
    unwrap_checkpoint(encoder.model)
    params=dict(encoder.model.named_parameters())
    with torch.no_grad():
        for name,values in state['encoder_delta'].items():params[name].copy_(values.to(device))
    encoder.model.eval();indices=[encoder.names.index(n) for n in state['feature_names']]
    weight=state['head']['weight'].to(device);bias=state['head']['bias'].to(device)
    mu=torch.as_tensor(state['mu'],device=device);sd=torch.as_tensor(state['sd'],device=device)
    rows={s:load_rows(RUN/'manifests'/(s+'.csv')) for s in ['valid','test']}
    result={}
    # The recipe is fixed in advance: Standard when EAT argmax is Standard; otherwise XLSR region.
    write_json(OUT/'protocol.json',{'routing':'EAT argmax Standard -> Standard; otherwise XLSR regional argmax',
               'gate_threshold_tuned':False,'xlsr_checkpoint':str(CKPT),
               'xlsr_checkpoint_sha256':hashlib.sha256(CKPT.read_bytes()).hexdigest(),
               'evaluation':'existing 6-class held-out set; no independent North recordings'})
    for split in ['valid','test']:
        saved=OUT/('xlsr_'+split+'_logits.npz')
        if saved.exists():
            loaded=np.load(saved);logits=loaded['logits']
            assert np.array_equal(loaded['row_id'],np.array([r['row_id'] for r in rows[split]]))
        else:
            order=sorted(range(len(rows[split])),key=lambda i:float(rows[split][i]['duration_sec']))
            loader=DataLoader(Subset(enc.PreparedClips(rows[split],'xlsr2b'),order),batch_size=16,num_workers=2,collate_fn=enc.Collator('xlsr2b'),pin_memory=True)
            logits=np.full((len(order),5),np.nan,np.float32)
            with torch.no_grad():
                for batch,(idx,values,lengths,errors) in enumerate(loader):
                    if errors:raise RuntimeError('Audio errors: '+str(errors[:2]))
                    features=encoder(values,lengths)[:,indices,:].reshape(len(values),-1).half().float()
                    output=((features-mu)/sd)@weight.T+bias
                    if not torch.isfinite(output).all():raise RuntimeError('Invalid XLSR logits')
                    logits[idx]=output.cpu().numpy()
                    if batch%50==0:progress('infer_'+split,processed=min((batch+1)*16,len(order)),total=len(order))
            np.savez_compressed(saved,logits=logits,row_id=np.array([r['row_id'] for r in rows[split]]))
        eat=np.load(BASE/(split+'_predictions.npz'))
        assert np.array_equal(eat['row_id'],np.array([r['row_id'] for r in rows[split]]))
        pred=np.where(eat['logits'].argmax(1)==5,5,logits.argmax(1))
        output=np.eye(6,dtype=np.float32)[pred]
        result[split]=metrics(rows[split],output)
        np.savez_compressed(OUT/(split+'_predictions.npz'),predictions=pred,labels=eat['labels'],row_id=eat['row_id'])
        if split=='valid':write_json(OUT/'validation_result.json',result[split])
    result.update(method='EAT Standard gate + XLSR 2B regional classifier',elapsed_seconds=time.time()-started,
                  two_encoders_required=True,gate_training='No new parameters; fixed routing rule')
    write_json(OUT/'report.json',result)
    progress('complete',status='complete',valid_top1=result['valid']['top1'],test_top1=result['test']['top1'])

if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'progress.json',{'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise

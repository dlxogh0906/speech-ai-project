"""Check captured layer identities against native hidden-state APIs on real audio."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/scripts/eval'),
             str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import gc
import json
import numpy as np
import torch
from diagnose_h0_representations import Backbone, LayerHooks, read_wav, write_json, file_sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    root=Path(a.root);p=json.loads((root/'protocol.json').read_text());parent=json.loads(Path(p['parent_protocol']).read_text())
    records=[json.loads(s) for s in (root/'utterances.jsonl').open()]
    samples=[next(r for r in records if r['role']==role and (item is None or r['item']==item))
             for role,item in [('train',None),('calibration',None),('selection','RW054'),('selection','RS012')]]
    report=[]
    for kind in ('w2vbert','xlsr300m'):
        ck=torch.load(str(Path(p['experiments'])/(kind+'_stage2')/'step_2000.pt'),map_location='cpu')
        model=Backbone(kind,parent['backbones'][kind]['path'],max(ck['vocab'].values())+1,torch.device('cuda'))
        model.load_state_dict(ck['model'],strict=True);del ck
        model.eval();model.requires_grad_(False);hooks=LayerHooks(model)
        for r in samples:
            x=read_wav(r['path'],20.)
            with torch.inference_mode(),torch.cuda.amp.autocast():
                logits,lengths=model([x]);n=int(lengths[0])
                captured={k:v[:n].clone() for k,v in hooks.values.items()}
                if kind=='w2vbert':
                    enc=model.fe([x],sampling_rate=16000,return_tensors='pt',padding=True)
                    native=model.net(input_features=enc['input_features'].cuda(),attention_mask=enc['attention_mask'].cuda(),output_hidden_states=True)
                    reference={k:native.hidden_states[k][0,:n] for k in (6,12,24)}
                    direct_logits=native.logits[0,:n]
                else:
                    src=torch.nn.functional.layer_norm(torch.from_numpy(x),(len(x),))[None].cuda()
                    native=model.net.extract_features(src,padding_mask=torch.zeros_like(src,dtype=torch.bool),mask=False)
                    reference={6:native['layer_results'][5][0][:n,0],12:native['layer_results'][11][0][:n,0],24:native['x'][0,:n]}
                    direct_logits=model.head(native['x'])[0,:n]
                differences={str(k):float((captured[k]-reference[k]).abs().max()) for k in reference}
                logitdiff=float((direct_logits-logits[0,:n]).abs().max())
                if any(v!=0 for v in differences.values()) or logitdiff!=0:raise ValueError('layer identity mismatch')
            report.append(dict(backbone=kind,uid=r['uid'],layer_max_abs_difference=differences,logit_max_abs_difference=logitdiff))
        for h in hooks.handles:h.remove()
        del model,hooks,native,captured,reference,logits,direct_logits;gc.collect();torch.cuda.empty_cache()
    write_json(root/'native_layer_verification.json',dict(status='passed',utterances=report,encoder_updates=0,source_sha256=file_sha(__file__)))


if __name__=='__main__':main()

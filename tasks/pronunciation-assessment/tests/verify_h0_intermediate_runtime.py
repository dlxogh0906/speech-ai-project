"""Real-audio native-forward parity before any P3 model training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT: raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
              str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import gc
import hashlib
import json
import torch
from train_h0_intermediate import P3Model,bind,replay_prefix
from train_h0_backbone_control import Backbone,file_sha,ctc_loss
from train_w2vbert_ctc import read_split,read_wav
from mdd.h0_intermediate import restore_rng,rng_state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); p=bind(a.protocol); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    rows,labels=read_split(p['manifest'],'train'); batches,_,_,_=replay_prefix(rows,p)
    ids=next(batches)[:4]; xs=[read_wav(rows[i][0],20.) for i in ids]
    ck=torch.load(p['parent_checkpoint'],map_location='cpu'); vocab=ck['vocab']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results={}; reference_outputs={}
    for arm in ('B0','B1','B2','B3'):
        m=P3Model(p['source'],vocab,device,arm,ck['model'])
        results[arm]={}
        for training in (False,True):
            m.train(training); restore_rng(ck['rng']); state=rng_state()
            with torch.no_grad(),torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                native,nlengths=Backbone.forward(m,xs)
                native_rng=rng_state(); restore_rng(state)
                final,lengths,aux=m(xs,auxiliary=True)
                branch_rng=rng_state()
            assert torch.equal(native,final), (arm,training,float((native-final).abs().max()))
            assert torch.equal(lengths,nlengths)
            assert torch.equal(native_rng['torch'],branch_rng['torch'])
            assert all(torch.equal(x,y) for x,y in zip(native_rng['cuda'],branch_rng['cuda']))
            if training in reference_outputs:
                assert torch.equal(reference_outputs[training],final.cpu())
            else: reference_outputs[training]=final.cpu().clone()
            results[arm]['train' if training else 'eval']=dict(native_final_max_error=0.,
                same_rng_after_forward=True,shape=list(final.shape),
                sha256=hashlib.sha256(final.float().cpu().numpy().tobytes()).hexdigest(),
                auxiliary_finite=bool(torch.isfinite(aux).all()) if aux is not None else None)
        # A fresh module reconstruction exactly restores all new learned tensors.
        payload={k:v.detach().cpu().clone() for k,v in m.added.state_dict().items()}
        m.added.load_state_dict(payload,strict=True)
        del m,native,final,aux
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    (out/'finished.json').write_text(json.dumps(dict(protocol_sha256=file_sha(a.protocol),
        utterances=4,continuation_row_ids=ids,results=results,
        claim='initial native parity and matched RNG, not performance or optimizer training'),indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__': main()

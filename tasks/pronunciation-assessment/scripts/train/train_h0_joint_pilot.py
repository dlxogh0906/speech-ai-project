"""Strict fold lineage and one fixed C0/lambda=.3 continuation comparison."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import gc
import hashlib
import json
import random
import shutil
import time
import numpy as np
import torch
from train_h0_backbone_control import Backbone, file_sha, stream
from train_w2vbert_ctc import read_split, read_wav, load_vocab, tri_stage
from mdd.h0_joint_weighting import weighted_ctc_loss
from audit_h0_p5_readiness import write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--phase', choices=['stage1','common','C0','weighted'], required=True)
    ap.add_argument('--solution', type=Path, required=True)
    a = ap.parse_args(); p = json.loads((a.root/'protocol.json').read_text())
    verify = json.loads((a.root/'verification/report.json').read_text())
    audio = json.loads((a.root/'audio_annotation_preflight/report.json').read_text())
    runtime = json.loads((a.root/'runtime_check.json').read_text())
    if not verify['pretraining_treatment_passed'] or audio['status'] != 'passed' or runtime['status'] != 'passed':
        raise ValueError('pretraining checks have not passed')
    # Runtime integration receipt must bind trainer, loss, and evaluation code.
    for path, digest in runtime['source_bindings'].items():
        assert file_sha(path) == digest, path
    for path, digest in runtime['input_bindings'].items():
        assert file_sha(path) == digest, path
    if str(a.root.resolve()).startswith('/dev/shm/'):
        raise ValueError('persistent storage required')
    remaining = {'stage1':15, 'common':11, 'C0':8, 'weighted':5}[a.phase]*1024**3
    if shutil.disk_usage(a.root).free < remaining:
        raise ValueError('insufficient persistent capacity for remaining checkpoints/evaluation')
    out = a.root/'training'/a.phase; out.mkdir(parents=True, exist_ok=False)
    source = p['public_SSL']
    for path, digest in source['sha256'].items():
        assert file_sha(path) == digest, path
    native = a.phase == 'stage1'; manifest = a.root/'manifests'/('stage1' if native else 'stage2')
    bindings = json.loads((a.root/'manifests/report.json').read_text())['bindings']
    for name in ('train.tsv','train.ltr','dict.ltr.txt'):
        path = manifest/name; assert file_sha(path) == bindings[str(path)]['sha256']
    rows, labels = read_split(str(manifest), 'train'); vocab = load_vocab(str(manifest/'dict.ltr.txt'))
    seed = p['training']['seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise ValueError('full training requires the declared GPU path')
    model = Backbone('w2vbert', source['path'], max(vocab.values())+1, device)
    start, stop = (0,10000) if native else ((0,2000) if a.phase=='common' else (2000,4000))
    parent = None; parent_hash = None
    if not native:
        phase = 'stage1' if a.phase == 'common' else 'common'
        parent = a.root/'training'/phase/('step_%d.pt'%(10000 if phase=='stage1' else 2000))
        parent_hash = file_sha(parent); ck = torch.load(parent, map_location='cpu')
        assert ck['protocol_sha256'] == file_sha(a.root/'protocol.json') and ck['phase'] == phase
        model.load_state_dict(ck['model'], strict=True)
    weights = np.ones(len(rows))
    if a.phase == 'weighted':
        for path,digest in verify['hashes'].items(): assert file_sha(path)==digest,path
        weights = np.load(a.solution/'manifest_weights.npy')
    if len(weights) != len(rows):
        raise ValueError('weights not aligned to fold manifest')
    # Both continuation arms intentionally start with a fresh AdamW state.
    config = p['training']['stage1' if native else 'stage2']
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9,.98), eps=1e-8, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    if start:
        scaler.load_state_dict(ck['scaler']); state = ck['rng']
        torch.set_rng_state(state['torch']); np.random.set_state(state['numpy']); random.setstate(state['random'])
        torch.cuda.set_rng_state_all(state['cuda'])
    if not native:
        del ck
    batches = stream(rows, p['training']['batch_samples'], seed)
    accum = p['training']['accum']; order = hashlib.sha256(); draws = samples = 0
    for micro in range(start*accum):
        ids = next(batches); order.update((' '.join(map(str,ids))+'\n').encode())
        draws += len(ids); samples += sum(rows[i][1] for i in ids)
    expected = audio['planned_exposures']['stage1' if native else 'stage2']['receipts']
    if start:
        assert dict(row_draws=draws,audio_samples=samples,batch_order_sha256=order.hexdigest()) == expected[0]
    write(out/'execution.json', dict(protocol_sha256=file_sha(a.root/'protocol.json'),
        parent=str(parent), parent_sha256=parent_hash, source_sha256=file_sha(__file__),
        phase=a.phase, weights_sha256=file_sha(a.solution/'manifest_weights.npy') if a.phase=='weighted' else None,
        start_step=start, stop_step=stop, physical_manifest=str(manifest), optimizer='fresh AdamW',
        saved_optimizer=False, weight_arithmetic='FP32 loss weights; independent TRAIN bounds use float64 with 2e-6 tolerance'))
    begin = time.time(); model.train()
    with (out/'updates.jsonl').open('w') as log:
        for step in range(start+1, stop+1):
            model.encoder().requires_grad_(not (native and step<=2000))
            opt.zero_grad(set_to_none=True); losses=[]
            for _ in range(accum):
                ids=next(batches); order.update((' '.join(map(str,ids))+'\n').encode())
                xs=[read_wav(rows[i][0],20.) for i in ids]
                if any(len(x)!=rows[i][1] for x,i in zip(xs,ids)):
                    raise ValueError('changed or truncated audio')
                ys=[[vocab[t] for t in labels[i]] for i in ids]
                draws+=len(ids); samples+=sum(len(x) for x in xs)
                with torch.cuda.amp.autocast():
                    logits,lengths=model(xs); loss=weighted_ctc_loss(logits,lengths,ys,weights[ids])/accum
                scaler.scale(loss).backward(); losses.append(float(loss.detach())*accum)
            lr=tri_stage(step-1,config['schedule_steps'],config['lr'])
            for group in opt.param_groups:group['lr']=lr
            scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradient; no silent update skip')
            scaler.step(opt); scaler.update()
            row=dict(step=step,loss=float(np.mean(losses)),lr=lr,gradient_norm=float(norm),row_draws=draws,
                audio_samples=samples,batch_order_sha256=order.hexdigest(),elapsed_seconds=time.time()-begin)
            log.write(json.dumps(row)+'\n'); log.flush()
            if step%100==0:print(json.dumps(row),flush=True)
        receipt={k:row[k] for k in ('row_draws','audio_samples','batch_order_sha256')}
        assert receipt==expected[0 if a.phase=='common' else -1]
        payload=dict(protocol_sha256=file_sha(a.root/'protocol.json'),backbone='w2vbert',phase=a.phase,
            stage='stage1' if native else 'stage2',step=stop,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            vocab=vocab,exposure=row,optimizer=None,scaler=scaler.state_dict(),
            rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),random=random.getstate(),cuda=torch.cuda.get_rng_state_all()))
        destination=out/('step_%d.pt'%stop); torch.save(payload,str(destination)+'.partial')
        Path(str(destination)+'.partial').rename(destination); del payload; gc.collect()
    write(out/'finished.json',dict(status='trained_not_evaluated',phase=a.phase,updates=stop-start,
        final_exposure=row,checkpoint_sha256=file_sha(destination),parent_sha256=parent_hash,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()

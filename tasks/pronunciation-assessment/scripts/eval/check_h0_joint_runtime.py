"""Exercise the fixed fold loss, full batch and evaluation path before training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import gzip
import json
import random
import numpy as np
import torch
from train_h0_backbone_control import Backbone,file_sha,stream,ctc_loss
from train_w2vbert_ctc import read_split,read_wav,load_vocab
from mdd.h0_joint_weighting import weighted_ctc_loss
from mdd.h0_joint_evaluation import score_record,site_counts,aggregate
from eval_h0_rp_stress import forward
from audit_h0_p5_readiness import write


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--solution',type=Path,required=True);a=ap.parse_args()
    assert not (a.root/'runtime_check.json').exists()
    p=json.loads((a.root/'protocol.json').read_text())
    verification=json.loads((a.root/'verification/report.json').read_text())
    assert verification['pretraining_treatment_passed']
    assert json.loads((a.root/'audio_annotation_preflight/report.json').read_text())['status']=='passed'
    manifest=a.root/'manifests/stage2';vocab=load_vocab(str(manifest/'dict.ltr.txt'))
    rows,labels=read_split(str(manifest),'train');batches=stream(rows,1600000,42)
    for _ in range(8000):next(batches)
    ids=next(batches);xs=[read_wav(rows[i][0],20.) for i in ids];ys=[[vocab[t] for t in labels[i]] for i in ids]
    weights=np.load(a.solution/'manifest_weights.npy')[ids]
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    device=torch.device('cuda');model=Backbone('w2vbert',p['public_SSL']['path'],max(vocab.values())+1,device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.98),eps=1e-8,weight_decay=0)
    scaler=torch.cuda.amp.GradScaler(init_scale=128);model.train()
    updates=[]
    watched=model.net.wav2vec2_bert.feature_projection.projection.weight
    for step in (0,1):
        model.encoder().requires_grad_(step==1)
        before=watched.detach().clone();head_before=model.net.lm_head.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
        with torch.cuda.amp.autocast():logits,lengths=model(xs)
        # Exact same dropout/LayerDrop stream and input must give the same logits.
        if step==0:
            after_rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
            with torch.no_grad(),torch.cuda.amp.autocast():repeat,repeat_lengths=model(xs)
            torch.testing.assert_close(logits,repeat,rtol=0,atol=0)
            assert torch.equal(lengths,repeat_lengths)
            assert torch.equal(torch.get_rng_state(),after_rng[0])
            assert all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),after_rng[1]))
            del repeat
        ordinary=ctc_loss(logits,lengths,ys)
        ones=weighted_ctc_loss(logits,lengths,ys,np.ones(len(ids)))
        torch.testing.assert_close(ordinary,ones,rtol=0,atol=0)
        loss=weighted_ctc_loss(logits,lengths,ys,weights if step else np.ones(len(ids)))
        scaler.scale(loss).backward();scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(norm)
        scaler.step(opt);scaler.update()
        head_changed=not torch.equal(head_before,model.net.lm_head.weight)
        encoder_changed=not torch.equal(before,watched)
        assert head_changed and encoder_changed==(step==1)
        updates.append(dict(encoder_unfrozen=bool(step),loss=float(loss),gradient_norm=float(norm),
            encoder_changed=encoder_changed,head_changed=head_changed,all_one_loss_difference=float(ones-ordinary)))
        del logits,ordinary,ones,loss,before,head_before
    # Use four existing TRAIN recordings for scored-path smoke; no validation metrics used.
    from mdd.h0_partial_reference import partial_tokens,partial_mapping,classify_truth
    from mdd.h0_partial_scoring import mapping_indices,annotation_mask,gap_value
    # Real heldout audio is not needed to validate inference and partial scoring.
    model.eval();checks=[]
    for i in ids[:4]:
        ref=labels[i];owners=[dict(start=0.,end=rows[i][1]/16000) for _ in ref]
        truth=[dict(q=q,expected=t,actual=t,status='normal') for q,t in enumerate(ref)]
        r=dict(uid=Path(rows[i][0]).stem,speaker='runtime_only',item='runtime_only',task='runtime_only',
            language='runtime_only',role='runtime_only',reference=ref,actual=ref,reference_owners=owners,
            actual_owners=owners,sites=truth,annotation_mask=[None]*len(ref),truth_gaps=[None]+[[]]*(len(ref)-1)+[None])
        lp=forward(model,read_wav(rows[i][0],20.));one=score_record(r,lp,vocab)
        c=site_counts(one['sites'],None);assert c['normal']==len(ref)
        result=aggregate([one],None,vocab);assert result['sites']['all']['counts']['normal']==len(ref)
        checks.append(dict(uid=r['uid'],frames=len(lp),fixed_denominator=len(ref)))
    paths=[ROOT/'tasks/pronunciation-assessment'/name for name in (
        'scripts/train/train_h0_joint_pilot.py','scripts/train/train_h0_backbone_control.py',
        'scripts/train/train_w2vbert_ctc.py','scripts/eval/eval_h0_joint_pilot.py',
        'scripts/eval/check_h0_joint_runtime.py','scripts/eval/eval_h0_rp_stress.py',
        'src/mdd/h0_joint_weighting.py','src/mdd/h0_joint_evaluation.py','src/mdd/h0_partial_scoring.py',
        'src/mdd/h0_partial_reference.py','src/mdd/gop_feats_v2.py','src/mdd/h0_phone_audit.py')]
    inputs=[a.root/'protocol.json',a.root/'verification/report.json',a.root/'manifests/report.json',
            a.root/'audio_annotation_preflight/report.json',a.solution/'manifest_weights.npy']
    inputs += [a.root/'audio_annotation_preflight'/(role+'_annotations.jsonl.gz') for role in ('cal','inner','outer')]
    write(a.root/'runtime_check.json',dict(status='passed',updates=updates,
        smoke_only_optimizer_updates=2,main_training_optimizer_updates=0,public_model_discarded=True,
        synthetic_truth_for_path_check_only=True,train_audio_checks=checks,batch_rows=len(ids),
        batch_audio_samples=sum(len(x) for x in xs),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        source_bindings={str(path):file_sha(path) for path in paths},input_bindings={str(path):file_sha(path) for path in inputs}))
    print('real full-batch runtime check passed; smoke model discarded',flush=True)


if __name__=='__main__':main()

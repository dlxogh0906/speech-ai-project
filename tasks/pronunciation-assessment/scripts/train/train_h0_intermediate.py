"""P3: matched optimizer-reset continuation of H0 w2v-BERT 2k.

Uses the original continuous minibatch stream, global learning-rate position,
parent AMP and RNG states. All arms reset AdamW because the parent has no Adam
state. New heads do not draw randomness in forward; common dropout/LayerDrop
therefore see the same RNG stream in every arm. Existing assets are read-only.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import gc
import hashlib
import json
import shutil
import time
import numpy as np
import torch
from train_h0_backbone_control import Backbone, ctc_loss, file_sha, stream
from train_w2vbert_ctc import read_split, read_wav, load_vocab, tri_stage
from mdd.h0_intermediate import IntermediateHeads, rng_state, restore_rng


class P3Model(Backbone):
    def __init__(self, source, vocab, device, condition, parent_model=None):
        super().__init__('w2vbert', source, max(vocab.values()) + 1, device)
        if parent_model is not None:
            self.load_state_dict(parent_model, strict=True)
        self.added = IntermediateHeads(self.net.lm_head, condition).to(device)
        self.condition = condition

    def forward(self, xs, auxiliary=False):
        enc = self.fe(xs, sampling_rate=16000, return_tensors='pt', padding=True)
        am = enc['attention_mask'].to(self.device)
        outputs = self.net.wav2vec2_bert(input_features=enc['input_features'].to(self.device),
            attention_mask=am, output_hidden_states=True, return_dict=True)
        # hidden_states[12] is the output after 12 layer slots, including LayerDrop.
        fused, aux = self.added(outputs.hidden_states[12], outputs.last_hidden_state, auxiliary)
        logits = self.net.lm_head(self.net.dropout(fused))
        lengths = self.net._get_feat_extract_output_lengths(am.sum(-1)).long()
        return (logits, lengths, aux) if auxiliary else (logits, lengths)


def bind(protocol_path):
    p = json.loads(Path(protocol_path).read_text())
    for path, digest in p['bindings'].items():
        if file_sha(path) != digest:
            raise ValueError('changed input/code: ' + path)
    return p


def replay_prefix(rows, p):
    batches = stream(rows, p['batch_samples'], p['seed'])
    h, draws, samples = hashlib.sha256(), 0, 0
    for _ in range(p['start_step'] * p['accum']):
        ids = next(batches)
        h.update((' '.join(map(str, ids)) + '\n').encode())
        draws += len(ids); samples += sum(rows[i][1] for i in ids)
    observed = dict(row_draws=draws, audio_samples=samples, batch_order_sha256=h.hexdigest())
    if observed != p['parent_exposure']:
        raise ValueError(('parent stream reconstruction differs', observed))
    return batches, h, draws, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--condition', choices=['B0', 'B1', 'B2', 'B3'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true', help='two real continuation updates, no weights saved')
    a = ap.parse_args()
    p = bind(a.protocol)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    if not a.smoke and shutil.disk_usage(out).free < 6 * 1024**3:
        raise RuntimeError('need 6 GiB for this arm and evaluation; no existing assets removed')
    rows, labels = read_split(p['manifest'], 'train')
    vocab = load_vocab(str(Path(p['manifest']) / 'dict.ltr.txt'))
    batches, h, draws, samples = replay_prefix(rows, p)
    ck = torch.load(p['parent_checkpoint'], map_location='cpu')
    assert ck['optimizer'] is None and ck['step'] == p['start_step'] and ck['vocab'] == vocab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = P3Model(p['source'], vocab, device, a.condition, ck['model'])
    opt = torch.optim.AdamW([
        dict(params=model.net.parameters(), lr=p['lr'], lr_factor=1.),
        dict(params=[v for v in model.added.parameters() if v.requires_grad],
             lr=p['lr'] * p['new_parameter_lr_factor'], lr_factor=p['new_parameter_lr_factor'])],
        betas=(.9, .98), eps=1e-8, weight_decay=0.)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    scaler.load_state_dict(ck['scaler'])
    restore_rng(ck['rng']); del ck
    model.train()
    stop = p['start_step'] + 2 if a.smoke else p['stop_step']
    start = time.time()
    initial = {name: value.detach().cpu().clone() for name, value in model.added.named_parameters()}
    # The first registered parameter is masked_spec_embed, unused with masking
    # disabled. Check the first matrix on the actual acoustic path instead.
    first_base_name, first_base = next((n,v) for n,v in model.net.wav2vec2_bert.named_parameters() if v.ndim>=2)
    base_before = first_base.detach().clone() if a.smoke else None
    print(json.dumps(dict(event='start', condition=a.condition, start_step=p['start_step'],
                         stop_step=stop, optimizer_reset=True, trainable_parameters=
                         sum(v.numel() for v in model.parameters() if v.requires_grad))), flush=True)
    with (out / 'updates.jsonl').open('x') as log:
        for step in range(p['start_step'] + 1, stop + 1):
            opt.zero_grad(set_to_none=True)
            losses, finals, auxiliaries = [], [], []
            for _ in range(p['accum']):
                ids = next(batches)
                h.update((' '.join(map(str, ids)) + '\n').encode())
                xs = [read_wav(rows[i][0], 20.) for i in ids]
                if any(len(x) != rows[i][1] for x, i in zip(xs, ids)):
                    raise ValueError('audio length changed')
                ys = [[vocab[t] for t in labels[i]] for i in ids]
                draws += len(ids); samples += sum(map(len, xs))
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    logits, lengths, auxiliary = model(xs, auxiliary=True)
                    final = ctc_loss(logits, lengths, ys)
                    aux = ctc_loss(auxiliary, lengths, ys) if auxiliary is not None else final.new_zeros(())
                    loss = final + p['aux_lambda'] * aux
                scaler.scale(loss / p['accum']).backward()
                losses.append(float(loss.detach())); finals.append(float(final.detach()))
                auxiliaries.append(float(aux.detach()))
            lr = tri_stage(step - 1, p['schedule_steps'], p['lr'])
            for group in opt.param_groups:
                group['lr'] = lr * group['lr_factor']
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite gradient; cannot silently skip an update')
            scaler.step(opt); scaler.update()
            record = dict(condition=a.condition, step=step, continuation_updates=step-p['start_step'],
                loss=float(np.mean(losses)), final_loss=float(np.mean(finals)),
                auxiliary_loss=float(np.mean(auxiliaries)), lr=lr, gradient_norm=float(norm),
                gate=float(model.added.gate.tanh().detach()), row_draws=draws, audio_samples=samples,
                batch_order_sha256=h.hexdigest(), elapsed_seconds=time.time()-start)
            log.write(json.dumps(record)+'\n'); log.flush()
            (out/'live_status.json').write_text(json.dumps(record, indent=2))
            if step % 100 == 0 or a.smoke:
                print(json.dumps(record), flush=True)
            if not a.smoke and step in p['save_steps']:
                payload = dict(protocol_sha256=file_sha(a.protocol), condition=a.condition,
                    step=step, vocab=vocab, exposure=record, model={k:v.detach().cpu()
                    for k,v in model.state_dict().items()}, rng=rng_state(), scaler=scaler.state_dict(),
                    optimizer=None, optimizer_retention='model_only_snapshots_as_preregistered')
                dest = out / ('step_%d.pt' % step)
                torch.save(payload, str(dest)+'.partial')
                Path(str(dest)+'.partial').rename(dest)
                del payload; gc.collect()
    changes = {name:float((value.detach().cpu()-initial[name]).abs().max())
               for name, value in model.added.named_parameters()}
    if not a.smoke:
        observed = {k:record[k] for k in p['expected_final_exposure']}
        if observed != p['expected_final_exposure']:
            raise ValueError(('full continuation exposure mismatch', observed))
    report = dict(condition=a.condition, smoke=a.smoke, updates=stop-p['start_step'],
        protocol_sha256=file_sha(a.protocol), final_exposure=record, added_parameter_changes=changes,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
        claim='execution_check_not_performance' if a.smoke else 'trained_not_evaluated')
    if a.smoke:
        report['encoder_checked_parameter'] = first_base_name
        report['encoder_first_tensor_max_change'] = float((first_base.detach()-base_before).abs().max())
        assert report['encoder_first_tensor_max_change'] > 0
        if model.added.use_aux:
            assert changes['aux.weight'] > 0
        if model.added.use_fusion:
            assert changes['gate'] > 0
    (out/'finished.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

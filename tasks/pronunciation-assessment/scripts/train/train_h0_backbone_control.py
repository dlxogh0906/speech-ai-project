"""Common CTC training loop for H0 backbone screening.

Both backbones start from public SSL weights for Stage 1, then from their own
matched Stage 1 for Stage 2. Does not load M2/M3 or modify deployment. --smoke
checks head-only and encoder backward on two small batches; it is NOT a run of
the preregistered training schedule and saves no checkpoint.
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
import random
import shutil
import time
import numpy as np
import torch
from train_w2vbert_ctc import batches_by_frames, load_vocab, read_split, read_wav, tri_stage


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Backbone(torch.nn.Module):
    def __init__(self, kind, source, n_out, device):
        super().__init__()
        self.kind, self.device = kind, device
        if kind == 'w2vbert':
            from transformers import Wav2Vec2BertForCTC, SeamlessM4TFeatureExtractor
            self.net = Wav2Vec2BertForCTC.from_pretrained(source, local_files_only=True,
                vocab_size=n_out, pad_token_id=0, ctc_loss_reduction='mean', ctc_zero_infinity=False,
                apply_spec_augment=False, layerdrop=0.1, activation_dropout=0.1)
            self.fe = SeamlessM4TFeatureExtractor.from_pretrained(source, local_files_only=True)
        elif kind == 'xlsr300m':
            from shared.fairseq_guard import ensure_fairseq
            ensure_fairseq()
            from fairseq import checkpoint_utils
            models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task([source], strict=True)
            self.net = models[0]
            assert cfg.task.normalize and cfg.task.sample_rate == 16000
            assert len(self.net.encoder.layers) == 24 and cfg.model.encoder_embed_dim == 1024
            self.net.remove_pretraining_modules()
            self.net.feature_grad_mult = 1.0
            self.net.encoder.layerdrop = 0.1
            for layer in self.net.encoder.layers:
                layer.dropout2.p = 0.1
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(1024, n_out)
            # Same declared CTC-head initialization distribution as the HF model.
            torch.nn.init.normal_(self.head.weight, mean=0, std=0.02)
            torch.nn.init.zeros_(self.head.bias)
        else:
            raise ValueError(kind)
        self.to(device)

    def encoder(self):
        return self.net.wav2vec2_bert if self.kind == 'w2vbert' else self.net

    def forward(self, xs):
        if self.kind == 'w2vbert':
            enc = self.fe(xs, sampling_rate=16000, return_tensors='pt', padding=True)
            am = enc['attention_mask'].to(self.device)
            logits = self.net(input_features=enc['input_features'].to(self.device), attention_mask=am).logits
            lengths = self.net._get_feat_extract_output_lengths(am.sum(-1)).long()
        else:
            values = [torch.nn.functional.layer_norm(torch.from_numpy(x), (len(x),)) for x in xs]
            src = torch.nn.utils.rnn.pad_sequence(values, batch_first=True).to(self.device)
            raw_lengths = torch.tensor([len(x) for x in xs], device=self.device)
            mask = torch.arange(src.shape[1], device=self.device)[None] >= raw_lengths[:, None]
            out = self.net.extract_features(src, padding_mask=mask, mask=False)
            logits = self.head(self.dropout(out['x']))
            lengths = self.net._get_feat_extract_output_lengths(raw_lengths).long()
        return logits, lengths


def ctc_loss(logits, lengths, targets):
    sizes = torch.tensor([len(y) for y in targets], device=logits.device)
    flat = torch.tensor([t for y in targets for t in y], device=logits.device, dtype=torch.long)
    if bool((sizes == 0).any()):
        raise ValueError('empty CTC target')
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        loss = torch.nn.functional.ctc_loss(lp, flat, lengths, sizes, blank=0,
            reduction='none', zero_infinity=False)
    if not torch.isfinite(loss).all():
        raise ValueError('nonfinite CTC; do not silently drop different rows for different backbones')
    return (loss / sizes).mean()


def stream(rows, budget, seed):
    epoch = 0
    while True:
        for batch in batches_by_frames(rows, budget, shuffle_seed=seed + epoch):
            yield batch
        epoch += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--backbone', choices=['w2vbert', 'xlsr300m'], required=True)
    ap.add_argument('--stage', choices=['stage1', 'stage2'], required=True)
    ap.add_argument('--stage1-checkpoint')
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-full-batch', action='store_true',
        help='with --smoke, validate the actual 1.6M x 4 batch budget')
    a = ap.parse_args()
    if a.smoke_full_batch and not a.smoke:
        ap.error('--smoke-full-batch requires --smoke')
    protocol = json.loads(Path(a.protocol).read_text())
    config = protocol[a.stage]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    # Verify corpus and source bindings before allocating a model or writing weights.
    bindings = json.loads(Path(protocol['input_audit']).read_text())['sha256']
    for name in ('train.tsv', 'train.ltr', 'dict.ltr.txt'):
        p = str(Path(config['manifest']) / name)
        expected = bindings.get(p)
        if expected is None or file_sha(p) != expected:
            raise ValueError('unbound/changed manifest: ' + p)
    source = protocol['backbones'][a.backbone]
    for p, digest in source['sha256'].items():
        if file_sha(p) != digest:
            raise ValueError('changed public initialization: ' + p)
    # Launcher reserves 30 GiB for both runs. Per-stage checks account for files
    # already written by preceding stages instead of demanding 30 GiB each time.
    required_free = (4 if a.stage == 'stage1' else 12) * 1024**3
    if not a.smoke and shutil.disk_usage(str(out)).free < required_free:
        (out / 'blocked.json').write_text(json.dumps(dict(reason='insufficient_persistent_storage',
            required=required_free, available=shutil.disk_usage(str(out)).free)))
        raise SystemExit('insufficient storage; no model loaded and no training started')
    if not a.smoke and str(out.resolve()).startswith('/dev/shm/'):
        raise SystemExit('full training needs persistent checkpoint storage')
    seed = protocol['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rows, labels = read_split(config['manifest'], 'train')
    vocab = load_vocab(str(Path(config['manifest']) / 'dict.ltr.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Backbone(a.backbone, source['path'], max(vocab.values()) + 1, device)
    if a.stage == 'stage2':
        if not a.stage1_checkpoint:
            raise ValueError('matched Stage1 checkpoint required')
        ck = torch.load(a.stage1_checkpoint, map_location='cpu')
        if (ck['protocol_sha256'] != file_sha(a.protocol) or ck['backbone'] != a.backbone
                or ck['stage'] != 'stage1' or ck['step'] != protocol['stage1']['stop_at']):
            raise ValueError('Stage1 provenance/budget differs')
        model.load_state_dict(ck['model'], strict=True)
        del ck
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-8, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda', init_scale=128)
    accum, budget = ((2, 320000) if a.smoke and not a.smoke_full_batch
                     else (protocol['accum'], protocol['batch_samples']))
    steps = 2 if a.smoke else config['stop_at']
    batches = stream(rows, budget, seed)
    model.train()
    order_hash, draws, samples = hashlib.sha256(), 0, 0
    start = time.time()
    log = (out / 'updates.jsonl').open('w')
    results = []
    print(json.dumps(dict(event='model_loaded', backbone=a.backbone, stage=a.stage,
        parameters=sum(p.numel() for p in model.parameters()), smoke=a.smoke)), flush=True)
    for step in range(1, steps + 1):
        frozen = (step == 1) if a.smoke else step <= config['freeze_updates']
        model.encoder().requires_grad_(not frozen)
        reference = next(p for p in model.encoder().parameters() if p.ndim >= 2)
        before = reference.detach().clone() if a.smoke else None
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(accum):
            ids = next(batches)
            order_hash.update((' '.join(map(str, ids)) + '\n').encode())
            xs = [read_wav(rows[i][0], 20.0) for i in ids]
            if any(len(x) != rows[i][1] for x, i in zip(xs, ids)):
                raise ValueError('truncated/changed WAV length')
            ys = [[vocab[t] for t in labels[i]] for i in ids]
            draws += len(ids); samples += sum(len(x) for x in xs)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits, lengths = model(xs)
                loss = ctc_loss(logits, lengths, ys) / accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * accum)
        lr = tri_stage(step - 1, config['schedule_steps'], config['lr'])
        for group in opt.param_groups:
            group['lr'] = lr
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite gradient; no silently skipped optimizer step')
        scaler.step(opt); scaler.update()
        row = dict(step=step, loss=float(np.mean(losses)), lr=lr, frozen_encoder=frozen,
            gradient_norm=float(norm), row_draws=draws, audio_samples=samples,
            batch_order_sha256=order_hash.hexdigest(), elapsed_seconds=time.time()-start)
        if a.smoke:
            change = float((reference.detach()-before).abs().max())
            row['first_encoder_tensor_max_change'] = change
            assert (change == 0) if frozen else (change > 0), row
            results.append(row)
        log.write(json.dumps(row) + '\n'); log.flush()
        if step % 100 == 0 or a.smoke:
            print(json.dumps(row), flush=True)
        if not a.smoke and step in config['save_steps']:
            payload = dict(protocol_sha256=file_sha(a.protocol), backbone=a.backbone,
                stage=a.stage, step=step, model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                exposure=row, vocab=vocab,
                optimizer=opt.state_dict() if a.stage == 'stage2' and step == steps else None,
                scaler=scaler.state_dict(),
                rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), random=random.getstate(),
                    cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))
            destination = out / ('step_%d.pt' % step)
            if destination.exists():
                raise FileExistsError(str(destination))
            torch.save(payload, str(destination)+'.partial')
            Path(str(destination)+'.partial').rename(destination)
            del payload
            gc.collect()
    log.close()
    report = dict(backbone=a.backbone, stage=a.stage, smoke=a.smoke,
        smoke_full_batch=a.smoke_full_batch, updates=steps,
        protocol_sha256=file_sha(a.protocol), results=results,
        final_exposure=row, peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
        claim='execution_check_only_no_performance_measurement' if a.smoke else 'trained_not_evaluated')
    (out / 'finished.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

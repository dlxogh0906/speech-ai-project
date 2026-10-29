"""Train a fresh six-class EAT model after content/identity-group deduplication.

Never loads an earlier dialect checkpoint, teacher, or fitted normalizer.
Uses training-only SpecAugment, selects on validation, and does not score test.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import train_dialect_comparison as engine
from train_standard_six_class import metrics, write_json, LABELS


def load(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def rank(value):
    return value['macro_f1'], value['top1']


def augment(values, lengths):
    values = values.clone()
    for i, length in enumerate(lengths.tolist()):
        for _ in range(2):
            width = int(torch.randint(0, min(40, length // 12) + 1, ()).item())
            start = int(torch.randint(0, length - width + 1, ()).item())
            values[i, :, start:start + width, :] = 0
            width = int(torch.randint(0, 9, ()).item())
            start = int(torch.randint(0, 128 - width + 1, ()).item())
            values[i, :, :length, start:start + width] = 0
    return values


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.40, device=0)
    device = torch.device('cuda:0')

    def progress(stage, **extra):
        state = {'status': 'running', 'stage': stage, 'pid': os.getpid(),
                 'updated_at': datetime.now().astimezone().isoformat(),
                 'elapsed_seconds': round(time.monotonic() - started, 1), **extra}
        write_json(args.out / 'pipeline_status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    progress('verify_prepared_split')
    summary = json.loads((args.manifests / 'summary.json').read_text())
    manifest_sha = hashlib.sha256(b''.join((args.manifests / (s + '.csv')).read_bytes() for s in ['train', 'valid', 'test'])).hexdigest()
    assert manifest_sha == summary['manifest_sha256']
    assert all(n == 0 for values in summary['overlaps'].values() for n in values.values())
    rows = {s: load(args.manifests / (s + '.csv')) for s in ['train', 'valid']}
    assert not {r['person_id'] for r in rows['train']} & {r['person_id'] for r in rows['valid']}
    assert not {r['pcm16_fingerprint'] for r in rows['train']} & {r['pcm16_fingerprint'] for r in rows['valid']}
    assert all({int(r['label_id']) for r in rs} == set(range(6)) for rs in rows.values())
    args.model, args.workers = 'eat_large', 4
    protocol = {'seed': 42, 'manifest_sha256': manifest_sha, 'model': engine.MODELS['eat_large'],
                'initialization': 'Generic EAT pretrained model; fresh random head; no earlier dialect adaptations, teachers or fitted normalizers.',
                'features': ['layer15_mean', 'layer23_mean'], 'normalizer_fit': 'training only',
                'loss': 'Cross-entropy, equal class mass and equal identity-component mass within class',
                'head_epochs': 40, 'head_patience': 8, 'head_lr': .001,
                'encoder_last_blocks': 8, 'encoder_epochs': args.epochs, 'encoder_lr': .000005,
                'adapted_head_lr': .00005, 'batch_size': 16, 'accumulation': 2,
                'augmentation': 'Training only: two time masks up to min(40,L/12) frames and two frequency masks up to 8 bins',
                'selection': 'Validation macro F1, then accuracy; frozen clean head remains fallback',
                'test_inference': False, 'deployment': False,
                'evaluation_limitation': summary['evaluation_status'], 'data_limitations': summary['limitations'],
                'generic_pretraining_overlap': 'Not independently audited'}
    write_json(args.out / 'protocol.json', protocol)
    snapshot = args.out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['train_content_grouped_eat.py', 'train_dialect_comparison.py', 'train_standard_six_class.py']:
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    progress('load_generic_pretrained_encoder')
    encoder = engine.Encoder('eat_large', device)
    selected = [encoder.names.index(n) for n in protocol['features']]
    weight_file = Path(encoder.info['path']) / 'model.safetensors'
    protocol['generic_model_sha256'] = hashlib.sha256(weight_file.read_bytes()).hexdigest()
    write_json(args.out / 'protocol.json', protocol)
    progress('extract_training_features', rows=len(rows['train']))
    args.batch_size = 64
    cache, _ = engine.extract(args, {'train': rows['train']}, encoder)
    args.batch_size = 16
    progress('extract_validation_features', rows=len(rows['valid']))
    engine.extract(args, {'valid': rows['valid']}, encoder)
    assert not (cache / 'test.npy').exists()
    x, y = {}, {}
    for split in rows:
        assert np.load(cache / (split + '.ok.npy')).all()
        array = np.load(cache / (split + '.npy'), mmap_mode='r')
        x[split] = np.asarray(array[:, selected, :], np.float32).reshape(len(array), -1)
        y[split] = np.array([int(r['label_id']) for r in rows[split]])
    mu, sd = x['train'].mean(axis=0), np.maximum(x['train'].std(axis=0), .0001)
    for split in x:
        x[split] = (x[split] - mu) / sd
    counts = Counter((r['label_id'], r['person_id']) for r in rows['train'])
    people = {i: {r['person_id'] for r in rows['train'] if int(r['label_id']) == i} for i in range(6)}
    weights = np.array([1 / counts[(r['label_id'], r['person_id'])] / len(people[int(r['label_id'])]) for r in rows['train']], np.float32)
    weights /= weights.mean()
    head = nn.Linear(len(mu), 6).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
    best, best_name, stale, history = None, None, 0, []
    rng = np.random.default_rng(42)
    blocks = encoder.model.model.blocks[-8:]
    for block in blocks:
        for p in block.parameters():
            p.requires_grad = True
    adapted = {n: p for n, p in encoder.model.named_parameters() if p.requires_grad}
    checkpoint = args.out / 'best.pt'

    def save(name, value):
        state = {'labels': LABELS, 'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                 'encoder_delta': {k: v.detach().cpu() for k, v in adapted.items()},
                 'generic_encoder': encoder.info, 'generic_model_sha256': protocol['generic_model_sha256'],
                 'mu': mu, 'sd': sd, 'feature_names': protocol['features'], 'manifest_sha256': manifest_sha,
                 'selection': name, 'valid': value, 'test_scored': False}
        tmp = checkpoint.with_suffix('.tmp.pt')
        torch.save(state, tmp); tmp.replace(checkpoint)

    def cached_validation():
        head.eval()
        with torch.no_grad():
            logits = np.concatenate([head(torch.from_numpy(x['valid'][i:i+1024]).to(device)).cpu().numpy()
                                     for i in range(0, len(x['valid']), 1024)])
        return metrics(rows['valid'], logits), logits

    progress('train_fresh_head')
    for epoch in range(1, 41):
        head.train()
        order = rng.permutation(len(rows['train']))
        for at in range(0, len(order), 1024):
            idx = order[at:at + 1024]
            logits = head(torch.from_numpy(x['train'][idx]).to(device))
            loss = (F.cross_entropy(logits, torch.as_tensor(y['train'][idx], device=device), reduction='none') * torch.as_tensor(weights[idx], device=device)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        value, logits = cached_validation()
        history.append({'stage': 'head', 'epoch': epoch, 'accuracy': value['top1'], 'macro_f1': value['macro_f1']})
        if best is None or rank(value) > rank(best):
            best, best_name, stale = value, 'frozen_head_epoch_' + str(epoch), 0
            save(best_name, best)
            np.savez_compressed(args.out / 'valid_predictions.npz', logits=logits, row_id=np.array([r['row_id'] for r in rows['valid']]))
        else:
            stale += 1
        write_json(args.out / 'history.json', history)
        progress('head_epoch_complete', epoch=epoch, valid_top1=value['top1'], valid_macro_f1=value['macro_f1'])
        if stale >= 8:
            break
    initial = torch.load(checkpoint, map_location='cpu')
    head.load_state_dict(initial['head'])
    frozen_valid = initial['valid']
    del initial, x, optimizer
    mu_t, sd_t = torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device)
    differentiable = inspect.unwrap(engine.Encoder.__call__)

    def forward(values, lengths):
        features = differentiable(encoder, values, lengths)[:, selected, :].reshape(len(values), -1).half().float()
        return head((features - mu_t) / sd_t)

    datasets = {s: engine.PreparedClips(rs, 'eat_large') for s, rs in rows.items()}

    def validate():
        encoder.model.eval(); head.eval()
        order = sorted(range(len(rows['valid'])), key=lambda i: float(rows['valid'][i]['duration_sec']))
        loader = DataLoader(Subset(datasets['valid'], order), batch_size=16, num_workers=4, collate_fn=engine.Collator('eat_large'))
        logits = np.full((len(order), 6), np.nan, np.float32)
        with torch.no_grad():
            for idx, values, lengths, errors in loader:
                assert not errors, errors[:2]
                logits[idx] = forward(values, lengths).cpu().numpy()
        assert np.isfinite(logits).all()
        return metrics(rows['valid'], logits), logits

    optimizer = torch.optim.AdamW([{'params': list(adapted.values()), 'lr': .000005},
                                  {'params': head.parameters(), 'lr': .00005}], weight_decay=.001)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    batches = math.ceil(len(rows['train']) / 16)
    total_steps = args.epochs * math.ceil(batches / 2)
    step, stale = 0, 0
    progress('adapt_encoder', epoch=0, optimizer_steps=0)
    for epoch in range(1, args.epochs + 1):
        encoder.model.eval(); head.train()
        for block in blocks:
            block.train()
        loader = DataLoader(datasets['train'], batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(42 + epoch), num_workers=4,
                            collate_fn=engine.Collator('eat_large'), pin_memory=True)
        seen, total_loss = 0, 0.
        optimizer.zero_grad(set_to_none=True)
        for batch, (idx, values, lengths, errors) in enumerate(loader):
            assert not errors, errors[:2]
            logits = forward(augment(values, lengths), lengths)
            loss = (F.cross_entropy(logits, torch.as_tensor(y['train'][idx], device=device), reduction='none') * torch.as_tensor(weights[idx], device=device)).mean()
            assert torch.isfinite(loss)
            divisor = min(2, batches - (batch // 2) * 2)
            scaler.scale(loss / divisor).backward()
            if (batch + 1) % 2 == 0 or batch + 1 == batches:
                scaler.unscale_(optimizer)
                assert all(p.grad is not None for p in adapted.values()), 'Disconnected encoder gradient'
                torch.nn.utils.clip_grad_norm_(list(adapted.values()) + list(head.parameters()), 1.)
                factor = min((step + 1) / max(1, int(.05 * total_steps)), 1.) * .5 * (1 + math.cos(math.pi * step / total_steps))
                optimizer.param_groups[0]['lr'] = .000005 * factor
                optimizer.param_groups[1]['lr'] = .00005 * factor
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                step += 1
            seen += len(idx); total_loss += loss.item() * len(idx)
            if batch % 50 == 0:
                progress('adapt_encoder', epoch=epoch, seen=seen, total=len(rows['train']), optimizer_steps=step, loss=total_loss / seen)
        value, logits = validate()
        history.append({'stage': 'adapt', 'epoch': epoch, 'accuracy': value['top1'], 'macro_f1': value['macro_f1'], 'loss': total_loss / seen})
        if rank(value) > rank(best):
            best, best_name, stale = value, 'augmented_encoder_epoch_' + str(epoch), 0
            save(best_name, best)
            np.savez_compressed(args.out / 'valid_predictions.npz', logits=logits, row_id=np.array([r['row_id'] for r in rows['valid']]))
        else:
            stale += 1
        write_json(args.out / 'history.json', history)
        progress('encoder_epoch_complete', epoch=epoch, valid_top1=value['top1'], valid_macro_f1=value['macro_f1'])
        if stale >= 2:
            break
    report = {'completed_at': datetime.now().astimezone().isoformat(), 'selected': best_name, 'valid': best,
              'frozen_valid': frozen_valid, 'test': None, 'test_scored': False, 'history': history,
              'protocol': protocol, 'data': summary, 'checkpoint': str(checkpoint),
              'elapsed_seconds': time.monotonic() - started, 'deployed': False}
    write_json(args.out / 'report.json', report)
    progress('complete', status='complete', validation_top1=best['top1'], test_scored=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / 'pipeline_status.json', {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        raise

"""Fresh six-class training on the full, fixed content-grouped training split.

Only immutable generic pretrained feature caches may be reused. Previous dialect
heads, encoder deltas, teachers, and fitted normalizers are never loaded.
Test audio is never read here. Checkpoint selection uses validation only.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import gc
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

ROOT = Path('/data/speech-ai/out/dialect')


def load(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def hash_files(paths):
    h = hashlib.sha256()
    for path in paths:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def save_torch(path, value):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp)
    tmp.replace(path)


def prefill(args, rows, encoder):
    """Remap generic features by exact row/audio identity, never by old split."""
    cache = args.out / 'features'
    cache.mkdir(exist_ok=True)
    wanted = engine.identity(args, encoder)
    identity_file = cache / 'identity.json'
    if identity_file.exists():
        assert json.loads(identity_file.read_text()) == wanted
    write_json(identity_file, wanted)
    if (cache / 'prefill.json').exists():
        return
    shape = (len(rows), len(encoder.names), encoder.width)
    p, d = cache / 'train.npy', cache / 'train.ok.npy'
    data = np.lib.format.open_memmap(str(p), mode='r+' if p.exists() else 'w+', dtype=np.float16, shape=shape)
    done = np.lib.format.open_memmap(str(d), mode='r+' if d.exists() else 'w+', dtype=bool, shape=(len(rows),))
    target = {r['row_id']: i for i, r in enumerate(rows)}
    sources = [(ROOT / 'speakerid_20261011/manifests', ROOT / 'compare_speakerid_20261011' / args.model / 'features', ['train', 'valid', 'test'])]
    if args.model == 'eat_large':
        sources.append((ROOT / 'content_grouped_20261014/manifests', ROOT / 'content_grouped_20261014/eat_augmented/features', ['train']))
    provenance = []
    for manifests, source, splits in sources:
        ident = json.loads((source / 'identity.json').read_text())
        assert ident['manifest_sha256'] == hash_files([manifests / (s + '.csv') for s in ['train', 'valid', 'test']])
        if 'model' in ident:
            for key in ['model', 'files', 'features', 'preprocessing', 'feature_clip', 'implementation_sha256']:
                assert ident[key] == wanted[key], (source, key)
        else:
            assert args.model == 'xlsr2b'
            weight = Path(encoder.info['path'])
            assert ident['encoder_path'] == str(weight)
            assert ident['encoder_bytes'] == weight.stat().st_size
            assert ident['encoder_mtime_ns'] == weight.stat().st_mtime_ns
            assert ident['layers'] == [31, 47] and ident['normalize_waveform'] is False
            assert ident['implementation_sha256'] == hash_files([Path(__file__).with_name('train_xlsr2b_dialect.py')])
        for split in splits:
            old_rows = load(manifests / (split + '.csv'))
            old = np.load(source / (split + '.npy'), mmap_mode='r')
            ok = np.load(source / (split + '.ok.npy'), mmap_mode='r')
            assert old.shape == (len(old_rows), len(encoder.names), encoder.width)
            pairs = []
            for j, row in enumerate(old_rows):
                i = target.get(row['row_id'])
                if i is None or done[i] or not ok[j]:
                    continue
                assert all(rows[i][k] == row[k] for k in ['audio_path', 'start_sec', 'duration_sec', 'label_id'])
                pairs.append((i, j))
            for at in range(0, len(pairs), 4096):
                ii, jj = np.asarray(pairs[at:at + 4096]).T
                values = np.asarray(old[jj])
                assert np.isfinite(values).all()
                data[ii] = values
                done[ii] = True
            data.flush(); done.flush()
            provenance.append(dict(cache=str(source), original_split=split, copied_rows=len(pairs), identity=ident))
            print('[generic cache] %s %s: %d rows' % (source, split, len(pairs)), flush=True)
            del old_rows, old, ok, pairs
    write_json(cache / 'prefill.json', dict(reused=int(done.sum()), total=len(rows), sources=provenance,
               policy='Raw generic encoder features only; row/audio/start/duration matched; new training normalizer fitted independently. New validation features recomputed.'))
    del data, done
    gc.collect()


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.40 if args.model == 'eat_large' else .65)
    device = torch.device('cuda:0')
    args.workers = 4
    train_batch = 16 if args.model == 'eat_large' else 8
    accumulation = 4
    last_blocks = 4 if args.model == 'eat_large' else 8

    def progress(stage, **extra):
        state = dict(status='running', stage=stage, model=args.model, pid=os.getpid(),
                     updated_at=datetime.now().astimezone().isoformat(),
                     elapsed_seconds=round(time.monotonic() - started, 1))
        state.update(extra)
        write_json(args.out / 'pipeline_status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    progress('verify_fixed_split')
    summary = json.loads((args.manifests / 'summary.json').read_text())
    manifest_sha = hash_files([args.manifests / (s + '.csv') for s in ['train', 'valid', 'test']])
    assert manifest_sha == summary['manifest_sha256']
    assert all(n == 0 for v in summary['overlaps'].values() for n in v.values())
    for s, expected in summary['fixed_heldout_sha256'].items():
        assert hash_files([args.manifests / (s + '.csv')]) == expected
    rows = {s: load(args.manifests / (s + '.csv')) for s in ['train', 'valid']}
    assert all({int(r['label_id']) for r in rs} == set(range(6)) for rs in rows.values())
    for key in ['person_id', 'original_person_id', 'record_key', 'audio_path', 'pcm16_fingerprint']:
        assert not {r[key] for r in rows['train']} & {r[key] for r in rows['valid']}
    protocol = dict(seed=42, manifest_sha256=manifest_sha, model=engine.MODELS[args.model],
                    initialization='Generic pretrained encoder plus fresh random six-class head. No prior dialect weights, teacher, or fitted normalizer.',
                    normalizer_fit='New training rows only', train_rows=len(rows['train']), valid_rows=len(rows['valid']),
                    features=['layer15_mean', 'layer23_mean'] if args.model == 'eat_large' else ['layer31_mean', 'layer47_mean'],
                    encoder_last_blocks=last_blocks, encoder_epochs=args.epochs, encoder_lr=1e-5, head_lr=1e-4,
                    head_epochs=100, head_patience=12, head_initial_lr=.001,
                    train_batch=train_batch, accumulation=accumulation, validation_batch=16,
                    augmentation='None for this full-volume comparison', loss='Equal class mass, equal identity-component mass within class',
                    selection='Validation macro F1, then top1; best frozen head is retained as fallback',
                    resume='Atomic optimizer/scaler/RNG checkpoint each 1000 updates and each epoch',
                    test_inference=False, deployment=False, limitations=summary['limitations'])
    protocol_file = args.out / 'protocol.json'
    if protocol_file.exists():
        old_protocol = json.loads(protocol_file.read_text())
        assert all(old_protocol[k] == v for k, v in protocol.items()), 'Protocol changed during resume'
    write_json(protocol_file, protocol)
    snapshot = args.out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['train_content_grouped_full.py', 'train_dialect_comparison.py', 'train_standard_six_class.py', 'train_xlsr2b_dialect.py']:
        destination = snapshot / name
        source = Path(__file__).with_name(name)
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes(), 'Training implementation changed; use a new run'
        else:
            shutil.copyfile(source, destination)
    progress('load_generic_pretrained_encoder')
    encoder = engine.Encoder(args.model, device)
    if args.model == 'xlsr2b':
        from fairseq.modules.checkpoint_activations import unwrap_checkpoint
        unwrap_checkpoint(encoder.model)
    selected = [encoder.names.index(name) for name in protocol['features']]
    blocks = encoder.model.model.blocks[-last_blocks:] if args.model == 'eat_large' else encoder.model.encoder.layers[-last_blocks:]
    adapted = {}
    for block in blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True
    adapted = {n: p for n, p in encoder.model.named_parameters() if p.requires_grad}
    differentiable = inspect.unwrap(engine.Encoder.__call__)
    datasets = {s: engine.PreparedClips(rs, args.model) for s, rs in rows.items()}
    progress('gradient_preflight')
    # A full batch of the longest training windows exercises the actual graph.
    longest = sorted(range(len(rows['train'])), key=lambda i: float(rows['train'][i]['duration_sec']), reverse=True)[:train_batch]
    _, values, lengths, errors = engine.Collator(args.model)([datasets['train'][i] for i in longest])
    assert not errors, errors
    features = differentiable(encoder, values, lengths)[:, selected, :].reshape(len(values), -1)
    probe = nn.Linear(features.shape[1], 6).to(device)
    F.cross_entropy(probe(features), torch.tensor([int(rows['train'][i]['label_id']) for i in longest], device=device)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapted.values()), 'Encoder gradients disconnected/nonfinite'
    assert any(torch.count_nonzero(p.grad).item() for p in adapted.values()), 'All encoder gradients zero'
    write_json(args.out / 'gradient_preflight.json', dict(passed=True, batch_size=train_batch,
               windows=[rows['train'][i]['row_id'] for i in longest], trainable_parameters=sum(p.numel() for p in adapted.values()),
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
    encoder.model.zero_grad(set_to_none=True)
    del probe, features, values, lengths
    torch.cuda.empty_cache()
    if args.preflight_only:
        progress('preflight_complete', status='complete')
        return
    progress('verify_generic_weight_file')
    weight = Path(encoder.info['path'])
    if weight.is_dir():
        weight = weight / 'model.safetensors'
    protocol['generic_model_sha256'] = hash_files([weight])
    write_json(protocol_file, protocol)
    progress('reuse_verified_generic_features')
    prefill(args, rows['train'], encoder)
    args.batch_size = 64 if args.model == 'eat_large' else 16
    progress('extract_missing_training_features')
    cache, _ = engine.extract(args, {'train': rows['train']}, encoder)
    args.batch_size = 16
    progress('extract_fresh_validation_features')
    engine.extract(args, {'valid': rows['valid']}, encoder)
    assert not (cache / 'test.npy').exists()
    x, y = {}, {}
    for split in rows:
        assert np.load(cache / (split + '.ok.npy')).all()
        array = np.load(cache / (split + '.npy'), mmap_mode='r')
        x[split] = np.asarray(array[:, selected, :], np.float32).reshape(len(array), -1)
        y[split] = np.array([int(r['label_id']) for r in rows[split]])
    mu, sd = x['train'].mean(0), np.maximum(x['train'].std(0), 1e-4)
    for split in x:
        x[split] -= mu
        x[split] /= sd
    counts = Counter((r['label_id'], r['person_id']) for r in rows['train'])
    people = {i: {r['person_id'] for r in rows['train'] if int(r['label_id']) == i} for i in range(6)}
    weights = np.array([1 / counts[(r['label_id'], r['person_id'])] / len(people[int(r['label_id'])]) for r in rows['train']], np.float32)
    weights /= weights.mean()
    torch.manual_seed(42)
    head = nn.Linear(len(mu), 6).to(device)
    checkpoint, head_done = args.out / 'best.pt', args.out / 'head_complete.json'
    history, best, best_name = [], None, None

    def save_best(name, value, logits):
        save_torch(checkpoint, dict(labels=LABELS, head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                   encoder_delta={k: v.detach().cpu() for k, v in adapted.items()}, generic_encoder=encoder.info,
                   generic_model_sha256=protocol['generic_model_sha256'], mu=mu, sd=sd, feature_names=protocol['features'],
                   manifest_sha256=manifest_sha, selection=name, valid=value, test_scored=False))
        np.savez_compressed(args.out / 'valid_predictions.npz', logits=logits,
                            row_id=np.array([r['row_id'] for r in rows['valid']]))

    if not head_done.exists():
        progress('train_fresh_head')
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
        rng, stale = np.random.default_rng(42), 0
        for epoch in range(1, 101):
            head.train()
            order = rng.permutation(len(rows['train']))
            for at in range(0, len(order), 1024):
                idx = order[at:at + 1024]
                logits = head(torch.from_numpy(x['train'][idx]).to(device))
                loss = (F.cross_entropy(logits, torch.as_tensor(y['train'][idx], device=device), reduction='none') * torch.as_tensor(weights[idx], device=device)).mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            head.eval()
            with torch.no_grad():
                logits = np.concatenate([head(torch.from_numpy(x['valid'][i:i + 1024]).to(device)).cpu().numpy() for i in range(0, len(x['valid']), 1024)])
            value = metrics(rows['valid'], logits)
            history.append(dict(stage='head', epoch=epoch, accuracy=value['top1'], macro_f1=value['macro_f1']))
            if best is None or (value['macro_f1'], value['top1']) > (best['macro_f1'], best['top1']):
                best, best_name, stale = value, 'frozen_head_epoch_' + str(epoch), 0
                save_best(best_name, best, logits)
            else:
                stale += 1
            write_json(args.out / 'history.json', history)
            progress('head_epoch_complete', epoch=epoch, valid_top1=value['top1'], valid_macro_f1=value['macro_f1'])
            if stale >= 12:
                break
        write_json(head_done, dict(frozen_valid=best, selected=best_name))
        del optimizer
    initial = torch.load(checkpoint, map_location='cpu')
    assert initial['manifest_sha256'] == manifest_sha
    assert initial['generic_model_sha256'] == protocol['generic_model_sha256']
    assert np.array_equal(mu, initial['mu']) and np.array_equal(sd, initial['sd'])
    head.load_state_dict(initial['head'])
    with torch.no_grad():
        for name, value in initial['encoder_delta'].items():
            adapted[name].copy_(value.to(device))
    best, best_name = initial['valid'], initial['selection']
    frozen_valid = json.loads(head_done.read_text())['frozen_valid']
    history = json.loads((args.out / 'history.json').read_text())
    del initial, x, array
    gc.collect(); torch.cuda.empty_cache()
    mu_t, sd_t = torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device)

    def forward(values, lengths):
        feature = differentiable(encoder, values, lengths)[:, selected, :].reshape(len(values), -1).half().float()
        return head((feature - mu_t) / sd_t)

    def validate():
        encoder.model.eval(); head.eval()
        order = sorted(range(len(rows['valid'])), key=lambda i: float(rows['valid'][i]['duration_sec']))
        loader = DataLoader(Subset(datasets['valid'], order), batch_size=16, num_workers=4, collate_fn=engine.Collator(args.model))
        logits = np.full((len(order), 6), np.nan, np.float32)
        with torch.no_grad():
            for idx, values, lengths, errors in loader:
                assert not errors, errors[:2]
                logits[idx] = forward(values, lengths).cpu().numpy()
        assert np.isfinite(logits).all()
        return metrics(rows['valid'], logits), logits

    optimizer = torch.optim.AdamW([{'params': list(adapted.values()), 'lr': 1e-5},
                                  {'params': head.parameters(), 'lr': 1e-4}], weight_decay=.001)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    batches = math.ceil(len(rows['train']) / train_batch)
    total_steps = args.epochs * math.ceil(batches / accumulation)
    step, updates, skipped, start_epoch, start_batch, seen, total_loss = 0, 0, 0, 1, 0, 0, 0.
    resume_path = args.out / 'resume.pt'
    if resume_path.exists():
        state = torch.load(resume_path, map_location='cpu')
        assert state['manifest_sha256'] == manifest_sha
        head.load_state_dict(state['head'])
        with torch.no_grad():
            for name, value in state['encoder_delta'].items():
                adapted[name].copy_(value.to(device))
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step, updates, skipped = state['step'], state['updates'], state['skipped']
        start_epoch, start_batch = state['epoch'], state['next_batch']
        seen, total_loss, history = state['seen'], state['total_loss'], state['history']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state

    def save_resume(epoch, next_batch, seen, total_loss):
        save_torch(resume_path, dict(manifest_sha256=manifest_sha, epoch=epoch, next_batch=next_batch,
                   seen=seen, total_loss=total_loss, step=step, updates=updates, skipped=skipped, history=history,
                   encoder_delta={n: p.detach().cpu() for n, p in adapted.items()},
                   head={k: v.detach().cpu() for k, v in head.state_dict().items()}, optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))

    anchor = max(adapted.values(), key=lambda p: p.numel())
    anchor_initial = anchor.detach().reshape(-1)[:4096].clone()
    resumed_step = step
    progress('adapt_encoder', epoch=start_epoch, optimizer_updates=updates)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        encoder.model.eval(); head.train()
        for block in blocks:
            block.train()
        order = torch.randperm(len(rows['train']), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        offset = start_batch if epoch == start_epoch else 0
        if not offset:
            seen, total_loss = 0, 0.
        loader = DataLoader(Subset(datasets['train'], order[offset * train_batch:]), batch_size=train_batch,
                            num_workers=4, collate_fn=engine.Collator(args.model), pin_memory=True,
                            generator=torch.Generator().manual_seed(4242 + epoch))
        optimizer.zero_grad(set_to_none=True)
        for batch, (idx, values, lengths, errors) in enumerate(loader, offset):
            assert not errors, errors[:2]
            logits = forward(values, lengths)
            loss = (F.cross_entropy(logits, torch.as_tensor(y['train'][idx], device=device), reduction='none') * torch.as_tensor(weights[idx], device=device)).mean()
            assert torch.isfinite(loss)
            divisor = min(accumulation, batches - batch // accumulation * accumulation)
            scaler.scale(loss / divisor).backward()
            seen += len(idx); total_loss += loss.item() * len(idx)
            if (batch + 1) % accumulation == 0 or batch + 1 == batches:
                scaler.unscale_(optimizer)
                assert all(p.grad is not None for p in adapted.values()), 'Disconnected encoder gradient'
                torch.nn.utils.clip_grad_norm_(list(adapted.values()) + list(head.parameters()), 1.)
                factor = min((step + 1) / 200, 1.) * .5 * (1 + math.cos(math.pi * step / total_steps))
                optimizer.param_groups[0]['lr'] = 1e-5 * factor
                optimizer.param_groups[1]['lr'] = 1e-4 * factor
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                did_update = scaler.get_scale() >= old_scale
                updates += int(did_update); skipped += int(not did_update); step += 1
                changed = not torch.equal(anchor_initial, anchor.detach().reshape(-1)[:4096])
                if step - resumed_step >= 50 and updates == 0:
                    raise RuntimeError('No actual optimizer updates after 50 attempts')
                if step - resumed_step >= 50 and not changed:
                    raise RuntimeError('Trainable encoder weights did not change')
                if step == 1 or step % 20 == 0:
                    rate = (seen - offset * train_batch) / max(time.monotonic() - epoch_started, 1e-6)
                    progress('adapt_encoder', epoch=epoch, seen=seen, total=len(rows['train']),
                             optimizer_attempts=step, optimizer_updates=updates, amp_skips=skipped,
                             encoder_weights_changed=changed, loss=total_loss / seen, clips_per_second=rate,
                             epoch_remaining_seconds=(len(rows['train']) - seen) / max(rate, 1e-6))
                if step % 1000 == 0:
                    save_resume(epoch, batch + 1, seen, total_loss)
        progress('validate_encoder_epoch', epoch=epoch, optimizer_updates=updates)
        value, logits = validate()
        history.append(dict(stage='adapt', epoch=epoch, accuracy=value['top1'], macro_f1=value['macro_f1'], loss=total_loss / seen,
                            optimizer_updates=updates, amp_skips=skipped, epoch_seconds=time.monotonic() - epoch_started))
        if (value['macro_f1'], value['top1']) > (best['macro_f1'], best['top1']):
            best, best_name = value, 'encoder_epoch_' + str(epoch)
            save_best(best_name, best, logits)
        write_json(args.out / 'history.json', history)
        save_resume(epoch + 1, 0, 0, 0.)
        progress('encoder_epoch_complete', epoch=epoch, valid_top1=value['top1'], valid_macro_f1=value['macro_f1'], optimizer_updates=updates)
    # Re-evaluate the selected checkpoint independently of its saved predictions.
    chosen = torch.load(checkpoint, map_location='cpu')
    head.load_state_dict(chosen['head'])
    with torch.no_grad():
        for name, value in chosen['encoder_delta'].items():
            adapted[name].copy_(value.to(device))
    check, logits = validate()
    saved = np.load(args.out / 'valid_predictions.npz')
    assert np.array_equal(saved['row_id'], np.array([r['row_id'] for r in rows['valid']]))
    assert np.array_equal(saved['logits'].argmax(1), logits.argmax(1)), 'Saved checkpoint/prediction mismatch'
    assert abs(check['top1'] - best['top1']) < 1e-12
    write_json(args.out / 'report.json', dict(completed_at=datetime.now().astimezone().isoformat(), selected=best_name,
               valid=best, frozen_valid=frozen_valid, test=None, test_scored=False, history=history, protocol=protocol,
               data=summary, checkpoint=str(checkpoint), elapsed_seconds=time.monotonic() - started,
               optimizer_updates=updates, amp_skips=skipped, selected_checkpoint_verified=True, deployed=False))
    progress('complete', status='complete', validation_top1=best['top1'], optimizer_updates=updates, test_scored=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=['eat_large', 'xlsr2b'], required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / 'pipeline_status.json', dict(status='failed', error=type(exc).__name__ + ': ' + str(exc)))
        raise

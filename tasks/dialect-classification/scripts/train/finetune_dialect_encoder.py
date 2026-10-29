"""Fine-tune the last encoder blocks and dialect head on verified person splits.

Start from the validation-selected frozen baseline. Select an adapted checkpoint
using validation Top1 (Macro F1 as the tie-breaker) by default, including the
frozen baseline as an eligible fallback. Macro F1 selection is also available.
The test split is scored only after that selection. Source weights stay intact.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

spec = importlib.util.spec_from_file_location('dialect_encoders', Path(__file__).with_name('train_dialect_comparison.py'))
encoders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoders)
heads = encoders.heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['xlsr2b', 'eat_large'], required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifests', default='/data/speech-ai/out/dialect/speakerid_20261011/manifests')
    parser.add_argument('--last-blocks', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--selection-metric', choices=['top1', 'macro_f1'], default='top1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    heads.write_json(out / 'config.json', vars(args))
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.55, device=0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = {split: heads.load_rows(Path(args.manifests) / (split + '.csv')) for split in ['train', 'valid', 'test']}
    summary = heads.require_person_disjoint_split(args, rows)
    record_sets = {split: {r['record_key'] for r in values} for split, values in rows.items()}
    if any(record_sets[a] & record_sets[b] for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]):
        raise RuntimeError('Recording overlap in fine-tuning manifests')
    baseline = Path(args.baseline)
    original = json.loads((baseline / 'report.json').read_text())
    if original['data_summary']['manifest_sha256'] != summary['manifest_sha256']:
        raise RuntimeError('Frozen baseline and fine-tuning use different manifests')
    state = torch.load(baseline / 'best.pt', map_location='cpu')
    run = {'model': args.model, 'pid': os.getpid(), 'status': 'running', 'data_summary': summary,
           'baseline_report': str(baseline / 'report.json'), 'started_at': datetime.datetime.now().astimezone().isoformat()}

    def progress(stage, **extra):
        run.update(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(), **extra)
        heads.write_json(out / 'progress.json', run)
        print('[fine-tune] ' + json.dumps({k: v for k, v in run.items() if k != 'data_summary'}), flush=True)

    try:
        progress('load_encoder')
        encoder = encoders.Encoder(args.model, device)
        if args.model == 'xlsr2b':
            # Fairseq's checkpoint wrapper only sees tensor inputs, not enclosed
            # parameters. A frozen prefix gives it inputs without gradients and
            # disconnects newly unfrozen blocks. Partial training fits in memory
            # without that wrapper, so keep ordinary autograd for these blocks.
            from fairseq.modules.checkpoint_activations import unwrap_checkpoint
            unwrap_checkpoint(encoder.model)
        if Path(state['encoder_checkpoint']).resolve() != Path(encoder.info['path']).resolve():
            raise RuntimeError('Baseline head belongs to another pretrained encoder')
        selected = [encoder.names.index(name) for name in state['feature_names']]
        head = nn.Linear(len(selected) * encoder.width, len(heads.LABELS)).to(device)
        head.load_state_dict(state['head'])
        mu, sd = torch.as_tensor(state['mu'], device=device), torch.as_tensor(state['sd'], device=device)
        blocks = encoder.model.encoder.layers if args.model == 'xlsr2b' else encoder.model.model.blocks
        if not 0 < args.last_blocks <= len(blocks):
            raise RuntimeError('Invalid number of encoder blocks')
        for block in blocks[-args.last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        adapted = {name: p for name, p in encoder.model.named_parameters() if p.requires_grad}
        count = sum(p.numel() for p in adapted.values())
        if not count:
            raise RuntimeError('No encoder parameters were enabled for training')
        progress('initialize_head', encoder_trainable_parameters=count,
                 encoder_total_parameters=encoder.info['parameters'], head_trainable_parameters=sum(p.numel() for p in head.parameters()),
                 first_trainable_block=len(blocks) - args.last_blocks, last_trainable_block=len(blocks) - 1)
        # The comparison's feature computation has a no_grad wrapper. Unwrap it
        # here to preserve exactly the same preprocessing/pooling with autograd.
        differentiable_forward = inspect.unwrap(encoders.Encoder.__call__)

        def forward(values, lengths):
            features = differentiable_forward(encoder, values, lengths)
            if not torch.isfinite(features).all():
                raise RuntimeError('Nonfinite fine-tuning features')
            # Match the original FP16 feature cache before normalization.
            features = features[:, selected, :].reshape(len(values), -1).half().float()
            return head((features - mu) / sd)

        datasets = {split: encoders.PreparedClips(values, args.model, encoder.processor) for split, values in rows.items()}
        collator = encoders.Collator(args.model)
        ys = {split: np.array([int(r['label_id']) for r in values], dtype=np.int64) for split, values in rows.items()}
        person_counts = Counter(r['person_id'] for r in rows['train'])
        region_people = {label: {r['person_id'] for r in rows['train'] if int(r['label_id']) == label} for label in range(5)}
        weights = np.array([1 / person_counts[r['person_id']] / len(region_people[int(r['label_id'])]) for r in rows['train']], dtype=np.float32)
        weights /= weights.mean()

        def evaluate(split):
            encoder.model.eval(); head.eval()
            order = sorted(range(len(rows[split])), key=lambda i: (float(rows[split][i]['duration_sec']), i))
            loader = DataLoader(Subset(datasets[split], order), batch_size=args.eval_batch_size, num_workers=args.workers,
                                collate_fn=collator, pin_memory=True)
            logits = np.full((len(rows[split]), 5), np.nan, dtype=np.float32)
            predictions = np.full(len(rows[split]), -1, dtype=np.int64)
            started = time.perf_counter()
            with torch.no_grad():
                for number, (indices, values, lengths, errors) in enumerate(loader):
                    if errors:
                        raise RuntimeError('Evaluation audio errors: ' + str(errors[:3]))
                    result = forward(values, lengths)
                    if not torch.isfinite(result).all():
                        raise RuntimeError('Nonfinite evaluation logits')
                    logits[indices] = result.cpu().numpy()
                    predictions[indices] = logits[indices].argmax(1)
                    if number % 100 == 0:
                        progress('evaluate_' + split, evaluated=min((number + 1) * args.eval_batch_size, len(order)),
                                 evaluation_total=len(order), evaluation_elapsed_seconds=round(time.perf_counter() - started, 1))
            return heads.metrics(ys[split], predictions), logits, predictions

        cached_identity = json.loads((baseline / 'features/identity.json').read_text())
        if cached_identity['manifest_sha256'] != summary['manifest_sha256']:
            raise RuntimeError('Baseline feature cache has different rows')
        cached = np.load(baseline / 'features/valid.npy', mmap_mode='r')
        x = np.asarray(cached[:, selected, :], np.float32).reshape(len(cached), -1)
        valid_ok = np.load(baseline / 'features/valid.ok.npy')
        _, baseline_predictions = heads.evaluate(head, x, state['mu'], state['sd'], device, valid_ok)
        baseline_valid = heads.metrics(ys['valid'], baseline_predictions)
        del cached, x
        def rank_validation(metrics):
            primary, secondary = ('top1', 'macro_f1') if args.selection_metric == 'top1' else ('macro_f1', 'top1')
            return metrics[primary], metrics[secondary]

        best = rank_validation(baseline_valid)
        best_epoch, best_valid, stale = -1, baseline_valid, 0
        checkpoint_path = out / 'best_adapted.pt'

        def save_checkpoint(epoch, valid):
            checkpoint = {'model': args.model, 'encoder_metadata': encoder.info,
                          'encoder_delta': {name: p.detach().cpu() for name, p in adapted.items()},
                          'head': {name: p.detach().cpu() for name, p in head.state_dict().items()},
                          'mu': state['mu'], 'sd': state['sd'], 'feature_names': state['feature_names'],
                          'labels': heads.LABELS, 'data_summary': summary, 'epoch': epoch, 'valid': valid, 'config': vars(args)}
            temporary = checkpoint_path.with_suffix('.tmp.pt')
            torch.save(checkpoint, temporary)
            os.replace(temporary, checkpoint_path)

        save_checkpoint(-1, baseline_valid)
        optimizer = torch.optim.AdamW([{'params': list(adapted.values()), 'lr': args.encoder_lr},
                                       {'params': head.parameters(), 'lr': args.head_lr}], weight_decay=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        train_parameters = list(adapted.values()) + list(head.parameters())
        anchor_name = max(adapted, key=lambda name: adapted[name].numel())
        anchor = adapted[anchor_name].detach().flatten()[:1024].clone()
        total_steps = args.epochs * math.ceil(math.ceil(len(rows['train']) / args.batch_size) / args.accumulation)
        step, actual_steps, skipped_steps = 0, 0, 0
        history = []
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            encoder.model.eval()
            for block in blocks[-args.last_blocks:]:
                block.train()
            head.train()
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, generator=generator,
                                num_workers=args.workers, collate_fn=collator, pin_memory=True)
            optimizer.zero_grad(set_to_none=True)
            epoch_loss, epoch_seen, epoch_started = 0., 0, time.perf_counter()
            for number, (indices, values, lengths, errors) in enumerate(loader):
                if errors:
                    raise RuntimeError('Training audio errors: ' + str(errors[:3]))
                predictions = forward(values, lengths)
                if not predictions.requires_grad:
                    raise RuntimeError('Fine-tuning forward disabled gradients')
                labels = torch.as_tensor(ys['train'][indices], device=device)
                sample_weights = torch.as_tensor(weights[indices], device=device)
                loss = (F.cross_entropy(predictions, labels, reduction='none') * sample_weights).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                group_size = min(args.accumulation, len(loader) - number // args.accumulation * args.accumulation)
                scaler.scale(loss / group_size).backward()
                if epoch == 0 and number == 0:
                    run['encoder_parameters_with_grad'] = sum(p.grad is not None for p in adapted.values())
                    if not run['encoder_parameters_with_grad']:
                        raise RuntimeError('Encoder backward produced no parameter gradients')
                epoch_loss += float(loss.detach()) * len(indices)
                epoch_seen += len(indices)
                if (number + 1) % args.accumulation == 0 or number + 1 == len(loader):
                    step += 1
                    warmup = min(1., step / max(args.warmup_steps, 1))
                    fraction = max(0., min(1., (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)))
                    factor = warmup * .5 * (1 + math.cos(math.pi * fraction))
                    optimizer.param_groups[0]['lr'] = args.encoder_lr * factor
                    optimizer.param_groups[1]['lr'] = args.head_lr * factor
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(train_parameters, 1.)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= old_scale:
                        actual_steps += 1
                    else:
                        skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if step == 1 or step % 100 == 0:
                        changed = bool((adapted[anchor_name].detach().flatten()[:1024] != anchor).any())
                        progress('train', epoch=epoch + 1, epoch_seen=epoch_seen, epoch_total=len(rows['train']),
                                 optimizer_steps=actual_steps, skipped_optimizer_steps=skipped_steps,
                                 loss=epoch_loss / epoch_seen, gradient_norm=float(norm),
                                 encoder_weights_updated=changed, encoder_lr=optimizer.param_groups[0]['lr'],
                                 epoch_elapsed_seconds=round(time.perf_counter() - epoch_started, 1),
                                 clips_per_second=round(epoch_seen / (time.perf_counter() - epoch_started), 2))
                    if skipped_steps >= 50 and actual_steps == 0:
                        raise RuntimeError('No optimizer update succeeded after 50 scaler reductions')
                del predictions, loss, values, lengths
            valid, _, _ = evaluate('valid')
            history.append({'epoch': epoch + 1, 'train_loss': epoch_loss / epoch_seen, 'valid': valid,
                            'elapsed_seconds': time.perf_counter() - epoch_started})
            rank = rank_validation(valid)
            if rank > best:
                best, best_epoch, best_valid, stale = rank, epoch, valid, 0
                save_checkpoint(epoch, valid)
            else:
                stale += 1
            heads.write_json(out / 'history.json', history)
            progress('epoch_complete', completed_epoch=epoch + 1, validation_top1=valid['top1'],
                     validation_macro_f1=valid['macro_f1'], selected_epoch=best_epoch + 1)
            if stale >= args.patience:
                break
        selected_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder.model.load_state_dict(selected_checkpoint['encoder_delta'], strict=False)
        head.load_state_dict(selected_checkpoint['head'])
        test, logits, predictions = evaluate('test')
        groups = defaultdict(list)
        for index, row in enumerate(rows['test']):
            groups[(row['speaker_key'], int(row['label_id']))].append(index)
        labels3, predicted3 = [], []
        for (_, label), indices in groups.items():
            indices.sort(key=lambda index: float(rows['test'][index]['start_sec']))
            for at in range(0, len(indices) - 2, 3):
                labels3.append(label); predicted3.append(int(logits[indices[at:at+3]].mean(0).argmax()))
        report = {'model': args.model, 'encoder_frozen': False, 'trainable_encoder_parameters': count,
                  'last_blocks': args.last_blocks, 'baseline_report': str(baseline / 'report.json'),
                  'baseline_validation': baseline_valid, 'selected_validation': best_valid,
                  'selected_epoch': best_epoch + 1, 'selected_encoder_adapted': best_epoch >= 0,
                  'selection_metric': args.selection_metric,
                  'selection_rule': 'Validation %s first; include frozen baseline, test only after selection' % args.selection_metric,
                  'test_single_window': test, 'test_three_utterance_windows_mean_logits': heads.metrics(labels3, predicted3),
                  'data_summary': summary, 'config': vars(args), 'history': history,
                  'elapsed_seconds': time.perf_counter() - started,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'optimizer_steps': actual_steps, 'skipped_optimizer_steps': skipped_steps}
        np.savez(out / 'test_predictions.npz', logits=logits, predictions=predictions, labels=ys['test'])
        heads.write_json(out / 'report.json', report)
        run['status'] = 'complete'
        progress('complete', test_top1=test['top1'], test_macro_f1=test['macro_f1'], selected_encoder_adapted=best_epoch >= 0)
    except Exception as exc:
        run.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        progress('failed')
        raise


if __name__ == '__main__':
    main()

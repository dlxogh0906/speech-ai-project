"""Train XLSR six-class heads, partially fine-tune, then select on validation.

The previous EAT/XLSR ensemble remains an eligible baseline. Test predictions
are computed only after saving the final validation selection. All outputs are
isolated from the previous experiment and from serving applications.
"""
import argparse
from collections import Counter
import csv
import datetime
import gc
import hashlib
import html
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
from train_standard_six_class import LABELS, KO, metrics, write_json
from extract_xlsr_six_class import BASE, REGIONAL, ROOT, verify_data

ALPHAS = [.25, .5, .75, 1.]
MODES = ['preserved_regional_rows', 'all_rows_ce', 'all_rows_with_regional_kl']


def softmax(value):
    value = value - value.max(1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(1, keepdims=True)


def fuse(eat, xlsr, alpha):
    return (1. - alpha) * softmax(eat) + alpha * softmax(xlsr)


def balanced_weights(rows):
    counts = Counter((int(r['label_id']), r['person_id']) for r in rows)
    people = {k: {r['person_id'] for r in rows if int(r['label_id']) == k} for k in range(6)}
    weights = np.array([1. / counts[(int(r['label_id']), r['person_id'])] /
                        len(people[int(r['label_id'])]) for r in rows], np.float32)
    return weights / weights.mean()


def objective(logits, labels, weights, teacher, kd_weight):
    ce = F.cross_entropy(logits, labels, reduction='none')
    if kd_weight:
        kl = F.kl_div(F.log_softmax(logits[:, :5] / 2., 1),
                      F.softmax(teacher / 2., 1), reduction='none').sum(1) * 4.
        ce = ce + kd_weight * kl * (labels < 5).float()
    return (ce * weights).mean()


def rank(value):
    return value['top1'], value['macro_f1']


def save_torch(path, value):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(value, temporary)
    os.replace(temporary, path)


def predictions(path, rows, logits):
    np.savez_compressed(path, logits=logits,
                        labels=np.array([int(r['label_id']) for r in rows]),
                        row_id=np.array([r['row_id'] for r in rows]))


def paired_interval(rows, before, after):
    rng = np.random.default_rng(42)
    samples = []
    for label in range(6):
        people = sorted({r['person_id'] for r in rows if int(r['label_id']) == label})
        lookup = {person: i for i, person in enumerate(people)}
        delta, counts = np.zeros(len(people)), np.zeros(len(people))
        for index, row in enumerate(rows):
            if int(row['label_id']) != label:
                continue
            person = lookup[row['person_id']]
            delta[person] += int(after[index] == label) - int(before[index] == label)
            counts[person] += 1
        draws = rng.integers(0, len(people), (2000, len(people)))
        samples.append(delta[draws].sum(1) / counts[draws].sum(1))
    differences = np.mean(samples, axis=0) * 100.
    return dict(method='Paired class-stratified speaker-cluster bootstrap, 2000 draws, seed 42',
                lower_delta_pp=float(np.quantile(differences, .025)),
                upper_delta_pp=float(np.quantile(differences, .975)))


def self_test():
    torch.manual_seed(42)
    rows = [dict(label_id=str(k), person_id=str(k) + '_' + str(person))
            for k in range(6) for person in range(2) for _ in range(person + 1)]
    weights = balanced_weights(rows)
    mass = [weights[[int(r['label_id']) == k for r in rows]].sum() for k in range(6)]
    assert np.allclose(mass, mass[0])
    for k in range(6):
        a = weights[[r['person_id'] == str(k) + '_0' for r in rows]].sum()
        b = weights[[r['person_id'] == str(k) + '_1' for r in rows]].sum()
        assert np.isclose(a, b)
    x = np.array([[10000., 9999., -10000., 0., 1., 2.]], np.float32)
    z = x[:, ::-1].copy()
    assert np.allclose(fuse(x, z, 1.), softmax(z))
    assert np.allclose(fuse(x, z, .5).sum(1), 1.)
    logits = torch.randn(8, 6, requires_grad=True)
    teacher = torch.randn(8, 5)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 5, 0])
    loss = objective(logits, labels, torch.ones(8), teacher, .25)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    standard = torch.full((8,), 5)
    assert torch.allclose(objective(logits, standard, torch.ones(8), teacher, 1.),
                          objective(logits, standard, torch.ones(8), teacher, 0.))
    labels = np.array([int(r['label_id']) for r in rows])
    interval = paired_interval(rows, labels, labels)
    assert interval['lower_delta_pp'] == interval['upper_delta_pp'] == 0.
    print('PASS: class/person weighting, stable fusion, gradient flow, Standard excludes regional KD, paired interval')


def main(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.set_per_process_memory_fraction(.55, device=0)
    device = torch.device('cuda:0')

    def progress(stage, **extra):
        value = dict(status='running', stage=stage, pid=os.getpid(),
                     updated_at=datetime.datetime.now().astimezone().isoformat(),
                     elapsed_seconds=round(time.time() - started, 1))
        value.update(extra)
        write_json(out / 'pipeline_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    progress('verify_cache')
    manifests = BASE / 'manifests'
    rows, summary = verify_data(manifests)
    extraction = json.loads((out / 'extraction_status.json').read_text())
    if extraction['status'] != 'complete':
        raise RuntimeError('Feature extraction must complete first')
    cache = out / 'features'
    identity = json.loads((cache / 'identity.json').read_text())
    if identity['manifest_sha256'] != summary['manifest_sha256']:
        raise RuntimeError('Feature cache row identity mismatch')
    state = torch.load(REGIONAL, map_location='cpu')
    source_sha = hashlib.sha256(REGIONAL.read_bytes()).hexdigest()
    if identity['model']['adaptation_sha256'] != source_sha:
        raise RuntimeError('Feature cache uses another encoder')
    protocol = dict(labels=LABELS, seed=42, manifests=str(manifests),
                    manifest_sha256=summary['manifest_sha256'], regional_checkpoint=str(REGIONAL),
                    regional_checkpoint_sha256=source_sha, head_modes=MODES,
                    head_epochs=args.head_epochs, head_lr=5e-4, head_patience=8,
                    head_batch_size=1024, class_and_person_balanced_loss=True,
                    encoder_last_blocks=8, encoder_lr=2e-6, encoder_head_lr=2e-5,
                    encoder_epochs=args.adapt_epochs, encoder_batch_size=8, accumulation=4,
                    evaluation_batch_size=16,
                    warmup_fraction=.1, schedule='cosine', encoder_kd_weight=.25,
                    encoder_patience=2, audio_augmentation=False,
                    fusion='(1-alpha) * softmax(EAT six logits) + alpha * softmax(XLSR six logits)',
                    fusion_alpha_grid=ALPHAS, baseline_eligible=True,
                    selection='Validation Top1, then Macro F1; exact ties retain previous baseline',
                    test_policy='Save final selection before any new test inference',
                    test_limitation='Previously inspected development comparison set, not a new external test',
                    actual_microphone_evaluation='No verified dialect-labelled microphone set found',
                    source_data_limitations=summary['limitations'])
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Protocol changed; use another output directory')
    write_json(protocol_path, protocol)
    sources = out / 'source_snapshot'
    sources.mkdir(exist_ok=True)
    hashes = {}
    for filename in ['train_xlsr_six_class.py', 'extract_xlsr_six_class.py',
                     'train_dialect_comparison.py', 'train_standard_six_class.py', 'run_xlsr_six_class.sh']:
        path = Path(__file__).with_name(filename)
        shutil.copyfile(path, sources / filename)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out / 'source_sha256.json', hashes)

    indices = [identity['features'].index(n) for n in state['feature_names']]
    mu, sd = np.asarray(state['mu']), np.asarray(state['sd'])
    features, ys = {}, {}
    for split in ['train', 'valid']:
        if not np.load(cache / (split + '.ok.npy')).all():
            raise RuntimeError('Failed feature rows: ' + split)
        arr = np.load(cache / (split + '.npy'), mmap_mode='r')
        features[split] = (np.asarray(arr[:, indices, :], np.float32).reshape(len(arr), -1) - mu) / sd
        ys[split] = np.array([int(r['label_id']) for r in rows[split]])
    weights = balanced_weights(rows['train'])
    teacher = features['train'] @ state['head']['weight'].numpy().T + state['head']['bias'].numpy()
    eat = np.load(BASE / 'eat_large/valid_predictions.npz')
    xr_old = np.load(BASE / 'xlsr_cascade/xlsr_valid_logits.npz')
    valid_ids = np.array([r['row_id'] for r in rows['valid']])
    if not np.array_equal(eat['row_id'], valid_ids) or not np.array_equal(xr_old['row_id'], valid_ids):
        raise RuntimeError('Validation alignment failed')
    old_regions = (.5 * softmax(eat['logits'][:, :5]) + .5 * softmax(xr_old['logits'])).argmax(1)
    old_pred = np.where(eat['logits'].argmax(1) == 5, 5, old_regions)
    baseline_valid = metrics(rows['valid'], np.eye(6)[old_pred])
    candidates = {}
    histories = json.loads((out / 'head_history.json').read_text()) if (out / 'head_history.json').exists() else []

    def combinations(name, logits):
        result = []
        for alpha in ALPHAS:
            value = metrics(rows['valid'], fuse(eat['logits'], logits, alpha))
            result.append(dict(candidate=name, alpha=alpha, top1=value['top1'], macro_f1=value['macro_f1']))
        return result

    def record_candidate(name, path, logits, kind):
        candidates[name] = dict(path=str(path), kind=kind, logits=logits,
                                valid=metrics(rows['valid'], logits), combinations=combinations(name, logits))

    head = nn.Linear(len(mu), 6).to(device)

    def cached_logits(split):
        head.eval()
        with torch.no_grad():
            return np.concatenate([head(torch.from_numpy(features[split][at:at+2048]).to(device)).cpu().numpy()
                                   for at in range(0, len(features[split]), 2048)])

    def head_checkpoint(name, value):
        return dict(model='xlsr2b_six_class', labels=LABELS, labels_ko=KO,
                    source_encoder_checkpoint=str(REGIONAL), source_encoder_sha256=source_sha,
                    head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    mu=mu, sd=sd, feature_names=state['feature_names'],
                    manifest_sha256=summary['manifest_sha256'], candidate=name, valid=value)

    for mode in MODES:
        directory = out / mode
        directory.mkdir(exist_ok=True)
        head_path = directory / 'head.pt'
        if (directory / 'complete.json').exists():
            saved = torch.load(head_path, map_location='cpu')
            head.load_state_dict(saved['head'])
            record_candidate(mode, head_path, cached_logits('valid'), 'frozen')
            continue
        histories = [item for item in histories if item['mode'] != mode]
        with torch.no_grad():
            head.weight[:5].copy_(state['head']['weight'].to(device))
            head.bias[:5].copy_(state['head']['bias'].to(device))
            head.weight[5].zero_()
            head.bias[5] = float(state['head']['bias'].min()) - 5.
        optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-3)
        rng, best, stale = np.random.default_rng(42), None, 0
        for epoch in range(1, args.head_epochs + 1):
            head.train()
            order, loss_sum = rng.permutation(len(rows['train'])), 0.
            for at in range(0, len(order), 1024):
                idx = order[at:at+1024]
                logits = head(torch.from_numpy(features['train'][idx]).to(device))
                loss = objective(logits, torch.as_tensor(ys['train'][idx], device=device),
                                 torch.as_tensor(weights[idx], device=device),
                                 torch.as_tensor(teacher[idx], device=device),
                                 1. if mode == 'all_rows_with_regional_kl' else 0.)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite head loss')
                optimizer.zero_grad()
                loss.backward()
                if mode == 'preserved_regional_rows':
                    head.weight.grad[:5].zero_()
                    head.bias.grad[:5].zero_()
                optimizer.step()
                if mode == 'preserved_regional_rows':
                    with torch.no_grad():
                        head.weight[:5].copy_(state['head']['weight'].to(device))
                        head.bias[:5].copy_(state['head']['bias'].to(device))
                loss_sum += loss.item() * len(idx)
            logits = cached_logits('valid')
            value = metrics(rows['valid'], logits)
            fusion_value = max(combinations(mode, logits), key=rank)
            score = (*rank(fusion_value), *rank(value))
            if best is None or score > best:
                best, stale = score, 0
                save_torch(head_path, head_checkpoint(mode, value))
                predictions(directory / 'valid_predictions.npz', rows['valid'], logits)
            else:
                stale += 1
            histories.append(dict(stage='head', mode=mode, epoch=epoch, loss=loss_sum / len(order),
                                  valid_top1=value['top1'], best_fusion_top1=fusion_value['top1']))
            write_json(out / 'head_history.json', histories)
            progress('train_head', mode=mode, epoch=epoch, valid_top1=value['top1'],
                     fusion_valid_top1=fusion_value['top1'])
            if stale >= 8:
                break
        saved = torch.load(head_path, map_location='cpu')
        head.load_state_dict(saved['head'])
        record_candidate(mode, head_path, cached_logits('valid'), 'frozen')
        write_json(directory / 'complete.json', {'status': 'complete', 'selection': 'validation fusion then standalone'})

    initial_name = max(candidates, key=lambda n: (*rank(max(candidates[n]['combinations'], key=rank)),
                                                 *rank(candidates[n]['valid'])))
    write_json(out / 'frozen_selection.json', dict(candidate=initial_name,
               candidates={k: {'valid': v['valid'], 'combinations': v['combinations']} for k, v in candidates.items()}))
    progress('load_encoder_for_adaptation', initial_candidate=initial_name)
    encoder = engine.Encoder('xlsr2b', device)
    from fairseq.modules.checkpoint_activations import unwrap_checkpoint
    unwrap_checkpoint(encoder.model)
    params = dict(encoder.model.named_parameters())

    def restore(delta):
        with torch.no_grad():
            for name, value in delta.items():
                params[name].copy_(value.to(device))

    restore(state['encoder_delta'])
    encoder.info.update(adaptation_checkpoint=str(REGIONAL), adaptation_sha256=source_sha)
    head.load_state_dict(torch.load(candidates[initial_name]['path'], map_location='cpu')['head'])
    for block in encoder.model.encoder.layers[-8:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    adapted = {n: p for n, p in params.items() if p.requires_grad}
    if set(adapted) != set(state['encoder_delta']):
        raise RuntimeError('Trainable block set differs from the regional checkpoint')
    differentiable = inspect.unwrap(engine.Encoder.__call__)
    mu_t, sd_t = torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device)

    def forward(values, lengths):
        encoded = differentiable(encoder, values, lengths)[:, indices, :].reshape(len(values), -1).half().float()
        return head((encoded - mu_t) / sd_t)

    def evaluate(split):
        encoder.model.eval()
        head.eval()
        order = sorted(range(len(rows[split])), key=lambda i: float(rows[split][i]['duration_sec']))
        loader = DataLoader(Subset(engine.PreparedClips(rows[split], 'xlsr2b'), order), batch_size=16,
                            num_workers=4, collate_fn=engine.Collator('xlsr2b'), pin_memory=True)
        logits = np.full((len(order), 6), np.nan, np.float32)
        with torch.no_grad():
            for batch, (idx, values, lengths, errors) in enumerate(loader):
                if errors:
                    raise RuntimeError('Evaluation audio errors: ' + str(errors[:2]))
                result = forward(values, lengths)
                if not torch.isfinite(result).all():
                    raise RuntimeError('Nonfinite evaluation logits')
                logits[idx] = result.cpu().numpy()
                if batch % 50 == 0:
                    progress('evaluate_' + split, processed=min((batch + 1) * 16, len(order)), total=len(order))
        return logits

    optimizer = torch.optim.AdamW([{'params': list(adapted.values()), 'lr': 2e-6},
                                   {'params': head.parameters(), 'lr': 2e-5}], weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    train_params = list(adapted.values()) + list(head.parameters())
    batch_size, accumulation = 8, 4
    batches = math.ceil(len(rows['train']) / batch_size)
    total_steps = args.adapt_epochs * math.ceil(batches / accumulation)
    warmup_steps = max(1, int(total_steps * .1))
    best_score = rank(max(candidates[initial_name]['combinations'], key=rank))
    epoch_start, stale, step, actual_steps, history = 1, 0, 0, 0, []
    resume_path = out / 'adaptation_resume.pt'
    adapted_path = out / 'best_adapted.pt'
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu')
        restore(resume['encoder_delta'])
        head.load_state_dict(resume['head'])
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        epoch_start, stale, step = resume['epoch'] + 1, resume['stale'], resume['step']
        actual_steps, best_score, history = resume['actual_steps'], tuple(resume['best_score']), resume['history']
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state(resume['cuda_rng'], device=0)
        del resume
    anchor_name = max(adapted, key=lambda n: adapted[n].numel())
    anchor = adapted[anchor_name].detach().flatten()[:1024].clone()
    gradients_checked = False
    for epoch in range(epoch_start, args.adapt_epochs + 1):
        if stale >= 2:
            break
        encoder.model.eval()
        for block in encoder.model.encoder.layers[-8:]:
            block.train()
        head.train()
        loader = DataLoader(engine.PreparedClips(rows['train'], 'xlsr2b'), batch_size=batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(42 + epoch),
                            num_workers=4, collate_fn=engine.Collator('xlsr2b'), pin_memory=True)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen, epoch_started = 0., 0, time.time()
        for batch, (idx, values, lengths, errors) in enumerate(loader):
            if errors:
                raise RuntimeError('Training audio errors: ' + str(errors[:2]))
            logits = forward(values, lengths)
            loss = objective(logits, torch.as_tensor(ys['train'][idx], device=device),
                             torch.as_tensor(weights[idx], device=device),
                             torch.as_tensor(teacher[idx], device=device), .25)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite encoder loss')
            group_size = min(accumulation, len(loader) - batch // accumulation * accumulation)
            scaler.scale(loss / group_size).backward()
            if not gradients_checked:
                if any(p.grad is None for p in adapted.values()):
                    raise RuntimeError('Some encoder parameters are disconnected from gradients')
                gradients_checked = True
            if (batch + 1) % accumulation == 0 or batch + 1 == len(loader):
                step += 1
                warmup = min(1., step / warmup_steps)
                fraction = max(0., min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps)))
                factor = warmup * .5 * (1. + math.cos(math.pi * fraction))
                optimizer.param_groups[0]['lr'] = 2e-6 * factor
                optimizer.param_groups[1]['lr'] = 2e-5 * factor
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_params, 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                actual_steps += int(scaler.get_scale() >= old_scale)
                optimizer.zero_grad(set_to_none=True)
                if step >= 50 and actual_steps == 0:
                    raise RuntimeError('No successful optimizer updates')
            seen += len(idx)
            loss_sum += loss.item() * len(idx)
            if batch % 100 == 0:
                progress('adapt_encoder', epoch=epoch, processed=seen, total=len(rows['train']),
                         optimizer_steps=actual_steps, loss=loss_sum / seen,
                         clips_per_second=seen / max(time.time() - epoch_started, 1e-6),
                         encoder_weights_changed=not torch.equal(anchor, adapted[anchor_name].detach().flatten()[:1024]))
        if torch.equal(anchor, adapted[anchor_name].detach().flatten()[:1024]):
            raise RuntimeError('Encoder weights did not change')
        logits = evaluate('valid')
        value = metrics(rows['valid'], logits)
        fusion_value = max(combinations('adapted', logits), key=rank)
        score = rank(fusion_value)
        if score > best_score:
            best_score, stale = score, 0
            checkpoint = head_checkpoint('adapted', value)
            checkpoint.update(encoder_delta={n: p.detach().cpu().clone() for n, p in adapted.items()},
                              epoch=epoch, fusion_validation=fusion_value)
            save_torch(adapted_path, checkpoint)
            predictions(out / 'adapted_valid_predictions.npz', rows['valid'], logits)
            del checkpoint
        else:
            stale += 1
        history.append(dict(epoch=epoch, loss=loss_sum / seen, valid_top1=value['top1'],
                            fusion_validation=fusion_value, seconds=time.time() - epoch_started))
        write_json(out / 'adaptation_history.json', history)
        resume = dict(encoder_delta={n: p.detach().cpu().clone() for n, p in adapted.items()},
                      head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                      optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                      stale=stale, step=step, actual_steps=actual_steps, best_score=best_score,
                      history=history, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        save_torch(resume_path, resume)
        del resume
        progress('encoder_epoch_complete', epoch=epoch, valid_top1=value['top1'],
                 fusion_valid_top1=fusion_value['top1'], non_improving_epochs=stale)

    if adapted_path.exists():
        adapted_logits = np.load(out / 'adapted_valid_predictions.npz')['logits']
        record_candidate('adapted', adapted_path, adapted_logits, 'adapted')
    baseline = dict(candidate='previous_baseline', alpha=None, top1=baseline_valid['top1'],
                    macro_f1=baseline_valid['macro_f1'])
    grid = [baseline] + [v for candidate in candidates.values() for v in candidate['combinations']]
    selected = max(grid, key=lambda v: (*rank(v), v['candidate'] == 'previous_baseline'))
    # This record must exist before accessing held-out test features/predictions.
    write_json(out / 'selection.json', dict(selected=selected, grid=grid, selected_before_test=True,
               candidates={k: {key: val for key, val in v.items() if key != 'logits'} for k, v in candidates.items()}))
    progress('selection_complete', selected=selected)

    del optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    restore(state['encoder_delta'])
    encoder.model.eval()
    cache_args = argparse.Namespace(model='xlsr2b', out=out, manifests=manifests, batch_size=16, workers=4)
    test_cache, _ = engine.extract(cache_args, {'test': rows['test']}, encoder)
    if not np.load(test_cache / 'test.ok.npy').all():
        raise RuntimeError('Failed test features')
    arr = np.load(test_cache / 'test.npy', mmap_mode='r')
    test_features = (np.asarray(arr[:, indices, :], np.float32).reshape(len(arr), -1) - mu) / sd
    test_logits, candidate_test = {}, {}
    for name, value in candidates.items():
        checkpoint = torch.load(value['path'], map_location='cpu')
        if value['kind'] == 'adapted':
            restore(checkpoint['encoder_delta'])
            head.load_state_dict(checkpoint['head'])
            logits = evaluate('test')
        else:
            logits = test_features @ checkpoint['head']['weight'].numpy().T + checkpoint['head']['bias'].numpy()
        test_logits[name] = logits
        candidate_test[name] = metrics(rows['test'], logits)
        predictions(out / (name + '_test_predictions.npz'), rows['test'], logits)
        del checkpoint
    old_test = np.load(BASE / 'selected_predictions.npz')
    eat_test = np.load(BASE / 'eat_large/test_predictions.npz')
    test_ids = np.array([r['row_id'] for r in rows['test']])
    if not np.array_equal(old_test['row_id'], test_ids) or not np.array_equal(eat_test['row_id'], test_ids):
        raise RuntimeError('Test prediction alignment failed')
    if selected['candidate'] == 'previous_baseline':
        result_logits = np.eye(6, dtype=np.float32)[old_test['six_pred']]
    else:
        result_logits = fuse(eat_test['logits'], test_logits[selected['candidate']], selected['alpha'])
    selected_test = metrics(rows['test'], result_logits)
    baseline_test = metrics(rows['test'], np.eye(6)[old_test['six_pred']])
    predictions(out / 'selected_test_predictions.npz', rows['test'], result_logits)
    bundle = dict(labels=LABELS, selection=selected,
                  eat_checkpoint=str(BASE / 'eat_large/best_adapted.pt'),
                  xlsr_checkpoint=str(REGIONAL) if selected['candidate'] == 'previous_baseline'
                    else candidates[selected['candidate']]['path'],
                  source_xlsr_checkpoint=str(REGIONAL),
                  recipe='EAT Standard gate; 50:50 regional probabilities' if selected['candidate'] == 'previous_baseline'
                    else 'weighted six-class softmax probabilities',
                  deployment_changed=False)
    write_json(out / 'model_bundle.json', bundle)
    report = dict(completed_at=datetime.datetime.now().astimezone().isoformat(), protocol=protocol,
                  selected=selected, baseline_valid=baseline_valid, baseline_test=baseline_test,
                  selected_test=selected_test, candidate_test=candidate_test,
                  improvement_pp=100. * (selected_test['top1'] - baseline_test['top1']),
                  improvement_interval=paired_interval(rows['test'], old_test['six_pred'], result_logits.argmax(1)),
                  head_history=histories, adaptation_history=history,
                  total_training_seconds=time.time() - started,
                  extraction=extraction, model_bundle=bundle, data=summary,
                  limitations=['Existing test has been inspected in prior development experiments.',
                               'Standard code 01 is provisional and uses a different reading corpus.',
                               'No independent dialect-labelled booth microphone data were found.',
                               'No audio augmentation or measured serving latency in this experiment.'])
    write_json(out / 'report.json', report)
    with (out / 'per_class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['class', 'n', 'baseline_recall', 'selected_recall',
                                                    'selected_precision', 'selected_f1'])
        writer.writeheader()
        for label in LABELS:
            value = selected_test['per_class'][label]
            writer.writerow({'class': label, 'n': value['n'],
                             'baseline_recall': baseline_test['per_class'][label]['recall_top1'],
                             'selected_recall': value['recall_top1'], 'selected_precision': value['precision'],
                             'selected_f1': value['f1']})
    table = ''.join('<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td></tr>' %
                    (html.escape(KO[i]), 100 * baseline_test['per_class'][label]['recall_top1'],
                     100 * selected_test['per_class'][label]['recall_top1']) for i, label in enumerate(LABELS))
    document = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>XLSR 6클래스 실험</title>'
    document += '<style>body{font:17px sans-serif;max-width:1000px;margin:40px auto;line-height:1.7}td,th{padding:8px 20px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    document += '<h1>XLSR 6클래스 확장 실험</h1><p>북한말 제외. 검증 데이터로 선택 후 기존 테스트 6,000개에서 비교.</p>'
    document += '<p>기존 정확도 %.2f%% → 선택 모델 %.2f%% (%+.2f%%p)</p>' % (100 * baseline_test['top1'], 100 * selected_test['top1'], report['improvement_pp'])
    document += '<p>선택: %s / XLSR 비율 %s</p><table><tr><th>클래스</th><th>기존</th><th>선택 모델</th></tr>%s</table>' % (html.escape(selected['candidate']), str(selected['alpha']), table)
    document += '<h2>평가 한계</h2><p>기존에 열람한 개발 비교셋이며 새로운 외부 테스트가 아닙니다. 표준어 라벨 의미와 녹음 방식 편향은 미해결이며, 실제 부스 마이크 정확도는 측정하지 않았습니다.</p>'
    document += '<h2>실험 설정</h2><pre>' + html.escape(json.dumps(protocol, ensure_ascii=False, indent=2)) + '</pre></html>'
    (out / 'report.html').write_text(document)
    progress('complete', status='complete', baseline_test_top1=baseline_test['top1'],
             selected_test_top1=selected_test['top1'], improvement_pp=report['improvement_pp'], selected=selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'xlsr_six_class_20261014')
    parser.add_argument('--head-epochs', type=int, default=40)
    parser.add_argument('--adapt-epochs', type=int, default=3)
    parser.add_argument('--self-test', action='store_true')
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        try:
            main(arguments)
        except Exception as exc:
            arguments.out.mkdir(parents=True, exist_ok=True)
            write_json(arguments.out / 'pipeline_status.json', dict(status='failed',
                       error=type(exc).__name__ + ': ' + str(exc),
                       updated_at=datetime.datetime.now().astimezone().isoformat()))
            raise

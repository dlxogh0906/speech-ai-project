"""Compare six-class heads, then adapt EAT's last four blocks.

All six provider labels have person-disjoint validation/test recordings.
Validation selects every candidate; test is read for scoring only after
selection. The interpretation of 537 Dialect 01 is documented separately.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import gc
import hashlib
import html
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

spec = importlib.util.spec_from_file_location('dialect_encoders', Path(__file__).with_name('train_dialect_comparison.py'))
encoders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoders)
write_json = encoders.heads.write_json
LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard']
KO = ['강원', '경상', '전라', '제주', '충청', '표준어 후보(Dialect 01)']


def load_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def manifest_hash(root):
    h = hashlib.sha256()
    for split in ['train', 'valid', 'test']:
        h.update((root / (split + '.csv')).read_bytes())
    return h.hexdigest()


def metrics(rows, logits):
    y = np.array([int(r['label_id']) for r in rows])
    prediction = logits.argmax(1)
    p, r, f, n = precision_recall_fscore_support(y, prediction, labels=range(6), zero_division=0)
    person_correct = defaultdict(list)
    for row, correct in zip(rows, prediction == y):
        person_correct[(int(row['label_id']), row['person_id'])].append(bool(correct))
    return {'n': len(y), 'top1': float((prediction == y).mean()), 'macro_f1': float(f.mean()),
            'per_class': {LABELS[i]: {'name_ko': KO[i], 'n': int(n[i]), 'precision': float(p[i]),
                                    'recall_top1': float(r[i]), 'f1': float(f[i]),
                                    'people': sum(label == i for label, person in person_correct),
                                    'person_balanced_top1': float(np.mean([np.mean(v) for (label, person), v in person_correct.items() if label == i]))}
                          for i in range(6)},
            'confusion_labels': LABELS, 'confusion_matrix': confusion_matrix(y, prediction, labels=range(6)).tolist(),
            'original_five_region_top1': float((prediction[y < 5] == y[y < 5]).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, default=Path('/data/speech-ai/out/dialect/standard_six_class_20261013/manifests'))
    parser.add_argument('--adapted', type=Path, default=Path('/data/speech-ai/out/dialect/finetune_speakerid_20261012/eat_large/best_adapted.pt'))
    parser.add_argument('--out', type=Path, default=Path('/data/speech-ai/out/dialect/standard_six_class_20261013/eat_large'))
    parser.add_argument('--head-epochs', type=int, default=40)
    parser.add_argument('--adapt-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.model = 'eat_large'
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.20, device=0)
    device = torch.device('cuda:0')
    run = {'status': 'running', 'pid': os.getpid(), 'started_at': datetime.datetime.now().astimezone().isoformat()}

    def progress(stage, **values):
        run.update(stage=stage, updated_at=datetime.datetime.now().astimezone().isoformat(),
                   elapsed_seconds=round(time.time() - started, 1), **values)
        write_json(args.out / 'progress.json', run)
        print('[standard-six] ' + json.dumps(run, ensure_ascii=False), flush=True)

    try:
        summary = json.loads((args.manifests / 'summary.json').read_text())
        if not summary['source_person_disjoint_verified'] or summary['manifest_sha256'] != manifest_hash(args.manifests):
            raise RuntimeError('Prepared manifest identity is invalid')
        rows = {s: load_rows(args.manifests / (s + '.csv')) for s in ['train', 'valid', 'test']}
        people = {s: {r['person_id'] for r in v} for s, v in rows.items()}
        records = {s: {r['record_key'] for r in v} for s, v in rows.items()}
        for s, values in rows.items():
            if {int(r['label_id']) for r in values} != set(range(6)) or any(not r['person_id'] for r in values):
                raise RuntimeError('Missing class/person identity: ' + s)
            if any(r['source_dataset'] not in ['537', '71517', '71558'] for r in values):
                raise RuntimeError('Unexpected source corpus')
        for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
            if people[a] & people[b] or records[a] & records[b]:
                raise RuntimeError('Person/recording leakage')
        state = torch.load(args.adapted, map_location='cpu')
        if state['data_summary']['manifest_sha256'] != summary['original_regional_manifest_sha256']:
            raise RuntimeError('Regional backbone uses another person split')
        write_json(args.out / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        progress('load_encoder')
        encoder = encoders.Encoder('eat_large', device)
        params = dict(encoder.model.named_parameters())
        with torch.no_grad():
            for name, value in state['encoder_delta'].items():
                params[name].copy_(value.to(device))
        encoder.info.update(adaptation_checkpoint=str(args.adapted), adaptation_sha256=hashlib.sha256(args.adapted.read_bytes()).hexdigest())
        # Do not extract held-out test features until all selection is complete.
        selected = [encoder.names.index(n) for n in state['feature_names']]
        cache, resources = encoders.extract(args, {s: rows[s] for s in ['train', 'valid']}, encoder)
        arrays = {}
        for split in ['train', 'valid']:
            if not np.load(cache / (split + '.ok.npy')).all():
                raise RuntimeError('Resolve feature/audio failures first')
            v = np.load(cache / (split + '.npy'), mmap_mode='r')
            arrays[split] = np.asarray(v[:, selected, :], np.float32).reshape(len(v), -1)
        mu, sd = np.asarray(state['mu']), np.asarray(state['sd'])
        normalized = {s: (a - mu) / sd for s, a in arrays.items()}
        y = {s: np.array([int(r['label_id']) for r in v], dtype=np.int64) for s, v in rows.items()}
        person_count = Counter(r['person_id'] for r in rows['train'])
        label_people = {i: {r['person_id'] for r in rows['train'] if int(r['label_id']) == i} for i in range(6)}
        weights = np.array([1 / person_count[r['person_id']] / len(label_people[int(r['label_id'])]) for r in rows['train']], dtype=np.float32)
        weights /= weights.mean()
        teacher = normalized['train'] @ state['head']['weight'].numpy().T + state['head']['bias'].numpy()
        encoder.model.cpu(); torch.cuda.empty_cache()
        head = nn.Linear(len(mu), 6).to(device)
        rng = np.random.default_rng(args.seed)

        def cached_logits(split):
            head.eval()
            with torch.no_grad():
                return np.concatenate([head(torch.from_numpy(normalized[split][i:i+2048]).to(device)).cpu().numpy()
                                       for i in range(0, len(normalized[split]), 2048)])

        def score(value):
            return value['top1'], value['macro_f1']

        best_value, best_head, best_name = None, None, None
        history, candidate_valid = [], {}
        for mode in ['preserved_regional_rows', 'all_rows_with_regional_kl']:
            with torch.no_grad():
                head.weight[:5].copy_(state['head']['weight'].to(device)); head.bias[:5].copy_(state['head']['bias'].to(device))
                head.weight[5].zero_(); head.bias[5] = float(state['head']['bias'].min()) - 5
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
            mode_best = None
            for epoch in range(1, args.head_epochs + 1):
                head.train(); order = rng.permutation(len(y['train'])); total = 0.
                for at in range(0, len(order), 1024):
                    idx = order[at:at+1024]
                    logits = head(torch.from_numpy(normalized['train'][idx]).to(device))
                    loss_values = F.cross_entropy(logits, torch.as_tensor(y['train'][idx], device=device), reduction='none')
                    if mode == 'all_rows_with_regional_kl':
                        kd = F.kl_div(F.log_softmax(logits[:, :5] / 2, 1), F.softmax(torch.as_tensor(teacher[idx], device=device) / 2, 1), reduction='none').sum(1) * 4
                        loss_values = loss_values + kd * torch.as_tensor(y['train'][idx] < 5, device=device)
                    loss = (loss_values * torch.as_tensor(weights[idx], device=device)).mean()
                    optimizer.zero_grad(); loss.backward()
                    if mode == 'preserved_regional_rows':
                        head.weight.grad[:5].zero_(); head.bias.grad[:5].zero_()
                    optimizer.step()
                    if mode == 'preserved_regional_rows':
                        with torch.no_grad():
                            head.weight[:5].copy_(state['head']['weight'].to(device)); head.bias[:5].copy_(state['head']['bias'].to(device))
                    total += loss.item() * len(idx)
                valid = metrics(rows['valid'], cached_logits('valid'))
                item = {'stage': 'head', 'mode': mode, 'epoch': epoch, 'loss': total / len(order), 'valid_top1': valid['top1'], 'valid_macro_f1': valid['macro_f1']}
                history.append(item)
                if mode_best is None or score(valid) > score(mode_best):
                    mode_best = valid; candidate_valid[mode] = {'epoch': epoch, 'valid': valid}
                if best_value is None or score(valid) > score(best_value):
                    best_value, best_name = valid, mode + ':epoch' + str(epoch)
                    best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                if epoch == 1 or epoch % 5 == 0:
                    progress('train_head', mode=mode, epoch=epoch, loss=item['loss'], validation_top1=valid['top1'])
            write_json(args.out / 'history.json', history)
        head.load_state_dict(best_head)
        frozen_valid = best_value
        frozen_head = {k: v.clone() for k, v in best_head.items()}
        frozen_name = best_name
        torch.save({'head': frozen_head, 'mu': mu, 'sd': sd, 'labels': LABELS,
                    'base_adapted_encoder': str(args.adapted), 'data_summary': summary,
                    'selected_candidate': frozen_name, 'valid': frozen_valid}, args.out / 'best_frozen_head.pt')
        del optimizer
        encoder.model.to(device)
        for block in encoder.model.model.blocks[-4:]:
            for p in block.parameters(): p.requires_grad = True
        adapted = {n: p for n, p in encoder.model.named_parameters() if p.requires_grad}
        differentiable = inspect.unwrap(encoders.Encoder.__call__)
        mu_t, sd_t = torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device)

        def forward(values, lengths):
            features = differentiable(encoder, values, lengths)[:, selected, :].reshape(len(values), -1).half().float()
            if not torch.isfinite(features).all(): raise RuntimeError('Nonfinite encoder features')
            return head((features - mu_t) / sd_t)

        datasets = {s: encoders.PreparedClips(v, 'eat_large') for s, v in rows.items()}

        def evaluate(split):
            encoder.model.eval(); head.eval()
            order = sorted(range(len(rows[split])), key=lambda i: float(rows[split][i]['duration_sec']))
            loader = DataLoader(Subset(datasets[split], order), batch_size=args.batch_size, num_workers=args.workers, collate_fn=encoders.Collator('eat_large'), pin_memory=True)
            logits = np.full((len(order), 6), np.nan, np.float32)
            with torch.no_grad():
                for batch, (idx, values, lengths, errors) in enumerate(loader):
                    if errors: raise RuntimeError('Evaluation audio errors: ' + str(errors[:2]))
                    output = forward(values, lengths)
                    if not torch.isfinite(output).all(): raise RuntimeError('Invalid logits')
                    logits[idx] = output.cpu().numpy()
                    if batch % 100 == 0: progress('evaluate_' + split, evaluated=min((batch + 1) * args.batch_size, len(order)), evaluation_total=len(order))
            return metrics(rows[split], logits), logits

        checkpoint = args.out / 'best_adapted.pt'

        def save_best():
            torch.save({'labels': LABELS, 'labels_ko': KO, 'model': 'eat_large_standard_six',
                        'encoder_delta': {n: p.detach().cpu().clone() for n, p in adapted.items()},
                        'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                        'mu': mu, 'sd': sd, 'feature_names': state['feature_names'],
                        'data_summary': summary, 'selected_candidate': best_name, 'valid': best_value,
                        'source_encoder_metadata': encoder.info}, checkpoint.with_suffix('.tmp.pt'))
            os.replace(checkpoint.with_suffix('.tmp.pt'), checkpoint)

        save_best()  # Validation-selected frozen model is a fallback.
        optimizer = torch.optim.AdamW([{'params': list(adapted.values()), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-4}], weight_decay=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        grads_seen, steps, stale = None, 0, 0
        anchor_name = next(n for n, p in adapted.items() if p.numel() > 100000)
        anchor = adapted[anchor_name].detach().flatten()[:1024].clone()
        for epoch in range(1, args.adapt_epochs + 1):
            encoder.model.eval(); head.train()
            for block in encoder.model.model.blocks[-4:]: block.train()
            loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=args.workers, collate_fn=encoders.Collator('eat_large'), pin_memory=True)
            seen, loss_sum = 0, 0.
            for batch, (idx, values, lengths, errors) in enumerate(loader):
                if errors: raise RuntimeError('Training audio errors: ' + str(errors[:2]))
                logits = forward(values, lengths)
                ce = F.cross_entropy(logits, torch.as_tensor(y['train'][idx], device=device), reduction='none')
                kd = F.kl_div(F.log_softmax(logits[:, :5] / 2, 1), F.softmax(torch.as_tensor(teacher[idx], device=device) / 2, 1), reduction='none').sum(1) * 4
                loss = ((ce + kd * torch.as_tensor(y['train'][idx] < 5, device=device)) * torch.as_tensor(weights[idx], device=device)).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                if grads_seen is None:
                    grads_seen = sum(p.grad is not None for p in adapted.values())
                    if grads_seen != len(adapted): raise RuntimeError('Encoder blocks disconnected from gradients')
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(list(adapted.values()) + list(head.parameters()), 1.)
                scaler.step(optimizer); scaler.update(); steps += 1
                seen += len(idx); loss_sum += loss.item() * len(idx)
                if batch % 100 == 0: progress('adapt_encoder', epoch=epoch, epoch_seen=seen, epoch_total=len(rows['train']), loss=loss_sum / seen, encoder_parameters_with_grad=grads_seen)
            valid, logits = evaluate('valid')
            item = {'stage': 'encoder', 'epoch': epoch, 'loss': loss_sum / seen, 'valid_top1': valid['top1'], 'valid_macro_f1': valid['macro_f1']}
            history.append(item); write_json(args.out / 'history.json', history)
            if score(valid) > score(best_value):
                best_value, best_name = valid, 'last4_encoder:epoch' + str(epoch); save_best(); stale = 0
            else: stale += 1
            progress('encoder_epoch_complete', epoch=epoch, validation_top1=valid['top1'], selected_candidate=best_name)
            if stale >= 2: break
        weights_updated = not torch.equal(anchor, adapted[anchor_name].detach().flatten()[:1024])
        if not weights_updated: raise RuntimeError('Encoder optimization made no change')
        chosen = torch.load(checkpoint, map_location='cpu')
        with torch.no_grad():
            for name, p in chosen['encoder_delta'].items(): params[name].copy_(p.to(device))
        head.load_state_dict(chosen['head'])
        # First test inference occurs here, after all model selection.
        test, test_logits = evaluate('test')
        valid, valid_logits = evaluate('valid')
        np.savez_compressed(args.out / 'test_predictions.npz', logits=test_logits, labels=y['test'], row_id=np.array([r['row_id'] for r in rows['test']]))
        np.savez_compressed(args.out / 'valid_predictions.npz', logits=valid_logits, labels=y['valid'], row_id=np.array([r['row_id'] for r in rows['valid']]))
        if best_name == frozen_name:
            frozen_test = test
        else:
            with torch.no_grad():
                for name, value in state['encoder_delta'].items(): params[name].copy_(value.to(device))
            head.load_state_dict(frozen_head)
            frozen_test, frozen_logits = evaluate('test')
            np.savez_compressed(args.out / 'frozen_test_predictions.npz', logits=frozen_logits, labels=y['test'], row_id=np.array([r['row_id'] for r in rows['test']]))
        # Separate corpus check: these voices have IDs, but no verified dialect
        # tags. Never treat their output distribution as an accuracy figure.
        reference_path = Path('/data/speech-ai/out/dialect/standard_speaker_data_review_20261013/reading_reference_inventory.json')
        reference_report = None
        if reference_path.is_file():
            with torch.no_grad():
                for name, p in chosen['encoder_delta'].items(): params[name].copy_(p.to(device))
            head.load_state_dict(chosen['head']); head.eval(); encoder.model.eval()
            reference = json.loads(reference_path.read_text())
            ref_rows = reference['windows']
            loader = DataLoader(encoders.PreparedClips(ref_rows, 'eat_large'), batch_size=args.batch_size,
                                num_workers=args.workers, collate_fn=encoders.Collator('eat_large'))
            probabilities = np.full((len(ref_rows), 6), np.nan, np.float32)
            with torch.no_grad():
                for idx, values, lengths, errors in loader:
                    if errors: raise RuntimeError('Reference audio errors')
                    probabilities[idx] = F.softmax(forward(values, lengths), 1).cpu().numpy()
            reference_report = {'people': reference['people'], 'n': len(ref_rows),
                                'accuracy': None, 'predicted_labels': dict(Counter(LABELS[int(i)] for i in probabilities.argmax(1))),
                                'note': reference['note']}
            write_json(args.out / 'reading_reference_predictions.json', {
                'summary': reference_report,
                'windows': [{**r, 'predicted_label': LABELS[int(p.argmax())], 'probabilities': p.tolist()} for r, p in zip(ref_rows, probabilities)]})
        report = {'model': 'EAT six-class standard-dialect candidate', 'selected_candidate': best_name,
                  'test': test, 'valid': valid, 'frozen_candidates_valid': candidate_valid,
                  'best_frozen_candidate': frozen_name, 'best_frozen_valid': frozen_valid,
                  'best_frozen_test': frozen_test,
                  'reading_reference': reference_report,
                  'encoder_adaptation': {'last_blocks': 4, 'trainable_parameters': sum(p.numel() for p in adapted.values()),
                                         'parameter_tensors_with_grad': grads_seen, 'optimizer_steps': steps,
                                         'weights_updated': weights_updated},
                  'data_summary': summary, 'selection': 'six-class validation Top1, then Macro F1; test scored only after selection',
                  'elapsed_seconds': time.time() - started, 'checkpoint': str(checkpoint),
                  'interpretation': 'Provider-label classification. Standard-code semantics remain provisional; corpus/style bias and all-age real-user generalization are not resolved.'}
        write_json(args.out / 'report.json', report)
        with (args.out / 'per_class.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'name_ko', 'n', 'people', 'precision', 'recall_top1', 'f1', 'person_balanced_top1'])
            writer.writeheader(); writer.writerows({'class': k, **v} for k, v in test['per_class'].items())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matrix = np.array(test['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 7)); image = ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(6)); ax.set_xticklabels(LABELS, rotation=45, ha='right'); ax.set_yticks(range(6)); ax.set_yticklabels(LABELS)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True provider label')
        for i in range(6):
            for j in range(6): ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i,j]/matrix[i].sum() > .5 else 'black')
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(args.out / 'confusion_matrix.png', dpi=180); plt.close(fig)
        table = ''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%.4f</td></tr>' % (html.escape(v['name_ko']), v['n'], v['people'], v['recall_top1']*100, v['person_balanced_top1']*100, v['f1']) for v in test['per_class'].values())
        warnings = ''.join('<li>%s</li>' % html.escape(v) for v in summary['limitations'])
        page = '<!doctype html><meta charset="utf-8"><title>6클래스 실험</title><style>body{font-family:sans-serif;max-width:1000px;margin:40px auto;line-height:1.6}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%%}</style><h1>6클래스 EAT 실험 — 2026-10-13</h1><p>선택 모델: %s</p><p>테스트 Top-1 %.2f%% · Macro F1 %.4f</p><table><tr><th>클래스</th><th>녹음</th><th>화자</th><th>Top-1</th><th>화자 평균 Top-1</th><th>F1</th></tr>%s</table><h2>평가 조건</h2><ul>%s</ul><p>화자 ID는 제공 데이터의 식별자 기준이며 표준어 코드 해석은 설명서 확인 전입니다. 130번·71349번은 학습 및 평가에서 제외했습니다.</p><img src="confusion_matrix.png"><p><a href="per_class.csv">클래스별 CSV</a> · <a href="report.json">전체 수치 JSON</a></p>' % (html.escape(best_name),test['top1']*100,test['macro_f1'],table,warnings)
        (args.out / 'report.html').write_text(page)
        progress('complete', status='complete', test_top1=test['top1'], test_macro_f1=test['macro_f1'], report=str(args.out / 'report.html'))
    except Exception as exc:
        progress('failed', status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()

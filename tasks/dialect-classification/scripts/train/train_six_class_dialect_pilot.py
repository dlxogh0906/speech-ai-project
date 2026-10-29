"""Train a six-way head on the adapted EAT encoder with honest partial evaluation.

AI Hub 130 has no person IDs and is training-only. Fixed training epochs avoid
pretending the available five-region validation set selects six-class quality.
Reports explicitly omit full six-class accuracy and Seoul recall. The existing
five-class evaluation guard is neither modified nor bypassed for a certified
six-class result. This is a separate, clearly limited training experiment.
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
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

spec = importlib.util.spec_from_file_location('dialect_encoders', Path(__file__).with_name('train_dialect_comparison.py'))
encoders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoders)
write_json = encoders.heads.write_json
LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Seoul_Gyeonggi']
LABELS_KO = ['강원', '경상', '전라', '제주', '충청', '서울·경기(표준어 계열)']


def load_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def manifest_hash(root):
    h = hashlib.sha256()
    for split in ['train', 'valid', 'test']:
        h.update((root / (split + '.csv')).read_bytes())
    return h.hexdigest()


def diagnostic_metrics(y, predicted, classes=6):
    y, predicted = np.asarray(y), np.asarray(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=list(range(classes)), zero_division=0)
    per_class = {}
    for i in range(classes):
        per_class[LABELS[i]] = {'name_ko': LABELS_KO[i], 'n': int(support[i]),
                              'recall_top1': float(recall[i]) if support[i] else None,
                              'precision': float(precision[i]) if support[i] else None,
                              'f1': float(f1[i]) if support[i] else None}
    return {'n': len(y), 'top1_on_available_classes': float((y == predicted).mean()),
            'macro_f1_on_available_classes': float(f1[support > 0].mean()),
            'per_class': per_class,
            'confusion_labels': LABELS[:classes],
            'confusion_matrix': confusion_matrix(y, predicted, labels=list(range(classes))).tolist(),
            'predicted_seoul_on_regional_data_count': int((predicted == 5).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, default=Path('/data/speech-ai/out/dialect/six_class_20261012/manifests'))
    parser.add_argument('--original-manifests', type=Path, default=Path('/data/speech-ai/out/dialect/speakerid_20261011/manifests'))
    parser.add_argument('--adapted', type=Path, default=Path('/data/speech-ai/out/dialect/finetune_speakerid_20261012/eat_large/best_adapted.pt'))
    parser.add_argument('--out', type=Path, default=Path('/data/speech-ai/out/dialect/six_class_20261012/eat_large'))
    parser.add_argument('--north-korean-inventory', type=Path, default=Path('/data/speech-ai/out/dialect/north_korean_review_20261012/media_inventory.json'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preserve-regional-head', action='store_true')
    parser.add_argument('--feature-cache', type=Path, default=None)
    parser.add_argument('--adapt-epochs', type=int, default=0)
    args = parser.parse_args()
    args.model = 'eat_large'
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.25, device=0)
    summary = json.loads((args.manifests / 'summary.json').read_text())
    if summary['manifest_sha256'] != manifest_hash(args.manifests):
        raise RuntimeError('Prepared manifests have changed')
    rows = {split: load_rows(args.manifests / (split + '.csv')) for split in ['train', 'valid', 'test']}
    if {int(r['label_id']) for r in rows['train']} != set(range(6)):
        raise RuntimeError('Training data must include all six classes')
    if summary.get('seoul_evaluation_available') is not False:
        raise RuntimeError('This limited pilot expects no independent Seoul evaluation')
    for split in ['valid', 'test']:
        if any(not r['person_id'] or r['source_dataset'] == '130' or int(r['label_id']) == 5 for r in rows[split]):
            raise RuntimeError('Unverified 130 or Seoul data was put in a held-out diagnostic split')
    records = {split: {r['record_key'] for r in values} for split, values in rows.items()}
    people = {split: {r['person_id'] for r in values if r['person_id']} for split, values in rows.items()}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        if records[a] & records[b] or people[a] & people[b]:
            raise RuntimeError('Known person/recording overlap in prepared pilot manifests')
    state = torch.load(args.adapted, map_location='cpu')
    if state['labels'] != LABELS[:5] or state['data_summary']['manifest_sha256'] != manifest_hash(args.original_manifests):
        raise RuntimeError('Adapted encoder was not trained with the original person split')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.out / 'config.json', config)
    write_json(args.out / 'progress.json', {'status': 'running', 'stage': 'load_encoder', 'started_at': datetime.datetime.now().astimezone().isoformat(), 'pid': os.getpid()})
    north_items = []
    if args.north_korean_inventory.is_file():
        inventory = json.loads(args.north_korean_inventory.read_text())
        for item in inventory['clips']:
            duration = sf.info(item['audio_path']).duration
            for number, begin in enumerate(np.arange(0., duration, 8.)):
                length = min(8., duration - begin)
                if length < 3:
                    continue
                north_items.append({'row_id': item['clip'] + ':' + str(number), 'audio_path': item['audio_path'],
                                    'start_sec': float(begin), 'duration_sec': length, 'source_clip': item['clip'],
                                    'category': item['category']})
        if north_items:
            rows['north_reference'] = north_items
    if args.feature_cache:
        cache = args.feature_cache
        identity = json.loads((cache / 'identity.json').read_text())
        if identity['manifest_sha256'] != summary['manifest_sha256']:
            raise RuntimeError('Reused features have different manifest rows')
        if identity['model']['adaptation_sha256'] != hashlib.sha256(args.adapted.read_bytes()).hexdigest():
            raise RuntimeError('Reused features come from a different adapted encoder')
        if identity['implementation_sha256'] != hashlib.sha256(Path(encoders.__file__).read_bytes()).hexdigest():
            raise RuntimeError('Encoder feature implementation changed')
        if north_items and identity['model'].get('north_reference_inventory_sha256') != hashlib.sha256(args.north_korean_inventory.read_bytes()).hexdigest():
            raise RuntimeError('Reused North Korean reference inputs changed')
        selected = [identity['features'].index(name) for name in state['feature_names']]
        resources = {'reused_feature_cache': str(cache), 'identity': identity}
        for split in rows:
            if not (cache / (split + '.done.json')).is_file():
                raise RuntimeError('Incomplete feature cache: ' + split)
    else:
        encoder = encoders.Encoder('eat_large', device)
        params = dict(encoder.model.named_parameters())
        with torch.no_grad():
            for name, value in state['encoder_delta'].items():
                params[name].copy_(value.to(device))
        encoder.model.eval()
        encoder.info['adaptation_checkpoint'] = str(args.adapted)
        encoder.info['adaptation_sha256'] = hashlib.sha256(args.adapted.read_bytes()).hexdigest()
        if north_items:
            encoder.info['north_reference_inventory_sha256'] = hashlib.sha256(args.north_korean_inventory.read_bytes()).hexdigest()
        selected = [encoder.names.index(name) for name in state['feature_names']]
        cache, resources = encoders.extract(args, rows, encoder)
        del encoder, params
        gc.collect()
        torch.cuda.empty_cache()
    features = {split: np.load(cache / (split + '.npy'), mmap_mode='r') for split in rows}
    for split in rows:
        ok = np.load(cache / (split + '.ok.npy'))
        if not ok.all():
            raise RuntimeError('Audio feature failures must be resolved before reporting: ' + split)
    arrays = {split: np.asarray(value[:, selected, :], np.float32).reshape(len(value), -1) for split, value in features.items()}
    mu, sd = np.asarray(state['mu']), np.asarray(state['sd'])
    normalized = {split: (values - mu) / sd for split, values in arrays.items()}
    head = nn.Linear(len(mu), 6).to(device)
    with torch.no_grad():
        head.weight[:5].copy_(state['head']['weight'].to(device))
        head.bias[:5].copy_(state['head']['bias'].to(device))
        head.weight[5].zero_()
        head.bias[5] = state['head']['bias'].min().item() - 5
    regional_weight = head.weight[:5].detach().clone()
    regional_bias = head.bias[:5].detach().clone()
    y = {split: np.array([int(r['label_id']) for r in values], dtype=np.int64) for split, values in rows.items() if split != 'north_reference'}

    def predict(split):
        head.eval()
        with torch.no_grad():
            parts = [head(torch.from_numpy(normalized[split][at:at + 4096]).to(device)).cpu().numpy()
                     for at in range(0, len(normalized[split]), 4096)]
        return np.concatenate(parts)

    baseline = {}
    for split in ['valid', 'test']:
        logits = predict(split)[:, :5]
        baseline[split] = diagnostic_metrics(y[split], logits.argmax(1), classes=5)
    # Equal total weight per class. Verified people receive equal total weight
    # within their class/source. Identity-free 130 records have recording weight.
    units = [(r['source_dataset'], r['person_id'] or r['record_key']) for r in rows['train']]
    repeated = Counter(units)
    weights = np.array([1. / repeated[unit] for unit in units], dtype=np.float32)
    for label in range(6):
        mask = y['train'] == label
        weights[mask] /= weights[mask].sum()
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y['train']))
        head.train()
        loss_sum = 0.
        for at in range(0, len(order), 1024):
            indices = order[at:at + 1024]
            values = torch.from_numpy(normalized['train'][indices]).to(device)
            labels = torch.from_numpy(y['train'][indices]).to(device)
            sample_weights = torch.from_numpy(weights[indices]).to(device)
            loss = (F.cross_entropy(head(values), labels, reduction='none') * sample_weights).mean()
            optimizer.zero_grad()
            loss.backward()
            if args.preserve_regional_head:
                head.weight.grad[:5].zero_()
                head.bias.grad[:5].zero_()
            optimizer.step()
            if args.preserve_regional_head:
                # Restore the original rows exactly, including AdamW decay.
                with torch.no_grad():
                    head.weight[:5].copy_(regional_weight)
                    head.bias[:5].copy_(regional_bias)
            loss_sum += loss.item() * len(indices)
        info = {'epoch': epoch, 'training_loss': loss_sum / len(order)}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            info['old_region_validation_diagnostic'] = diagnostic_metrics(y['valid'], predict('valid').argmax(1))
            train_predictions = predict('train').argmax(1)
            info['training_fit_not_validation'] = diagnostic_metrics(y['train'], train_predictions)
        history.append(info)
        write_json(args.out / 'progress.json', {'status': 'running', 'stage': 'train_six_class_head', 'epoch': epoch,
                                              'epochs': args.epochs, 'elapsed_seconds': time.time() - started,
                                              'pid': os.getpid(), 'data_counts': summary['counts'], **info})
        print('[six-class-head] ' + json.dumps(info), flush=True)
    adaptation = None
    if args.adapt_epochs:
        # On identity-free 130 negatives, learn only "not Seoul". Do not force
        # their regional recording labels to rewrite the established classifier.
        # Distill the original five-region logits on the verified old training
        # source while adding the new class and adapting four encoder blocks.
        encoder = encoders.Encoder('eat_large', device)
        params = dict(encoder.model.named_parameters())
        with torch.no_grad():
            for name, value in state['encoder_delta'].items():
                params[name].copy_(value.to(device))
        for block in encoder.model.model.blocks[-4:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        adapted = {name: value for name, value in encoder.model.named_parameters() if value.requires_grad}
        encoder.model.eval()  # Deterministic teacher matching; gradients remain enabled.
        teacher = normalized['train'] @ state['head']['weight'].numpy().T + state['head']['bias'].numpy()
        old_source = np.array([r['source_dataset'] != '130' for r in rows['train']])
        weak_negative = (~old_source) & (y['train'] != 5)
        dataset = encoders.PreparedClips(rows['train'], 'eat_large')
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=args.workers,
                            collate_fn=encoders.Collator('eat_large'), pin_memory=True)
        optimizer = torch.optim.AdamW([{'params': list(adapted.values()), 'lr': 1e-5},
                                       {'params': head.parameters(), 'lr': 1e-4}], weight_decay=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        differentiable = inspect.unwrap(encoders.Encoder.__call__)
        mu_tensor, sd_tensor = torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device)
        updated_anchor = next(iter(adapted.values())).detach().flatten()[:1024].clone()
        gradient_count = None
        steps = 0
        for epoch in range(1, args.adapt_epochs + 1):
            loss_total, seen = 0., 0
            optimizer.zero_grad()
            for batch, (indices, values, lengths, errors) in enumerate(loader):
                if errors or not indices:
                    raise RuntimeError('Audio errors during encoder adaptation: ' + str(errors[:3]))
                features = differentiable(encoder, values, lengths)
                values = features[:, selected, :].reshape(len(indices), -1).half().float()
                logits = head((values - mu_tensor) / sd_tensor)
                targets = torch.as_tensor(y['train'][indices], device=device)
                sample_weights = torch.as_tensor(weights[indices], device=device)
                losses = F.cross_entropy(logits, targets, reduction='none')
                negative = torch.as_tensor(weak_negative[indices], device=device)
                binary_negative_loss = -torch.logsumexp(F.log_softmax(logits, dim=1)[:, :5], dim=1)
                losses = torch.where(negative, binary_negative_loss, losses)
                known = torch.as_tensor(old_source[indices], device=device)
                teacher_logits = torch.as_tensor(teacher[indices], device=device)
                kd = F.kl_div(F.log_softmax(logits[:, :5] / 2, dim=1),
                              F.softmax(teacher_logits / 2, dim=1), reduction='none').sum(1) * 4
                loss = ((losses + 2 * kd * known) * sample_weights).mean()
                scaler.scale(loss / 4).backward()
                if gradient_count is None:
                    gradient_count = sum(value.grad is not None for value in adapted.values())
                    if gradient_count != len(adapted):
                        raise RuntimeError('Encoder blocks are disconnected from autograd')
                if (batch + 1) % 4 == 0 or batch + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    if args.preserve_regional_head:
                        head.weight.grad[:5].zero_()
                        head.bias.grad[:5].zero_()
                    torch.nn.utils.clip_grad_norm_(list(adapted.values()) + list(head.parameters()), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    if args.preserve_regional_head:
                        with torch.no_grad():
                            head.weight[:5].copy_(regional_weight)
                            head.bias[:5].copy_(regional_bias)
                    optimizer.zero_grad()
                    steps += 1
                seen += len(indices)
                loss_total += loss.item() * len(indices)
                if batch % 50 == 0:
                    progress = {'status': 'running', 'stage': 'adapt_encoder', 'epoch': epoch,
                                'epoch_seen': seen, 'epoch_total': len(dataset),
                                'loss': loss_total / seen, 'encoder_parameters_with_grad': gradient_count,
                                'elapsed_seconds': time.time() - started}
                    write_json(args.out / 'progress.json', progress)
                    print('[six-class-encoder] ' + json.dumps(progress), flush=True)
        weights_updated = not torch.equal(updated_anchor, next(iter(adapted.values())).detach().flatten()[:1024])
        if not weights_updated:
            raise RuntimeError('Adaptation completed without changing encoder weights')
        adaptation = {'epochs': args.adapt_epochs, 'last_blocks': 4,
                      'trainable_encoder_parameters': sum(value.numel() for value in adapted.values()),
                      'encoder_parameters_with_grad': gradient_count, 'encoder_weights_updated': weights_updated,
                      'optimizer_steps': steps,
                      'loss': 'Six-class CE on original regions and Seoul; non-Seoul binary loss on other 130 samples; five-way teacher KL on original regional training recordings.'}
        adapted_path = args.out / 'six_class_encoder_delta.pt'
        torch.save({'encoder_delta': {name: value.detach().cpu() for name, value in adapted.items()},
                    'base_adaptation': str(args.adapted), 'data_summary': summary,
                    'adaptation': adaptation}, adapted_path)
        encoder.info['adaptation_checkpoint'] = str(args.adapted)
        encoder.info['adaptation_sha256'] = hashlib.sha256(args.adapted.read_bytes()).hexdigest()
        encoder.info['six_class_encoder_delta'] = str(adapted_path)
        encoder.info['six_class_encoder_delta_sha256'] = hashlib.sha256(adapted_path.read_bytes()).hexdigest()
        if north_items:
            encoder.info['north_reference_inventory_sha256'] = hashlib.sha256(args.north_korean_inventory.read_bytes()).hexdigest()
        del optimizer, scaler, loader, dataset, params, adapted, features, values, logits, loss, losses, kd
        gc.collect()
        torch.cuda.empty_cache()
        cache, resources = encoders.extract(args, rows, encoder)
        del encoder
        features = {split: np.load(cache / (split + '.npy'), mmap_mode='r') for split in rows}
        for split in rows:
            if not np.load(cache / (split + '.ok.npy')).all():
                raise RuntimeError('Adapted encoder feature failures: ' + split)
        arrays = {split: np.asarray(value[:, selected, :], np.float32).reshape(len(value), -1) for split, value in features.items()}
        normalized = {split: (value - mu) / sd for split, value in arrays.items()}
        gc.collect()
        torch.cuda.empty_cache()
    test_logits = predict('test')
    test = diagnostic_metrics(y['test'], test_logits.argmax(1))
    valid = diagnostic_metrics(y['valid'], predict('valid').argmax(1))
    training = diagnostic_metrics(y['train'], predict('train').argmax(1))
    regional_rows_unchanged = torch.equal(head.weight[:5], regional_weight) and torch.equal(head.bias[:5], regional_bias)
    if args.preserve_regional_head and not regional_rows_unchanged:
        raise RuntimeError('Regional head preservation failed')
    checkpoint = {'model': 'eat_large_six_class_pilot', 'labels': LABELS, 'labels_ko': LABELS_KO,
                  'head': {name: value.detach().cpu() for name, value in head.state_dict().items()},
                  'mu': mu, 'sd': sd, 'feature_names': state['feature_names'],
                  'adapted_encoder_checkpoint': str(args.adapted),
                  'adapted_encoder_sha256': hashlib.sha256(args.adapted.read_bytes()).hexdigest(),
                  'six_class_encoder_delta': str(args.out / 'six_class_encoder_delta.pt') if adaptation else None,
                  'data_summary': summary, 'epoch': args.epochs, 'config': config}
    torch.save(checkpoint, args.out / 'six_class_head.pt')
    np.savez(args.out / 'regional_test_predictions.npz', logits=test_logits, labels=y['test'], row_id=np.array([r['row_id'] for r in rows['test']]))
    north_results = []
    if north_items:
        logits = predict('north_reference')
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        for row, probability in zip(north_items, probabilities):
            best = np.argsort(probability)[::-1][:3]
            north_results.append({**row, 'out_of_model_taxonomy': True,
                                  'predictions': [{'label': LABELS[int(i)], 'score': float(probability[i])} for i in best]})
        write_json(args.out / 'north_korean_reference_predictions.json', {'note': 'The six-class model has no North Korean class. These are out-of-taxonomy responses, not North Korean classification accuracy.', 'windows': north_results})
    report = {'model': 'EAT adapted encoder + new six-way head', 'labels': LABELS, 'labels_ko': LABELS_KO,
              'encoder_frozen_in_this_stage': adaptation is None, 'encoder_previously_finetuned': True,
              'six_class_encoder_adaptation': adaptation,
              'preserve_original_regional_head': args.preserve_regional_head,
              'regional_head_rows_bitwise_unchanged': regional_rows_unchanged,
              'training_policy': 'Fixed 30 epochs by default; no claim that five-region validation selects six-class quality.',
              'six_class_test_top1': None, 'seoul_test_recall': None,
              'missing_evaluation': 'Independent identified Seoul/Gyeonggi speakers are not available.',
              'training_fit_not_generalization': training,
              'original_region_validation_diagnostic': valid,
              'original_region_test_diagnostic': test,
              'original_five_way_head_on_same_regional_windows': baseline,
              'data_summary': summary, 'config': config, 'resources': resources,
              'elapsed_seconds': time.time() - started,
              'north_korean_reference_windows': len(north_results),
              'limitations': summary['limitations'] + ['Cross-corpus physical speaker overlap between identity-free 130 and the old corpus cannot be independently ruled out.',
                                                     'Training-fit recall is not an estimate of standard-language performance.',
                                                     'No service deployment or NPU conversion is performed.']}
    write_json(args.out / 'report.json', report)
    write_json(args.out / 'history.json', history)
    body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>6클래스 방언 실험</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;line-height:1.65}table{width:100%;border-collapse:collapse}th,td{border:1px solid #ccd4dd;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}.note{background:#fff4d6;padding:16px}h2{margin-top:30px}</style>',
            '<h1>서울·경기를 추가한 6클래스 EAT 실험%s</h1>' % (' · 기존 5개 출력 고정' if args.preserve_regional_head else ''),
            '<p class="note">전체 6클래스 정확도와 서울·경기 정답률은 아직 측정할 수 없습니다. 130번 데이터는 화자 ID가 없어 학습에만 넣었으며, 아래 평가는 기존 5지역 자료에서만 수행한 진단입니다. 학습 자료를 다시 맞힌 비율은 실제 성능으로 보고하지 않습니다.</p>',
            '<h2>학습 구성</h2><table><tr><th>클래스</th><th>학습</th><th>검증 진단</th><th>테스트 진단</th></tr>']
    for en, ko in zip(LABELS, LABELS_KO):
        body.append('<tr><td>%s</td>%s</tr>' % (ko, ''.join('<td>%s</td>' % (summary['counts'][split].get(en, 0) or '미확보') for split in ['train', 'valid', 'test'])))
    before = baseline['test']['top1_on_available_classes']
    after = test['top1_on_available_classes']
    body.extend(['</table>', '<h2>기존 5지역 테스트 진단</h2>',
                 '<p>같은 %s개 구간에서 기존 5분류 헤드: %.2f%% / 새 6분류 헤드: %.2f%%. 다른 지역 음성을 서울·경기로 예측한 경우도 오답에 포함됩니다.</p>' % (test['n'], before * 100, after * 100),
                 '<table><tr><th>지역</th><th>테스트 구간</th><th>Top-1 정답률</th></tr>'])
    for label, values in test['per_class'].items():
        body.append('<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % (values['name_ko'], values['n'], '미측정' if values['recall_top1'] is None else '%.2f%%' % (values['recall_top1'] * 100)))
    body.extend(['</table><h2>북한 음성 자료</h2>',
                 '<p>영화 대화·방송 해설·구호 3개 영상의 %s개 구간을 추론했습니다. 북한 클래스가 아직 없으므로 이 결과는 기존 클래스에 어떻게 반응하는지를 보는 참고 자료입니다.</p>' % len(north_results),
                 '<h2>한계와 다음 단계</h2><ul>' + ''.join('<li>' + html.escape(v) + '</li>' for v in report['limitations']) + '</ul></html>'])
    (args.out / 'report.html').write_text('\n'.join(body))
    write_json(args.out / 'progress.json', {'status': 'complete', 'stage': 'six_class_pilot_complete',
                                          'report': str(args.out / 'report.html'), 'six_class_test_top1': None,
                                          'original_region_test_top1': after, 'elapsed_seconds': time.time() - started})
    print('[complete-six-class] ' + json.dumps({'regional_test_before': before, 'regional_test_after': after, 'six_class_test_top1': None}), flush=True)


if __name__ == '__main__':
    main()

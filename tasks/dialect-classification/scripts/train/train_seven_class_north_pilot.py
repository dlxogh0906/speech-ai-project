"""Add a cautious North Korean seventh option to the selected six-class model.

The seventh score is learned from the existing six logits and the three reviewed
course clips. Original six scores remain frozen. North clip leave-one-out is a
source-domain diagnostic, not speaker-disjoint validation or an accuracy claim.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard', 'North_Korean']
KO = ['강원', '경상', '전라', '제주', '충청', '표준어 후보', '북한말']


def north_rows(inventory):
    result = []
    for item in inventory['clips']:
        duration = sf.info(item['audio_path']).duration
        for number, begin in enumerate(np.arange(0., duration, 8.)):
            length = min(8., duration - begin)
            if length >= 3:
                result.append({'row_id': item['clip'] + ':' + str(number),
                               'source_clip': item['clip'], 'start_sec': float(begin),
                               'duration_sec': float(length)})
    return result


def six_logits(features, checkpoint):
    selected = [2, 3]  # layer15_mean, layer23_mean in the pinned feature cache
    vectors = np.asarray(features[:, selected, :], np.float32).reshape(len(features), -1)
    normalized = (vectors - np.asarray(checkpoint['mu'])) / np.asarray(checkpoint['sd'])
    head = checkpoint['head']
    return normalized @ head['weight'].numpy().T + head['bias'].numpy()


def fit_detector(negative, positive, valid_negative):
    scaler = StandardScaler().fit(negative)
    x = np.vstack([scaler.transform(negative), scaler.transform(positive)])
    y = np.r_[np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)]
    model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000,
                               solver='lbfgs', random_state=42).fit(x, y)
    valid_scores = model.decision_function(scaler.transform(valid_negative))
    threshold = float(np.quantile(valid_scores, 0.99, method='higher'))
    return scaler, model, threshold


def margin(scaler, model, threshold, logits):
    return model.decision_function(scaler.transform(logits)) - threshold


def predictions(scaler, model, threshold, logits):
    original = logits.argmax(axis=1)
    return np.where(margin(scaler, model, threshold, logits) > 0, 6, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--six-root', type=Path, default=Path('/data/speech-ai/out/dialect/standard_six_class_20261013/eat_large'))
    parser.add_argument('--north-cache', type=Path, default=Path('/data/speech-ai/out/dialect/six_class_20261012/eat_large/features'))
    parser.add_argument('--north-inventory', type=Path, default=Path('/data/speech-ai/out/dialect/north_korean_review_20261012/media_inventory.json'))
    parser.add_argument('--out', type=Path, default=Path('/data/speech-ai/out/dialect/seven_class_north_pilot_20261013'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    six = torch.load(args.six_root / 'best_adapted.pt', map_location='cpu')
    if six['labels'] != LABELS[:6] or six['selected_candidate'] != 'all_rows_with_regional_kl:epoch3':
        raise RuntimeError('Unexpected six-class checkpoint')
    original_identity = json.loads((args.six_root / 'features/identity.json').read_text())
    north_identity = json.loads((args.north_cache / 'identity.json').read_text())
    for key in ['adaptation_sha256', 'path']:
        if original_identity['model'][key] != north_identity['model'][key]:
            raise RuntimeError('Six and North features use different encoders')
    if original_identity['implementation_sha256'] != north_identity['implementation_sha256']:
        raise RuntimeError('Six and North features use different preprocessing')
    if original_identity['features'] != ['layer15_cls', 'layer23_cls', 'layer15_mean', 'layer23_mean']:
        raise RuntimeError('Unexpected feature layout')
    inventory_bytes = args.north_inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    rows = north_rows(inventory)
    north_status = json.loads((args.north_cache / 'north_reference.done.json').read_text())
    if north_status['rows'] != len(rows) or north_status['failed'] or not np.load(args.north_cache / 'north_reference.ok.npy').all():
        raise RuntimeError('North reference cache is incomplete or has another layout')
    train = six_logits(np.load(args.six_root / 'features/train.npy', mmap_mode='r'), six)
    valid = six_logits(np.load(args.six_root / 'features/valid.npy', mmap_mode='r'), six)
    north = six_logits(np.load(args.north_cache / 'north_reference.npy', mmap_mode='r'), six)
    test_file = np.load(args.six_root / 'test_predictions.npz')
    test = test_file['logits']
    test_labels = test_file['labels']
    valid_file = np.load(args.six_root / 'valid_predictions.npz')
    if not np.allclose(valid, valid_file['logits'], rtol=0.002, atol=0.02):
        raise RuntimeError('Cached six-class validation features do not reproduce selected checkpoint logits')
    fold_results = []
    source = np.array([r['source_clip'] for r in rows])
    for held in sorted(set(source)):
        fit = north[source != held]
        held_out = north[source == held]
        scaler, model, threshold = fit_detector(train, fit, valid)
        pred = predictions(scaler, model, threshold, held_out)
        fold_results.append({'held_out_source_clip': held, 'train_north_windows': len(fit),
                             'held_out_windows': len(held_out), 'predicted_north_windows': int((pred == 6).sum()),
                             'source_window_recall_diagnostic': float((pred == 6).mean())})
    scaler, model, threshold = fit_detector(train, north, valid)
    test_pred = predictions(scaler, model, threshold, test)
    baseline = test.argmax(1)
    valid_pred = predictions(scaler, model, threshold, valid)
    north_pred = predictions(scaler, model, threshold, north)
    per_class = []
    for i in range(6):
        mask = test_labels == i
        per_class.append({'label': LABELS[i], 'name_ko': KO[i], 'n': int(mask.sum()),
                          'baseline_recall': float((baseline[mask] == i).mean()),
                          'seven_option_recall': float((test_pred[mask] == i).mean()),
                          'false_north_count': int((test_pred[mask] == 6).sum())})
    result = {
        'status': 'pilot_complete', 'created_at': datetime.datetime.now().astimezone().isoformat(),
        'interpretation': 'Seven-option pilot with frozen original six-class logits and learned North detector. No independent North speaker validation exists; no seven-class accuracy is reported.',
        'labels': LABELS, 'labels_ko': KO,
        'north_source_clips': len(inventory['clips']), 'north_total_audio_sec': inventory['total_duration_sec'],
        'north_windows': len(north), 'north_speaker_ids_verified': False,
        'north_training_fit_recall_optimistic': float((north_pred == 6).mean()),
        'north_source_held_out_diagnostics': fold_results,
        'six_valid_false_north_count': int((valid_pred == 6).sum()),
        'six_valid_n': len(valid), 'six_test_false_north_count': int((test_pred == 6).sum()),
        'six_test_n': len(test), 'six_test_baseline_top1': float((baseline == test_labels).mean()),
        'six_test_seven_option_top1': float((test_pred == test_labels).mean()),
        'six_test_per_class': per_class,
        'threshold_policy': '99th percentile of North detector scores on 3,600 person-disjoint six-class validation utterances; fixed before six-class test evaluation.',
        'north_inventory_sha256': hashlib.sha256(inventory_bytes).hexdigest(),
        'six_checkpoint_sha256': hashlib.sha256((args.six_root / 'best_adapted.pt').read_bytes()).hexdigest(),
        'limitations': ['Only three source videos and about 143 seconds of North Korean reference audio.',
                        'North speaker identities and clip-specific voice origins are unverified.',
                        'Source-held-out clips have different speech styles and may share voices; diagnostic windows are not independent speakers.',
                        'Some source windows can contain background music, other voices or silence.',
                        'Six-class test accuracy describes provider corpora; app or web performance remains unknown.'],
    }
    (args.out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    checkpoint = {'labels': LABELS, 'labels_ko': KO, 'six_class_checkpoint': str(args.six_root / 'best_adapted.pt'),
                  'detector_scaler_mean': scaler.mean_, 'detector_scaler_scale': scaler.scale_,
                  'detector_coef': model.coef_, 'detector_intercept': model.intercept_,
                  'detector_threshold': threshold, 'north_inventory_sha256': result['north_inventory_sha256'],
                  'method': 'seven_option_frozen_six_logits_plus_north_detector'}
    torch.save(checkpoint, args.out / 'seven_option_head.pt')
    np.savez_compressed(args.out / 'diagnostic_predictions.npz', six_test_labels=test_labels,
                        six_test_original=baseline, six_test_seven_option=test_pred,
                        north_training_source=source, north_training_seven_option=north_pred)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

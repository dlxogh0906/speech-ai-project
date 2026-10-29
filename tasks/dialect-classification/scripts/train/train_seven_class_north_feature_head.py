"""Fit an exploratory North Korean seventh score on frozen EAT features.

Source-clip leave-one-out is diagnostic because no North speaker IDs or separate
provider recordings exist. The six-class backbone and its six scores stay fixed.
"""

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from train_seven_class_north_pilot import LABELS, KO, north_rows


SIX = Path('/data/speech-ai/out/dialect/standard_six_class_20261013/eat_large')
NORTH = Path('/data/speech-ai/out/dialect/six_class_20261012/eat_large/features')
INVENTORY = Path('/data/speech-ai/out/dialect/north_korean_review_20261012/media_inventory.json')
ROOT = Path('/data/speech-ai/out/dialect/seven_class_north_pilot_20261013')
OUT = ROOT / 'feature_head'


def features(path, checkpoint):
    data = np.load(path, mmap_mode='r')
    selected = [2, 3]
    vectors = np.asarray(data[:, selected, :], np.float32).reshape(len(data), -1)
    return (vectors - np.asarray(checkpoint['mu'])) / np.asarray(checkpoint['sd'])


def logits(features_, checkpoint):
    head = checkpoint['head']
    return features_ @ head['weight'].numpy().T + head['bias'].numpy()


def fit(negative, positive, valid_negative):
    x = np.vstack([negative, positive])
    y = np.r_[np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)]
    detector = LogisticRegression(C=0.01, class_weight='balanced', solver='liblinear',
                                  max_iter=1000, random_state=42).fit(x, y)
    threshold = float(np.quantile(detector.decision_function(valid_negative), 0.99, method='higher'))
    return detector, threshold


def classify(detector, threshold, features_, six_logits):
    margin = detector.decision_function(features_) - threshold
    return np.where(margin > 0, 6, six_logits.argmax(axis=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(SIX / 'best_adapted.pt', map_location='cpu')
    inventory = json.loads(INVENTORY.read_text())
    rows = north_rows(inventory)
    source = np.array([r['source_clip'] for r in rows])
    train = features(SIX / 'features/train.npy', checkpoint)
    valid = features(SIX / 'features/valid.npy', checkpoint)
    test = features(ROOT / 'test_extract/features/test.npy', checkpoint)
    north = features(NORTH / 'north_reference.npy', checkpoint)
    six = {name: logits(values, checkpoint) for name, values in [('train', train), ('valid', valid), ('test', test), ('north', north)]}
    test_file = np.load(SIX / 'test_predictions.npz')
    valid_file = np.load(SIX / 'valid_predictions.npz')
    if not np.allclose(six['valid'], valid_file['logits'], rtol=0.002, atol=0.02):
        raise RuntimeError('Validation features do not match selected six-class model')
    # Cached features are float16 and were extracted with a different batch size;
    # small logit differences are expected, but the original decisions must agree.
    if ((six['test'].argmax(1) == test_file['logits'].argmax(1)).mean() < 0.999
            or np.abs(six['test'] - test_file['logits']).mean() > 0.02):
        raise RuntimeError('Test features do not reproduce selected six-class decisions')
    if len(north) != len(rows) or not np.load(NORTH / 'north_reference.ok.npy').all():
        raise RuntimeError('North source windows differ from reviewed source cache')
    folds = []
    for held in sorted(set(source)):
        detector, threshold = fit(train, north[source != held], valid)
        predictions = classify(detector, threshold, north[source == held], six['north'][source == held])
        folds.append({'held_out_source_clip': held, 'training_north_windows': int((source != held).sum()),
                      'held_out_windows': int((source == held).sum()),
                      'predicted_north_windows': int((predictions == 6).sum())})
    detector, threshold = fit(train, north, valid)
    pred_valid = classify(detector, threshold, valid, six['valid'])
    pred_test = classify(detector, threshold, test, six['test'])
    pred_north = classify(detector, threshold, north, six['north'])
    y = test_file['labels']
    baseline = test_file['logits'].argmax(1)
    by_class = []
    for i in range(6):
        mask = y == i
        by_class.append({'label': LABELS[i], 'name_ko': KO[i], 'test_n': int(mask.sum()),
                         'baseline_recall': float((baseline[mask] == i).mean()),
                         'seven_option_recall': float((pred_test[mask] == i).mean()),
                         'false_north_count': int((pred_test[mask] == 6).sum())})
    result = {
        'status': 'pilot_complete', 'created_at': datetime.datetime.now().astimezone().isoformat(),
        'method': 'Frozen six-class EAT encoder and six-way head; new linear North detector on 2,048 EAT features; seven-option decision.',
        'labels': LABELS, 'labels_ko': KO,
        'north_source_clips': len(inventory['clips']), 'north_windows': len(rows),
        'north_total_audio_sec': inventory['total_duration_sec'], 'north_speaker_ids_verified': False,
        'north_source_held_out_diagnostics': folds,
        'north_training_fit_count_optimistic': int((pred_north == 6).sum()),
        'six_valid_false_north_count': int((pred_valid == 6).sum()), 'six_valid_n': len(valid),
        'six_test_false_north_count': int((pred_test == 6).sum()), 'six_test_n': len(test),
        'six_test_baseline_top1': float((baseline == y).mean()),
        'six_test_seven_option_top1': float((pred_test == y).mean()),
        'six_test_by_class': by_class,
        'threshold_policy': '99th percentile of North detector scores on 3,600 person-disjoint six-class validation utterances; fixed before six-class test evaluation.',
        'north_validation_accuracy': None, 'seven_class_test_accuracy': None,
        'reason_no_north_validation': 'Only three source videos, no verified North speaker IDs, and no independent North recording collection.',
        'limitations': ['The three North videos come from one lesson/site; model can learn production style or source artifacts.',
                        'Film dialogue, broadcast narration and slogan speech are not representative of ordinary North Korean conversation.',
                        'Eight-second windows from one video are correlated; leave-one-video-out checks source transfer only.',
                        'Some windows contain background sound, other voices or silence.',
                        'No evidence yet for app or web microphone generalization.'],
        'six_checkpoint_sha256': hashlib.sha256((SIX / 'best_adapted.pt').read_bytes()).hexdigest(),
        'north_inventory_sha256': hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
    }
    (OUT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    torch.save({'labels': LABELS, 'labels_ko': KO, 'method': result['method'],
                'six_checkpoint_path': str(SIX / 'best_adapted.pt'),
                'six_checkpoint_sha256': result['six_checkpoint_sha256'],
                'north_coef': detector.coef_.copy(), 'north_intercept': detector.intercept_.copy(),
                'north_threshold': threshold, 'feature_names': checkpoint['feature_names'],
                'feature_mu': checkpoint['mu'], 'feature_sd': checkpoint['sd']}, OUT / 'seven_option_head.pt')
    np.savez_compressed(OUT / 'diagnostic_predictions.npz', six_test_labels=y,
                        six_test_pred=pred_test, north_source=source, north_pred=pred_north)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

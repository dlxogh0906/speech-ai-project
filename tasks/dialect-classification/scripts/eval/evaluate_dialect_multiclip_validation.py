"""Validation-only, paired multi-recording experiment with fixed fusion rules.

No test predictions, fitting, threshold searches, or seed searches are used.
The existing validation set has already informed model selection: results here
are exploratory, not a new final-test accuracy claim.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('/data/speech-ai/out/dialect/expanded_six_class_20261013')
NEW = Path('/data/speech-ai/out/dialect/xlsr_six_class_20261014')
LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard']
KO = ['강원', '경상', '전라', '제주', '충청', '표준어 후보']


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def softmax(x):
    x = np.asarray(x, np.float64)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    return x / x.sum(axis=1, keepdims=True)


def metrics(items):
    result = {}
    for label, name in enumerate(LABELS):
        subset = [r for r in items if r['label'] == label]
        correct = sum(r['prediction'] == label for r in subset)
        result[name] = {'name_ko': KO[label], 'people': len(subset), 'correct': correct,
                        'accuracy': correct / len(subset) if subset else None}
    return {'people': len(items), 'correct': sum(r['prediction'] == r['label'] for r in items),
            'accuracy': sum(r['prediction'] == r['label'] for r in items) / len(items),
            'macro_class_accuracy': float(np.mean([v['accuracy'] for v in result.values()])),
            'regional_five_macro_accuracy': float(np.mean([result[k]['accuracy'] for k in LABELS[:5]])),
            'mean_input_seconds': float(np.mean([r['seconds'] for r in items])), 'per_class': result}


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    source_files = [BASE / 'manifests/valid.csv', BASE / 'eat_large/valid_predictions.npz',
                    BASE / 'xlsr_cascade/xlsr_valid_logits.npz', NEW / 'all_rows_ce/valid_predictions.npz']
    protocol = {'created_at': datetime.now().astimezone().isoformat(), 'split': 'existing_validation_only',
                'test_predictions_read': False, 'seed': 42,
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                'grouping': 'One outcome per person. Select distinct recordings by SHA256(42:person_id:row_id).',
                'cohorts': {'minimum_3_recordings': [1, 3], 'minimum_5_recordings': [1, 3, 5]},
                'fusion': {'existing_cascade': 'Mean EAT probabilities gate Standard; otherwise mean 0.5*EAT-region + 0.5*XLSR-region probabilities.',
                           'new_six_class': 'Mean probabilities of fixed 0.5*EAT-six + 0.5*XLSR-six fusion.'},
                'model_or_seed_search': False, 'unseen_final_test_claim': False,
                'limitations': ['Same existing validation already used in model development.',
                               'Eligibility excludes speakers with too few distinct recordings; their count is reported.',
                               'Separate recordings, not a continuous booth utterance or a live endpoint latency test.',
                               'Class counts differ, so equal-class macro accuracy is also reported.',
                               'Source ID independence and provisional Standard label limitations still apply.']}
    protocol_path = args.out / 'multiclip_protocol.json'
    assert not protocol_path.exists(), 'Use a fresh output directory; do not overwrite a selected experiment'
    write(protocol_path, protocol)
    with source_files[0].open() as stream:
        rows = list(csv.DictReader(stream))
    ids = np.array([r['row_id'] for r in rows])
    loaded = [np.load(p) for p in source_files[1:]]
    for value in loaded:
        assert np.array_equal(ids, value['row_id'])
    eat, regional, six = [value['logits'] for value in loaded]
    pe = softmax(eat)
    pr = .5 * softmax(eat[:, :5]) + .5 * softmax(regional)
    pn = .5 * pe + .5 * softmax(six)

    def classify(indices, model):
        if model == 'new_six_class':
            return int(pn[indices].mean(axis=0).argmax())
        return 5 if pe[indices].mean(axis=0).argmax() == 5 else int(pr[indices].mean(axis=0).argmax())

    labels = np.array([int(r['label_id']) for r in rows])
    all_single = {}
    for model in ['existing_cascade', 'new_six_class']:
        pred = np.array([classify([i], model) for i in range(len(rows))])
        all_single[model] = {'recordings': len(rows), 'accuracy': float(np.mean(pred == labels))}
    assert abs(all_single['existing_cascade']['accuracy'] - 3470 / 3600) < 1e-10
    assert abs(all_single['new_six_class']['accuracy'] - 3482 / 3600) < 1e-10
    people = defaultdict(list)
    for index, row in enumerate(rows):
        people[row['person_id']].append(index)
    ordered = {}
    for person, indices in people.items():
        assert len({rows[i]['label_id'] for i in indices}) == 1
        used, selected = set(), []
        for i in sorted(indices, key=lambda j: digest('42:' + person + ':' + rows[j]['row_id'])):
            key = rows[i]['record_key']
            if key not in used:
                used.add(key)
                selected.append(i)
        ordered[person] = selected
    report = {'protocol': protocol, 'all_single_validation_reproduction': all_single, 'cohorts': {}}
    predictions = []
    for minimum, amounts in [(3, [1, 3]), (5, [1, 3, 5])]:
        eligible = sorted(p for p, indices in ordered.items() if len(indices) >= minimum)
        name = 'minimum_{}_recordings'.format(minimum)
        cohort = {'people': len(eligible), 'excluded_people': len(people) - len(eligible), 'models': {}}
        for model in ['existing_cascade', 'new_six_class']:
            cohort['models'][model] = {}
            for n in amounts:
                outcomes = []
                for person in eligible:
                    indices = ordered[person][:n]
                    outcome = {'cohort': name, 'model': model, 'recordings': n, 'person_id': person,
                               'label': int(rows[indices[0]]['label_id']), 'prediction': classify(indices, model),
                               'seconds': sum(float(rows[i]['duration_sec']) for i in indices),
                               'row_ids': [rows[i]['row_id'] for i in indices]}
                    outcomes.append(outcome)
                cohort['models'][model][str(n)] = metrics(outcomes)
                predictions.extend(outcomes)
        report['cohorts'][name] = cohort
    report['completed_at'] = datetime.now().astimezone().isoformat()
    write(args.out / 'multiclip_validation.json', report)
    write(args.out / 'multiclip_predictions.json', predictions)
    with (args.out / 'multiclip_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cohort', 'model', 'recordings_per_person', 'people', 'correct', 'accuracy', 'macro_class_accuracy', 'regional_five_macro_accuracy', 'mean_input_seconds'])
        for name, cohort in report['cohorts'].items():
            for model, modes in cohort['models'].items():
                for n, value in modes.items():
                    writer.writerow([name, model, n] + [value[k] for k in ['people', 'correct', 'accuracy', 'macro_class_accuracy', 'regional_five_macro_accuracy', 'mean_input_seconds']])
                    print(json.dumps({'cohort': name, 'model': model, 'clips': n,
                                      **{k: v for k, v in value.items() if k != 'per_class'}}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())

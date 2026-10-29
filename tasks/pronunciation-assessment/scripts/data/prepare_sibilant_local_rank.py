"""TRAIN-only actual-C versus flipped-C sequence-rank targets.

Retain existing acoustic inputs, trusted phone sequences, regions and sampling
plan. Rank only an observed, uniquely labelled actual S/SS token in that region.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import json
import numpy as np
import eval_sibilant_directional as service
from audit_sibilant_vowel_binding import crop_actual_indices, stream
from mdd.sibilant_transition import PAIR


def rank_target(label, target, tag, actual, vocab):
    if target is None:
        return None, 'CTC_region_unavailable'
    if label['alignment_ambiguous'] or label.get('a') is None or label['actual'] not in PAIR:
        return None, 'not_unique_actual_S_SS'
    indices = crop_actual_indices(tag, actual, target['TAG_indices'])
    if [actual[i] for i in indices] != target['tokens']:
        raise ValueError('trusted target sequence differs')
    if label['a'] not in indices:
        return None, 'actual_C_outside_loss_region'
    position = indices.index(label['a'])
    token = label['actual']
    if target['tokens'][position] != token or target['ids'][position] != vocab[token]:
        raise ValueError('actual C index differs')
    alternate = PAIR[token]
    cf = list(target['ids'])
    cf[position] = vocab[alternate]
    minimum = len(cf) + sum(a == b for a, b in zip(cf, cf[1:]))
    if target['roi'][1] - target['roi'][0] < minimum:
        return None, 'counterfactual_CTC_path_impossible'
    return {'position': position, 'actual_id': vocab[token], 'alternate_id': vocab[alternate],
            'actual': token, 'alternate': alternate}, 'actual_pair_supported'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen input changed: ' + path)
    cache = Path(p['cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only rank targets')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    features = service.read_jsonl(cache / 'data.features.jsonl')
    mapped = [z for z in stream(cache / 'data.labels.jsonl') if z['feature_index'] is not None]
    labels = {z['feature_index']: z for z in mapped}
    if len(labels) != len(features) or len(mapped) != len(features):
        raise ValueError('unique preserved GT site required')
    records = {z['row']['uid']: z['row'] for z in stream(cache / 'data.responses.jsonl')}
    annotations = {z['stem']: z for z in stream(p['annotations']) if z['stem'] in records}
    targets = service.read_jsonl(p['targets'])
    y = np.load(cache / 'targets.npz')['y']
    vocab = json.loads((Path(p['checkpoint']) / 'jamo_vocab.json').read_text())['vocab']
    counts, rows = collections.Counter(), []
    for i, f in enumerate(features):
        if y[i] not in (0, 1):
            value, state = None, 'outside_preserved_S_SS_class_support'
        else:
            value, state = rank_target(labels[i], targets[i]['target'], annotations[f['uid']]['tag_phones'],
                                       records[f['uid']]['actual'], vocab)
        counts[state] += 1
        if value is not None:
            counts['eligible_label/' + labels[i]['label_kind']] += 1
            counts['eligible_actual/' + value['actual']] += 1
        rows.append({'feature_index': i, 'uid': f['uid'], 'rank': value, 'state': state})
    with (out / 'rank_targets.jsonl').open('x') as f:
        for row in rows:
            service.write_line(f, row)
    report = {'scope': p['scope'], 'counts': dict(counts), 'all_inputs': len(rows),
              'acoustic_or_CTC_targets_modified': False, 'inference_TAG_input': False,
              'rank_targets_sha256': service.sha(out / 'rank_targets.jsonl')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

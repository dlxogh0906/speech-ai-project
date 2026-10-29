"""Merge disjoint runtime TRAIN queries while retaining all unknown targets."""
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
import json
import numpy as np
import eval_sibilant_directional as service


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first', required=True)
    ap.add_argument('--second', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sources, out = [Path(a.first), Path(a.second)], Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    data, targets, features, queries = [], [], [], []
    labels, responses, keys = {}, {}, set()
    input_sha = {}
    for source in sources:
        report = json.loads((source / 'report.json').read_text())
        if report['partition'] != 'train' or not (source / 'finished.json').exists():
            raise ValueError('completed TRAIN cache required')
        block = dict(np.load(source / 'data.npz'))
        target = dict(np.load(source / 'targets.npz'))
        if set(block) != {'x', 'relative_vowel', 'lengths'} or set(target) != {'y', 'phase'}:
            raise ValueError('typed TRAIN coordinates/targets required')
        offset = len(features)
        fs = service.read_jsonl(source / 'data.features.jsonl')
        if len(fs) != len(target['y']) or len(fs) != len(block['x']):
            raise ValueError('TRAIN feature/target size mismatch')
        for i, f in enumerate(fs):
            key = (f['uid'], f['q_runtime'])
            if key in keys or f['feature_index'] != i:
                raise ValueError('overlapping or unordered runtime query')
            keys.add(key)
            f['feature_index'] += offset
            features.append(f)
        for f in service.read_jsonl(source / 'data.queries.jsonl'):
            if f['feature_index'] is not None:
                f['feature_index'] += offset
            queries.append(f)
        for label in service.read_jsonl(source / 'data.labels.jsonl'):
            if label['feature_index'] is not None:
                label['feature_index'] += offset
            key = (label['uid'], label['q'])
            if key in labels:
                old = labels[key]
                if ({k: v for k, v in old.items() if k != 'feature_index'}
                        != {k: v for k, v in label.items() if k != 'feature_index'}):
                    raise ValueError('GT identity/prior view differs across TRAIN caches')
                if old['feature_index'] is not None and label['feature_index'] is not None:
                    raise ValueError('GT maps to two observed runtime features')
                if label['feature_index'] is not None:
                    labels[key] = label
            else:
                labels[key] = label
        for record in service.read_jsonl(source / 'data.responses.jsonl'):
            uid = record['row']['uid']
            if uid in responses:
                old = responses[uid]
                if ({k: v for k, v in old.items() if k != 'presence_queries'}
                        != {k: v for k, v in record.items() if k != 'presence_queries'}):
                    raise ValueError('same TRAIN uid has different baseline record')
                old['presence_queries'] = sorted(set(old['presence_queries']) | set(record['presence_queries']))
            else:
                responses[uid] = record
        data.append(block)
        targets.append(target)
        for f in source.glob('data.*'):
            input_sha[str(f)] = service.sha(f)
        input_sha[str(source / 'targets.npz')] = service.sha(source / 'targets.npz')
    merged = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
    merged_targets = {k: np.concatenate([d[k] for d in targets]) for k in targets[0]}
    np.savez(out / 'data.npz', **merged)
    np.savez(out / 'targets.npz', **merged_targets)
    for name, rows in [('features', features), ('queries', queries), ('labels', labels.values()),
                       ('responses', responses.values())]:
        with (out / ('data.' + name + '.jsonl')).open('x') as stream:
            for row in rows:
                service.write_line(stream, row)
    warning_normals = [z for z in labels.values() if z['label_kind'] == 'correct'
                      and z['feature_index'] is not None and features[z['feature_index']]['baseline_warning']]
    report = {'scope': 'TRAIN-only runtime-warning exposure expansion; all old and new observations/unknown targets retained.',
              'partition': 'train', 'schema': report['schema'], 'features': len(features),
              'all_label_sites': len(labels), 'utterances': len(responses),
              'class_targets': {str(k): int(sum(merged_targets['y'] == k)) for k in [-100, 0, 1, 2, 3, 4]},
              'observed_normal_warnings': len(warning_normals),
              'supported_normal_warnings': int(sum(merged_targets['y'][z['feature_index']] >= 0 for z in warning_normals)),
              'sources_sha256': input_sha, 'runtime_TAG_inputs': False,
              'sha256': {f.name: service.sha(f) for f in out.glob('data.*')},
              'targets_sha256': service.sha(out / 'targets.npz')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps({k: report[k] for k in ['features', 'all_label_sites', 'utterances', 'class_targets',
                                           'observed_normal_warnings', 'supported_normal_warnings']}), flush=True)


if __name__ == '__main__':
    main()

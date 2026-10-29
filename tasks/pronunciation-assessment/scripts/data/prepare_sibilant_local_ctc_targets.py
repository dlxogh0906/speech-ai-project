"""Prepare TRAIN phone-sequence loss regions without changing input crops."""
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
from mdd.sibilant_local_ctc_targets import target_region


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    cache = Path(p['cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only target preparation')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    wanted = {f['uid'] for f in features}
    annotations = {}
    with open(p['annotations']) as stream:
        for line in stream:
            z = json.loads(line)
            if z['stem'] in wanted:
                annotations[z['stem']] = z
    records = {z['row']['uid']: z['row'] for z in service.read_jsonl(cache / 'data.responses.jsonl')}
    vocab = json.loads((Path(p['checkpoint']) / 'jamo_vocab.json').read_text())['vocab']
    y = np.load(cache / 'targets.npz')['y']
    counts = collections.Counter()
    rows = []
    for i, f in enumerate(features):
        target, state = target_region(f, annotations[f['uid']]['tag_phones'], records[f['uid']]['actual'], vocab)
        rows.append({'feature_index': i, 'uid': f['uid'], 'state': state, 'target': target})
        counts[state] += 1
        counts['class_' + str(int(y[i])) + '/' + state] += 1
    with (out / 'targets.jsonl').open('x') as stream:
        for row in rows:
            service.write_line(stream, row)
    report = {'scope': p['scope'], 'counts': dict(counts), 'all_inputs': len(rows),
              'runtime_input_modified': False, 'inference_TAG_input': False,
              'targets_sha256': service.sha(out / 'targets.jsonl')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

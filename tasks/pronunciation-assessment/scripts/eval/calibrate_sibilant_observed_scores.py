"""Apply the fixed full-CAL policy search to frozen observed-class scores."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import json
import numpy as np
import eval_sibilant_directional as service
from eval_sibilant_presence_screen import calibrate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    cache = Path(p['cache'])
    if json.loads((cache / 'report.json').read_text())['partition'] != 'cal':
        raise ValueError('threshold fitting requires a declared CAL cache')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    labels = service.read_jsonl(cache / 'data.labels.jsonl')
    scores = np.load(p['scores'])
    if len(labels) != p['expected_sites'] or len(scores['probabilities']) != len(features):
        raise ValueError('full denominator or score identity mismatch')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    result = calibrate(features, labels, scores['probabilities'], scores['available'])
    report = {'scope': p['scope'], 'deployment': False, 'all_sites': len(labels), 'all_inputs': len(features),
              'arms': {p['name']: result}}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

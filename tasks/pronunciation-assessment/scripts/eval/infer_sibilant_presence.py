"""Score a frozen V-reference artifact; never open TAG labels or fit policy."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import json
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_vowel_presence import model, scores


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
    data = dict(np.load(Path(p['cache']) / 'data.npz'))
    torch.set_num_threads(2)
    encoder, head = model(p, torch.load(p['artifact'], map_location='cpu'))
    encoder.eval()
    head.eval()
    torch.cuda.reset_peak_memory_stats()
    ps, av, sec = scores(encoder, head, data, batch=p['batch'])
    np.savez(out / 'scores.npz', probabilities=ps, available=av)
    report = {'scope': 'Frozen artifact, no TAG label input or threshold fitting; elapsed includes GPU work and transfer.',
              'inputs': len(ps), 'available': int(av.sum()), 'batch': p['batch'], 'seconds': sec,
              'peak_GPU_bytes': torch.cuda.max_memory_allocated(), 'deployment': False,
              'scores_sha256': service.sha(out / 'scores.npz')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

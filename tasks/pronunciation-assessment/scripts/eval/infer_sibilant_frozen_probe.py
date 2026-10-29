"""Score a bound frozen BERT representation with its TRAIN-fitted probe."""
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
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_frozen_probe import scores
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_vowel_presence import CLASSES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    rep = Path(p['representation'])
    source = json.loads((rep / 'report.json').read_text())
    binding = {
        'backbone_key': 'service_bert:' + source['backbone_sha256'] + ':h24:' + source['dtype'],
        'feature_schema': source['feature_schema'],
        'coordinate_schema': source['source_coordinate_schema'],
    }
    if binding != p['binding'] or source['partition'] != p['partition']:
        raise ValueError('representation/model/partition binding differs')
    artifact = torch.load(p['artifact'], map_location='cpu')
    if artifact['binding'].get('classes') != list(CLASSES):
        raise ValueError('actual-class vocabulary differs')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    model = load_probe(artifact, binding, device)
    data = dict(np.load(rep / 'data.npz'))
    features = service.read_jsonl(p['features'])
    if len(data['x']) != len(features) or list(data['x'].shape) != source['shape']:
        raise ValueError('representation coverage differs')
    if any(f['feature_index'] != i or f['input_frames'] != data['lengths'][i]
           for i, f in enumerate(features)):
        raise ValueError('feature order or frame lengths differ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    tick = time.monotonic()
    ps, av = scores(model, data, device, p['batch'])
    np.savez(out / 'scores.npz', probabilities=ps, available=av)
    report = {'scope': p['scope'], 'partition': p['partition'], 'binding': binding,
              'batch': p['batch'], 'inputs': len(features), 'available': int(av.sum()),
              'seconds': time.monotonic() - tick, 'scores_sha256': service.sha(out / 'scores.npz'),
              'encoder_or_head_updates': False, 'TAG_or_expectedC_model_input': False,
              'deployment': False}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

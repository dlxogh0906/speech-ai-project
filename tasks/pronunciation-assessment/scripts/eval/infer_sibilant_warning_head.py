"""Score runtime-only warning features with one frozen learned-support head."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import json
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_context_rank import model, score_single


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, root, out = json.loads(Path(a.protocol).read_text()), Path(a.features), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    report = json.loads((root/'report.json').read_text())
    if json.loads((root/'finished.json').read_text())['sha256'] != service.sha(root/'report.json'):
        raise ValueError('incomplete features')
    if service.sha(root/'cal.npz') != report['sha256']['cal.npz']:
        raise ValueError('acoustic input changed')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(p['threads'])
    artifact = torch.load(p['artifacts'][p['selected_arm']], map_location='cpu')
    if artifact.get('readout', 'existing') != 'existing':
        raise ValueError('ordinary learned-support artifact required')
    encoder, head = model(p, artifact)
    torch.cuda.reset_peak_memory_stats()
    ps, av, seconds = score_single(encoder, head, dict(np.load(root/'cal.npz')))
    np.savez(out/'scores.npz', probabilities=ps, available=av)
    result = {'features': len(ps), 'support_available': int(av.sum()), 'seconds': seconds,
              'inference_batch': 1, 'max_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
              'inputs_sha256': service.sha(root/'cal.npz'), 'scores_sha256': service.sha(out/'scores.npz'),
              'artifact_sha256': service.sha(p['artifacts'][p['selected_arm']]), 'TAG_labels_read': False}
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

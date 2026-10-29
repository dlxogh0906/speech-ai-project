"""Score coupled native-tail artifacts from original runtime PCM, with no TAG input."""
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
from train_sibilant_hubert_adaptation import RawDataset, score_raw
from mdd.sibilant_hubert_adaptation import load_native, load_coupled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen score input changed: ' + path)
    cache, representation = Path(p['cache']), Path(p['representation'])
    if json.loads((cache / 'report.json').read_text())['partition'] != p['partition']:
        raise ValueError('declared score partition differs')
    features = service.read_jsonl(representation / 'features.jsonl')
    original = service.read_jsonl(cache / 'data.features.jsonl')
    if len(features) != len(original) or len(features) != p['expected_inputs']:
        raise ValueError('score coverage differs')
    for i, (a, b) in enumerate(zip(features, original)):
        if (a['feature_index'] != i or any(a[k] != b[k] for k in
                ('uid', 'q_runtime', 'left', 'right', 'clip_sha256'))):
            raise ValueError('runtime feature identity differs')
    # Read geometry only. The stored native hidden states are intentionally not
    # read: the coupled encoder is run afresh for every input and every arm.
    geometry = np.load(representation / 'data.npz')
    data = RawDataset(cache, features, geometry['relative_vowel'], geometry['lengths'])
    torch.set_num_threads(2)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(args.protocol).read_text())
    report = {'scope': p['scope'], 'partition': p['partition'], 'arms': {}, 'deployment': False}
    for mode, path in p['artifacts'].items():
        encoder, processor, binding = load_native(p['model_manifest'], device)
        head = load_coupled(torch.load(path, map_location='cpu'), dict(binding, mode=mode), encoder, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        tick = time.monotonic()
        ps, available = score_raw(encoder, processor, head, data, device, p['batch'], progress=mode)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        dest = out / (mode + '.scores.npz')
        np.savez(dest, probabilities=ps, available=available)
        report['arms'][mode] = {'inputs': len(ps), 'available': int(available.sum()),
            'batch': p['batch'], 'seconds': seconds,
            'mean_seconds_per_site': seconds / len(ps),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'score_sha256': service.sha(dest), 'parameter_updates': 0,
            'cost_scope': 'Offline raw encoder+head+WAV IO, excludes service decoder, API, UI, NPU and model load; warm WAV cache may differ between arms.'}
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(mode, json.dumps(report['arms'][mode], ensure_ascii=False), flush=True)
        del encoder, head
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (out / 'finished.json').write_text(json.dumps({'report_sha256': service.sha(out / 'report.json')}))


if __name__ == '__main__':
    main()

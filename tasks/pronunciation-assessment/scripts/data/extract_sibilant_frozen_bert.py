"""Frozen h24 control on unchanged observed-V feature windows; no TAG input."""
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
import time
import numpy as np
import torch
from transformers import Wav2Vec2BertForCTC
import eval_sibilant_directional as service


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
    data = np.load(Path(p['cache']) / 'data.npz')
    source, lengths = data['x'], data['lengths']
    model = Wav2Vec2BertForCTC.from_pretrained(p['checkpoint']).eval().requires_grad_(False)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    if device.type == 'cuda':
        model.half()
    dtype = next(model.parameters()).dtype
    torch.set_num_threads(2)
    result = np.zeros((len(source), source.shape[1], model.config.hidden_size), np.float16 if dtype == torch.float16 else np.float32)
    tick = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(source), p['batch']):
            x = torch.as_tensor(source[start:start + p['batch']], device=device, dtype=dtype)
            n = torch.as_tensor(lengths[start:start + p['batch']], device=device)
            mask = torch.arange(x.shape[1], device=device)[None, :] < n[:, None]
            h = model.wav2vec2_bert(input_features=x, attention_mask=mask.long()).last_hidden_state
            if h.shape[:2] != x.shape[:2]:
                raise ValueError('control frame grid changed')
            result[start:start + len(x)] = h.masked_fill(~mask[..., None], 0).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('nonfinite frozen representation')
    np.savez(out / 'data.npz', x=result, relative_vowel=data['relative_vowel'], lengths=lengths)
    report = {'scope': p['scope'], 'shape': list(result.shape), 'dtype': str(result.dtype),
              'seconds': time.monotonic() - tick, 'TAG_or_expectedC_inputs': False,
              'feature_schema': 'frozen_service_bert_h24_fp16_v1', 'partition': p['partition'],
              'source_coordinate_schema': 'runtime_nextV_original20left11right_presence_v1',
              'backbone_sha256': service.sha(Path(p['checkpoint']) / 'model.safetensors'),
              'data_sha256': service.sha(out / 'data.npz'), 'deployment': False}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

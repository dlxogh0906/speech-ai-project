"""Memoize immutable native role observations; verify all original predictions."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import json
import shutil
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_precision_tail import PrecisionTailHead, SUMMARY_SCHEMA
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import observed_vowel_support


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen summary input changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('this memoization protocol is TRAIN-only')
    features = service.read_jsonl(p['features'])
    reference = service.read_jsonl(cache/'data.features.jsonl')
    if len(features) != len(reference):
        raise ValueError('native/source coverage differs')
    for i, (a, b) in enumerate(zip(features, reference)):
        if a['feature_index'] != i or any(a[k] != b[k] for k in ('uid', 'q_runtime', 'left', 'right', 'clip_sha256')):
            raise ValueError('native summary source identity differs')
    data = dict(np.load(p['data']))
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    base = load_probe(artifact, p['binding'], device)
    before = tensor_digest(base.state_dict())
    model = PrecisionTailHead(base, 'ce_full').to(device).eval()
    summaries, moments, probabilities, available = [], [], [], []
    tick = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(features), 64):
            x = torch.as_tensor(data['x'][start:start+64], device=device)
            n = torch.as_tensor(data['lengths'][start:start+64], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < n[:, None]
            relative = torch.as_tensor(data['relative_vowel'][start:start+64], device=device)
            original, phase = base(x, mask, relative)
            summary, moment, new_phase = model.summarize(x, mask, relative)
            logits = model.from_summary(summary, moment)
            if not torch.equal(original, logits) or not torch.equal(phase, new_phase):
                raise ValueError('memoized observation is not exactly original readout')
            summaries.extend(summary.cpu().numpy())
            moments.extend(moment.cpu().numpy())
            probabilities.extend(logits.softmax(-1).cpu().numpy())
            available.extend(observed_vowel_support(phase, mask).cpu().numpy())
    if tensor_digest(base.state_dict()) != before:
        raise ValueError('immutable native model changed during memoization')
    summary = np.asarray(summaries, np.float32)
    moment = np.asarray(moments, np.float32)
    ps, av = np.asarray(probabilities), np.asarray(available)
    expected = np.load(p['original_scores'])
    np.testing.assert_array_equal(ps, expected['probabilities'])
    np.testing.assert_array_equal(av, expected['available'])
    if shutil.disk_usage(out.parent).free < summary.nbytes+moment.nbytes+100000000:
        raise ValueError('insufficient disk without deleting old artifacts')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    np.savez(out/'summary.npz', summary=summary, moments=moment, available=av)
    np.savez(out/'base.scores.npz', probabilities=ps, available=av)
    report = {'scope': p['scope'], 'partition': 'train', 'schema': SUMMARY_SCHEMA,
        'inputs': len(features), 'summary_shape': list(summary.shape), 'moments_shape': list(moment.shape),
        'dtype': 'float32', 'all_original_source_identity_equal': True,
        'all13741_original_probabilities_bitwise_equal': True, 'all_original_phase_logits_equal': True,
        'base_unchanged': True, 'base_head_sha256': tensor_digest(artifact['head']), 'TAG_or_label_input': False,
        'availability': int(av.sum()), 'seconds': time.monotonic()-tick,
        'summary_sha256': service.sha(out/'summary.npz'), 'deployment': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

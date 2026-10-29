"""Extract existing raw acoustic observations without decimating to the CTC grid."""
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
import hashlib
import json
import time
import numpy as np
import eval_sibilant_directional as service
from mdd.sibilant_hires_acoustics import SCHEMA, NAMES, observations, normalization_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen acoustic source changed: '+path)
    cache = Path(p['cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != p['partition']:
        raise ValueError('acoustic partition differs')
    if p.get('fit_normalization') and p['partition'] != 'train':
        raise ValueError('normalization may only fit TRAIN inputs')
    features = service.read_jsonl(p['features'])
    original = service.read_jsonl(cache/'data.features.jsonl')
    if len(features) != len(original):
        raise ValueError('acoustic source coverage differs')
    for i, (a, b) in enumerate(zip(features, original)):
        if a['feature_index'] != i or any(a[k] != b[k] for k in ('uid', 'q_runtime', 'left', 'right', 'clip_sha256')):
            raise ValueError('acoustic site identity differs')
    wanted = {f['uid'] for f in features}
    rows = {}
    for line in (cache/'data.responses.jsonl').open():
        z = json.loads(line)
        if z['row']['uid'] in wanted:
            rows[z['row']['uid']] = {k: z['row'][k] for k in ('uid', 'wav', 'frames') if k in z['row']}
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    arrays, relatives, metadata = [], [], []
    counts, sums, squares = np.zeros(len(NAMES), np.int64), np.zeros(len(NAMES)), np.zeros(len(NAMES))
    uid, source = None, None
    tick = time.monotonic()
    for i, f in enumerate(features):
        if uid != f['uid']:
            uid = f['uid']
            source = service.load_audio(rows[uid])[:320000]
        if hashlib.sha256(source[f['left']:f['right']].tobytes()).hexdigest() != f['clip_sha256']:
            raise ValueError('original acoustic PCM differs')
        x, relative, first = observations(source, f, f['observed_reference_seconds'])
        valid = np.isfinite(x)
        z = np.where(valid, x, 0).astype(np.float64)
        counts += valid.sum(0)
        sums += z.sum(0)
        squares += (z*z).sum(0)
        arrays.append(x)
        relatives.append(relative)
        metadata.append(dict(f, acoustic_schema=SCHEMA, source_input_frames=f['input_frames'],
            input_frames=len(x), acoustic_first_source_frame=first,
            acoustic_frame_first_source_sample=first*32+79.5,
            spectral_mapping='nearest supported 25ms/hop5ms observation; no extrapolation beyond measured centers'))
        if (i+1) % 500 == 0:
            print(json.dumps({'partition': p['partition'], 'done': i+1, 'total': len(features),
                              'seconds': time.monotonic()-tick}), flush=True)
    lengths = np.asarray([len(x) for x in arrays], np.int64)
    width = int(lengths.max())
    x = np.full((len(arrays), width, len(NAMES)), np.nan, np.float32)
    relative = np.zeros((len(arrays), width), np.float32)
    for i, (a, b) in enumerate(zip(arrays, relatives)):
        x[i, :len(a)] = a
        relative[i, :len(b)] = b
    np.savez(out/'data.npz', x=x, relative_vowel=relative, lengths=lengths,
             first_frame=np.asarray([f['acoustic_first_source_frame'] for f in metadata], np.int64))
    with (out/'features.jsonl').open('w') as f:
        for row in metadata:
            service.write_line(f, row)
    normalization = None
    if p.get('fit_normalization'):
        mean = sums/np.maximum(counts, 1)
        scale = np.maximum(np.sqrt(np.maximum(squares/np.maximum(counts, 1)-mean*mean, 0)), 1e-3)
        normalization = {'mean': mean.astype(np.float32).tolist(), 'scale': scale.astype(np.float32).tolist(),
                         'finite_counts': counts.tolist(), 'partition': 'train', 'schema': SCHEMA}
        (out/'normalization.json').write_text(json.dumps(normalization, indent=2))
    report = {'scope': p['scope'], 'partition': p['partition'], 'schema': SCHEMA,
        'names': list(NAMES), 'shape': list(x.shape), 'finite_counts': counts.tolist(),
        'all_original_PCM_equal': True, 'TAG_or_actual_input': False,
        'normalization_sha256': normalization_digest(normalization) if normalization else None,
        'seconds': time.monotonic()-tick, 'deployment': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

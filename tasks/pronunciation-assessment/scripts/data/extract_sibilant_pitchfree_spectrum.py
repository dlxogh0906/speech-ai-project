"""Cache pitch-free original-PCM low-band spectra, retaining all runtime sites."""
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
import shutil
import time
import numpy as np
import eval_sibilant_directional as service
from mdd.sibilant_pitchfree_spectrum import (SCHEMA, BINS, WINDOW, HOP,
    observations, normalization_digest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen spectrum source changed: '+path)
    cache = Path(p['cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != p['partition']:
        raise ValueError('spectrum partition differs')
    if p.get('fit_normalization') and p['partition'] != 'train':
        raise ValueError('scaler may only fit TRAIN')
    features = service.read_jsonl(p['features'])
    original = service.read_jsonl(cache/'data.features.jsonl')
    if len(features) != len(original):
        raise ValueError('spectrum source coverage differs')
    for i, (a, b) in enumerate(zip(features, original)):
        if a['feature_index'] != i or any(a[k] != b[k] for k in ('uid', 'q_runtime', 'left', 'right', 'clip_sha256')):
            raise ValueError('spectrum source identity differs')
    first = np.asarray([(f['left']+HOP-1)//HOP for f in features], np.int64)
    lengths = np.asarray([max(0, (f['right']-WINDOW)//HOP-j+1)
                          for f, j in zip(features, first)], np.int64)
    width = max(1, int(lengths.max()))
    estimate = len(features)*width*(BINS*2+2+4)+first.nbytes+lengths.nbytes
    free = shutil.disk_usage(out.parent).free
    if free < estimate+int(p.get('reserve_bytes', 700000000)):
        raise ValueError('insufficient disk without deleting retained artifacts')
    wanted = {f['uid'] for f in features}
    rows = {}
    for line in (cache/'data.responses.jsonl').open():
        z = json.loads(line)
        if z['row']['uid'] in wanted:
            rows[z['row']['uid']] = {k: z['row'][k] for k in ('uid', 'wav', 'frames') if k in z['row']}
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    # Direct memory maps avoid a second serialized copy on the nearly full disk.
    x = np.lib.format.open_memmap(str(out/'spectrum.npy'), mode='w+', dtype=np.float16,
                                shape=(len(features), width, BINS))
    e = np.lib.format.open_memmap(str(out/'energy.npy'), mode='w+', dtype=np.float16,
                                shape=(len(features), width))
    r = np.lib.format.open_memmap(str(out/'relative.npy'), mode='w+', dtype=np.float32,
                                shape=(len(features), width))
    x[:] = np.nan
    e[:] = np.nan
    r[:] = 0
    np.save(out/'lengths.npy', lengths)
    np.save(out/'first_frame.npy', first)
    count, sums, squares = 0, np.zeros(BINS), np.zeros(BINS)
    esum, esquare, no_measurement = 0., 0., 0
    uid, source = None, None
    tick = time.monotonic()
    max_rounding = 0.
    with (out/'features.jsonl').open('w') as stream:
        for i, f in enumerate(features):
            if uid != f['uid']:
                uid = f['uid']
                source = service.load_audio(rows[uid])[:320000]
            if hashlib.sha256(source[f['left']:f['right']].tobytes()).hexdigest() != f['clip_sha256']:
                raise ValueError('original spectrum PCM differs')
            values, energy, relative, start = observations(source, f, f['observed_reference_seconds'])
            n = int(lengths[i])
            if start != first[i] or len(values) != n:
                raise ValueError('spectrum frame geometry differs')
            x[i, :n], e[i, :n], r[i, :n] = values, energy, relative
            valid = np.isfinite(values).all(-1) & np.isfinite(energy)
            a, b = values[valid].astype(np.float64), energy[valid].astype(np.float64)
            count += int(valid.sum())
            sums += a.sum(0)
            squares += (a*a).sum(0)
            esum += b.sum()
            esquare += (b*b).sum()
            no_measurement += int(not valid.any())
            service.write_line(stream, dict(f, spectral_schema=SCHEMA,
                spectral_frames=n, spectral_first_global_frame=int(start),
                spectral_first_center_sample=int(start)*HOP+(WINDOW-1)/2,
                measured_frames=int(valid.sum()), source_input_frames=f['input_frames']))
            if (i+1) % 1000 == 0:
                print(json.dumps({'partition': p['partition'], 'done': i+1, 'total': len(features),
                                  'seconds': time.monotonic()-tick}), flush=True)
    x.flush()
    e.flush()
    r.flush()
    norm = None
    if p.get('fit_normalization'):
        if count == 0:
            raise ValueError('no TRAIN spectrum measurements')
        mean = sums/count
        scale = np.maximum(np.sqrt(np.maximum(squares/count-mean**2, 0)), 1e-3)
        emean = esum/count
        norm = {'schema': SCHEMA, 'partition': 'train', 'mean': mean.tolist(),
                'scale': scale.tolist(), 'energy_mean': float(emean),
                'energy_scale': float(max(np.sqrt(max(esquare/count-emean**2, 0)), 1e-3)),
                'measured_frames': count, 'stored_dtype': 'float16',
                'fit_source': 'fine TRAIN stored precision only; shared by both arms'}
        (out/'normalization.json').write_text(json.dumps(norm, indent=2))
    files = ['spectrum.npy', 'energy.npy', 'relative.npy', 'lengths.npy', 'first_frame.npy', 'features.jsonl']
    report = {'scope': p['scope'], 'partition': p['partition'], 'schema': SCHEMA,
        'shape': list(x.shape), 'sites': len(features), 'physical_frames': int(lengths.sum()),
        'measured_frames': count, 'no_measurement_sites': no_measurement,
        'all_original_PCM_equal': True, 'TAG_or_actual_input': False,
        'window_samples': WINDOW, 'hop_samples': HOP, 'stored_dtype': 'float16',
        'FFT_bins_are_not_independent_resolution': True,
        'normalization_sha256': normalization_digest(norm) if norm else None,
        'free_bytes_before': free, 'estimated_new_bytes': estimate,
        'free_bytes_after': shutil.disk_usage(out).free,
        'data_sha256': {name: service.sha(out/name) for name in files},
        'seconds': time.monotonic()-tick, 'deployment': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

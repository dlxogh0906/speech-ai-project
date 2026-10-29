"""Check that the serving path (dialect_classifier.ensemble) reproduces the sealed-test scoring (2026-10-26).

Draws --n sealed-test windows (fixed seed), reads each the way PreparedClips does, scores it one window at a time with
DialectEnsemble and compares with <root>/sealed_test_<date>/{eat_large,xlsr2b}_test_logits.npz (scored in batches of 16).
Prints only aggregate agreement and the largest probability difference.

    PYTHONPATH=/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38 \
        $PY tasks/dialect-classification/scripts/eval/check_serving_parity.py \
        --root /data/speech-ai/out/dialect/content_grouped_full_20261015 --sealed sealed_test_20261026
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import csv
import json
from pathlib import Path


def main(args):
    import numpy as np
    import soundfile as sf
    from dialect_classifier.ensemble import DialectEnsemble, KINDS, SR
    model = DialectEnsemble(args.root)
    with (args.root / 'manifests' / 'test.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    saved = {k: np.load(args.root / args.sealed / ('%s_test_logits.npz' % k)) for k in KINDS}
    for k in KINDS:
        assert list(saved[k]['row_id']) == [r['row_id'] for r in rows]
    pick = np.random.default_rng(args.seed).choice(len(rows), args.n, replace=False)
    ref = {k: model.softmax(saved[k]['logits'][pick]) for k in KINDS}
    ref['selected'] = model.candidates(ref['eat_large'], ref['xlsr2b'])[model.rule]
    got = {k: [] for k in ref}
    ms = []
    for i in pick:
        r = rows[i]
        with sf.SoundFile(r['audio_path']) as source:
            source.seek(round(float(r['start_sec']) * SR))
            wav = source.read(round(float(r['duration_sec']) * SR), dtype='float32')
        out = model.score([wav])
        ms.append(out['encode_ms'])
        for k in got:
            got[k].append(out[k][0])
    result = dict(rule=model.rule, windows=int(args.n), median_encode_ms=float(np.median(ms)))
    for k in got:
        g = np.array(got[k])
        result[k] = dict(argmax_agreement=float((g.argmax(1) == ref[k].argmax(1)).mean()),
                         max_abs_prob_diff=float(np.abs(g - ref[k]).max()))
    print(json.dumps(result, indent=2), flush=True)
    print('PARITY_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sealed', default='sealed_test_20261026')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())

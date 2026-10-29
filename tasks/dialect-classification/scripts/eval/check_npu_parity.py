"""Score the NPU bundle's parity set and compare with the sealed-test scoring (2026-10-28).

parity.npz (written by scripts/export/export_npu_bundle.py) holds sealed-test windows as int16 PCM, their label ids, the saved
sealed-test probabilities and the GPU pure-path pooled features. This runs the same windows through
dialect_classifier.npu_ensemble.NpuEnsemble — backend rbln on the NPU server, torch for a CPU/GPU reference —
and prints aggregate agreement, the largest differences and per-window timing. No audio or labels are printed.

    /root/nat_serve/venv/bin/python tasks/dialect-classification/scripts/eval/check_npu_parity.py models --backend rbln --devices 5,5
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import json
import time
from pathlib import Path


def main(args):
    import numpy as np
    from dialect_classifier.ensemble import KINDS
    from dialect_classifier.npu_ensemble import NpuEnsemble, combine, softmax
    root = Path(args.models)
    with np.load(str(root / args.parity)) as f:
        z = {k: f[k] for k in f.files}       # 압축 npz 는 키를 읽을 때마다 다시 푼다 — 한 번만 꺼낸다
    ends = np.cumsum(z['lengths'])
    windows = [z['pcm'][e - n:e].astype(np.float32) / 32768 for e, n in zip(ends, z['lengths'])][:args.n or None]
    model = NpuEnsemble(root, backend=args.backend, devices=[int(d) for d in args.devices.split(',')], torch_device=args.torch_device)
    got = {k: [] for k in KINDS}
    pooled_diff = {k: 0.0 for k in KINDS}
    ms = {k: [] for k in KINDS}
    for j, wav in enumerate(windows):
        for k in KINDS:
            t0 = time.perf_counter()
            p = model.pooled(k, wav)
            ms[k].append((time.perf_counter() - t0) * 1000)
            pooled_diff[k] = max(pooled_diff[k], float(np.abs(p[0] - z['pooled_gpu_fp16_%s' % k][j]).max()))
            got[k].append(model.heads[k](p)[0])
    n = len(windows)
    probs = {k: softmax(np.array(v)) for k, v in got.items()}
    probs['selected'] = combine(model.rule, probs['eat_large'], probs['xlsr2b'])
    result = dict(backend=args.backend, device=model.device, rule=model.rule, windows=n,
                  pooled_max_abs_diff_vs_gpu=pooled_diff,
                  median_ms={k: float(np.median(v)) for k, v in ms.items()},
                  top1_on_labels=float((probs['selected'].argmax(1) == z['labels'][:n]).mean()),
                  saved_top1_on_labels=float((z['saved_selected'][:n].argmax(1) == z['labels'][:n]).mean()))
    for k in probs:
        ref = z['saved_%s' % k][:n]
        result[k] = dict(argmax_agreement=float((probs[k].argmax(1) == ref.argmax(1)).mean()),
                         max_abs_prob_diff=float(np.abs(probs[k] - ref).max()))
    print(json.dumps(result, indent=2), flush=True)
    print('PARITY_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', help='번들의 models 폴더 (meta.json · parity.npz)')
    parser.add_argument('--backend', default='rbln', choices=['rbln', 'torch'])
    parser.add_argument('--devices', default='5,5', help='rbln 장치 번호 eat_large,xlsr2b')
    parser.add_argument('--torch-device', default=None)
    parser.add_argument('--n', type=int, default=0, help='앞에서부터 이만큼만 (0 = 전부)')
    parser.add_argument('--parity', default='parity.npz', help='models 폴더 안의 대조 세트 파일')
    main(parser.parse_args())

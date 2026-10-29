"""Compile the NPU bundle's pure dialect encoders to .rbln (rebel-compiler 0.10.2 · torch 2.9.1, NPU server, 2026-10-28).

Input shapes are fixed at 12 s (dialect_classifier.npu_models): shorter windows are zero-padded and masked.
Run on the NPU server from the bundle root (it carries pyproject.toml, so the bootstrap below finds tasks/*/src):

    /root/nat_serve/venv/bin/python tasks/dialect-classification/scripts/export/compile_rbln.py models [--kinds xlsr2b]
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
    import rebel
    from dialect_classifier import npu_models as nm
    root = Path(args.models)
    meta = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
    shapes = dict(xlsr2b=[('source', [1, nm.STATIC_SAMPLES], 'float32'), ('frame_valid', [1, nm.XLSR_FRAMES], 'float32')],
                  eat_large=[('fbank', [1, 1, nm.EAT_FRAMES, nm.EAT_MELS], 'float32'), ('patch_valid', [1, nm.EAT_PATCHES], 'float32')])
    for kind in args.kinds.split(','):
        files = meta['files'][kind]
        model = nm.load_pure(kind, root / kind / files['state'])
        t0 = time.time()
        compiled = rebel.compile_from_torch(model, input_info=shapes[kind])
        dst = root / kind / files['rbln']
        compiled.save(str(dst))
        print('[compiled] %s %.0fs → %s (%.0f MB)' % (kind, time.time() - t0, dst, dst.stat().st_size / 1e6), flush=True)
    print('COMPILE_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', help='번들의 models 폴더 (meta.json 이 있는 곳)')
    parser.add_argument('--kinds', default='eat_large,xlsr2b')
    main(parser.parse_args())

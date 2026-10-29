# -*- coding: utf-8 -*-
"""매니페스트 부분집합 — CPU 변형(int8 등)을 test 전체(17.8만 발화)로 디코드하기엔 오래 걸릴 때.

행을 시드 고정으로 무작위 추출해 같은 형식(`{split}.tsv/.ltr/.canon`)의 새 폴더를 만든다.
원래 행 번호는 `{split}.orig_index` 에 남긴다 — 전체 디코드 캐시에서 같은 행만 뽑아 비교할 때 쓴다.
모든 변형을 **같은 부분집합**으로 재야 비교가 성립한다.

    $PY tasks/pronunciation-assessment/scripts/data/make_manifest_subset.py \\
        --src $MDD_OUTPUTS/manifest_tag_rebuilt --n 10000 --seed 0 --out $MDD_OUTPUTS/manifest_tag_rebuilt_sub10k
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(os.path.join(a.src, f"{a.split}.tsv")) as f:
        header = f.readline()
        tsv = f.read().splitlines()
    exts = [e for e in ("ltr", "canon") if os.path.exists(os.path.join(a.src, f"{a.split}.{e}"))]
    cols = {e: open(os.path.join(a.src, f"{a.split}.{e}"), encoding="utf-8").read().splitlines() for e in exts}
    for e, lines in cols.items():
        if len(lines) != len(tsv):
            raise SystemExit(f"{a.split}.{e} 줄 수 {len(lines)} ≠ tsv 행 수 {len(tsv)}")
    if a.n >= len(tsv):
        raise SystemExit(f"--n {a.n} 이 전체 {len(tsv)} 이상이다")

    idx = np.sort(np.random.default_rng(a.seed).choice(len(tsv), a.n, replace=False))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.split}.tsv"), "w") as f:
        f.write(header)
        f.writelines(tsv[i] + "\n" for i in idx)
    for e, lines in cols.items():
        with open(os.path.join(a.out, f"{a.split}.{e}"), "w", encoding="utf-8") as f:
            f.writelines(lines[i] + "\n" for i in idx)
    with open(os.path.join(a.out, f"{a.split}.orig_index"), "w") as f:
        f.writelines(f"{i}\n" for i in idx)
    sec = sum(int(tsv[i].split("\t")[1]) for i in idx) / 16000
    print(f"[subset] {a.n:,}/{len(tsv):,}행 · {sec/3600:.1f} h · seed {a.seed} → {a.out}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

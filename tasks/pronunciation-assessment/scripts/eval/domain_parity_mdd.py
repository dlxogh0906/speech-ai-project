# -*- coding: utf-8 -*-
"""경량화 변형의 **도메인 동등성** — 제대로 읽은 발화(LAR · 원어민)에서 인식이 원본과 같은가.

`build_domain_check_manifest.py` 로 만든 매니페스트는 모두 "정확히 읽은" 발화라 정답은 표준발음(canonical)이다.
그래서 오류 F1 대신 두 가지를 잰다.
  PER        예측 ↔ canonical 편집거리 / canonical 길이   (낮을수록 좋다 — 오탐이 적다)
  vs ref     예측 ↔ 기준 모델 예측 편집거리 / 기준 길이 · 완전 일치 비율 (0 / 100% 이면 동일)
ΔPER(cand − ref)의 95% 구간은 **화자 단위** 쌍 부트스트랩이다(`paired_bootstrap_mdd.py` 와 같은 이유).

예측 캐시는 `decode_manifest_backend.py --split test` 형식({"i", "pred"}).

    $PY tasks/pronunciation-assessment/scripts/eval/domain_parity_mdd.py --manifest <out>/lar \
        --ref preds_torch.jsonl --cand fp16=preds_fp16.jsonl --cand int8=preds_int8.jsonl --out parity_lar.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os
import numpy as np


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def load_preds(path, n):
    p = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        p[r["i"]] = r["pred"]
    missing = [i for i in range(n) if i not in p]
    if missing:
        raise SystemExit(f"{path}: 예측 {len(missing)}건 누락(예: {missing[:5]})")
    return [p[i] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref", required=True, help="기준 모델 예측 캐시")
    ap.add_argument("--cand", action="append", required=True, help="이름=경로 (여러 번)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    canon = [l.split() for l in open(os.path.join(a.manifest, "test.canon"), encoding="utf-8")]
    spk = [json.loads(l)["spk"] for l in open(os.path.join(a.manifest, "items.jsonl"), encoding="utf-8")]
    n = len(canon)
    assert len(spk) == n, "items.jsonl 과 test.canon 행 수가 다르다"
    L = np.array([len(c) for c in canon], float)
    ref = load_preds(a.ref, n)
    e_ref = np.array([edit_distance(p, c) for p, c in zip(ref, canon)], float)

    spk_ids = sorted(set(spk))
    idx_of = {s: k for k, s in enumerate(spk_ids)}
    sp = np.array([idx_of[s] for s in spk])
    rng = np.random.default_rng(a.seed)
    boots = rng.integers(0, len(spk_ids), size=(a.n_boot, len(spk_ids)))
    counts = np.stack([np.bincount(b, minlength=len(spk_ids)) for b in boots])     # (B, S) 화자 재표집 횟수

    def by_spk(v):
        return np.bincount(sp, weights=v, minlength=len(spk_ids))

    res = {"manifest": a.manifest, "n": n, "n_spk": len(spk_ids), "canon_tokens": int(L.sum()),
           "ref": {"path": a.ref, "PER": round(e_ref.sum() / L.sum(), 5),
                   "exact_canon": round(float((e_ref == 0).mean()), 4)}, "cands": {}}
    for item in a.cand:
        name, path = item.split("=", 1)
        pr = load_preds(path, n)
        e = np.array([edit_distance(p, c) for p, c in zip(pr, canon)], float)
        d_ref = np.array([edit_distance(p, r) for p, r in zip(pr, ref)], float)
        ref_len = np.array([max(1, len(r)) for r in ref], float)
        S_e, S_r, S_L = by_spk(e), by_spk(e_ref), by_spk(L)
        dper = (counts @ (S_e - S_r)) / (counts @ S_L)
        res["cands"][name] = {
            "path": path, "PER": round(e.sum() / L.sum(), 5),
            "exact_canon": round(float((e == 0).mean()), 4),
            "dPER": round((e.sum() - e_ref.sum()) / L.sum(), 5),
            "dPER_ci95": [round(float(np.percentile(dper, 2.5)), 5), round(float(np.percentile(dper, 97.5)), 5)],
            "token_edit_vs_ref": round(d_ref.sum() / ref_len.sum(), 5),
            "identical_to_ref": round(float((d_ref == 0).mean()), 4),
        }
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

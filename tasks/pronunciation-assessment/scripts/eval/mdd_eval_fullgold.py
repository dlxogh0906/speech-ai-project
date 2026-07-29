# -*- coding: utf-8 -*-
"""전체 gold(오디오 있는 것 전부) 위치탐지 평가 — 우리 모델(checkpoint_best.pt).
records_pronlab(188만 스펙)에서 오디오 있는 것만 → 디코드 → 위치단위 P/R/F1.
(주의: err_pos=phones티어 인덱스 vs target_phones=KoG2P → 20% 위치버그 포함. 참고치.)"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, MDD_ANALYSIS, MDD_OUTPUTS

import sys, os, json, wave
import numpy as np
from mdd.mdd_eval import Decoder, load_wav, batched
from mdd.mdd_eval_pos import _nw, prf_at_threshold
from mdd.l1prior.s0_taxonomy import SUPRASEG_FLUENCY

PRONLAB = f"{MDD_ANALYSIS}/040_041_042_043_both_pronlab/records_pronlab.jsonl"
GW = f"{DATA_ROOT}/mdd_gold_wav16k"
OUT = f"{MDD_OUTPUTS}/eval_fullgold.json"


def main():
    print("[1] records_pronlab + 오디오 필터...", flush=True)
    recs = []
    n = 0
    for line in open(PRONLAB, encoding="utf-8"):
        d = json.loads(line); n += 1
        f = d.get("file", ""); stem = f[:-5] if f.endswith(".json") else f
        wp = os.path.join(GW, stem[:2], stem + ".wav")
        if not os.path.exists(wp):
            continue
        try:
            fr = wave.open(wp, "rb").getnframes()
        except Exception:
            continue
        errs = d.get("errors") or []
        recs.append({"path": wp, "frames": fr, "stem": stem,
                     "canon": d.get("target_phones") or [],
                     "epos": [e.get("pos") for e in errs],
                     "etypes": [e.get("error_type") for e in errs]})
    print(f"[gold] 스펙 {n:,} | 오디오있음(평가대상) {len(recs):,}", flush=True)

    dec = Decoder()
    pg, pp, cg, cp = [], [], [], []
    npos = nerr = 0
    items = [(r["frames"], r) for r in recs]
    done = 0
    for bi, idxs in enumerate(batched(items)):
        wavs = [load_wav(items[i][1]["path"]) for i in idxs]
        preds = dec.decode(wavs)
        for j, i in enumerate(idxs):
            r = items[i][1]; C = r["canon"]
            if not C:
                continue
            pred = {op["tpos"] for op in _nw(C, preds[j]) if op["kind"] in ("sub", "del")}
            gall, gcore = set(), set()
            for p, t in zip(r["epos"], r["etypes"]):
                if isinstance(p, int) and 0 <= p < len(C):
                    gall.add(p)
                    if t not in SUPRASEG_FLUENCY: gcore.add(p)
            for k in range(len(C)):
                pg.append(1 if k in gall else 0); pp.append(1 if k in pred else 0)
                cg.append(1 if k in gcore else 0); cp.append(1 if k in pred else 0)
            npos += len(C); nerr += len(gall); done += 1
        if bi % 300 == 0:
            print(f"  {done:,}/{len(recs):,}", flush=True)
    m = prf_at_threshold(np.array(pp, float), np.array(pg, float), 0.5)
    mc = prf_at_threshold(np.array(cp, float), np.array(cg, float), 0.5)
    res = {"all": {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
           "core": {k: round(v, 4) if isinstance(v, float) else v for k, v in mc.items()},
           "n_utt": len(recs), "n_positions": npos, "n_err": nerr}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n[전체gold(오디오 {len(recs):,}) · 우리모델 · 위치탐지]")
    print(f"  전체 F1={m['F1']:.4f} P={m['P']:.4f} R={m['R']:.4f} FRR={m['FRR']:.4f} FAR={m['FAR']:.4f}")
    print(f"  CORE  F1={mc['F1']:.4f} P={mc['P']:.4f} R={mc['R']:.4f}")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()

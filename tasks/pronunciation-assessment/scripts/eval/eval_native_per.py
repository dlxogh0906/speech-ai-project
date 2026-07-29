# -*- coding: utf-8 -*-
"""원어민 test(manifest_native/test, 71음소 라벨) PER 평가.
라벨은 자모로 변환(깨진 줄 제외) 후 pred(자모)와 편집거리 → PER.
용도: 원어민 포함 학습의 게이트 ③ — 신규 ckpt vs 기존 ckpt 비교.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import MDD_OUTPUTS

import sys, os, json, time, argparse
import editdistance

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_native")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()

from mdd.jamo_to_kog2p import phones_to_jamo, INV

M = A.manifest
rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
ltr = open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()
rows = []
skip = 0
for r, l in zip(rel, ltr):
    toks = l.split()
    if any(t not in INV for t in toks):
        skip += 1; continue
    rows.append((r[0], int(r[1]), phones_to_jamo(toks)))
if A.limit:
    rows = rows[:A.limit]
print(f"[native test] 사용 {len(rows):,} / 깨진 라벨 제외 {skip}", flush=True)

from mdd import mdd_eval; mdd_eval.CKPT = A.ckpt
from mdd.mdd_eval import Decoder, load_wav, batched
dec = Decoder()

items = [(fr, i) for i, (_, fr, _) in enumerate(rows)]
P = [None] * len(rows)
t0 = time.time(); done = 0
for idxs in batched(items):
    orig = [items[j][1] for j in idxs]
    outs = dec.decode([load_wav("/" + rows[o][0]) for o in orig])
    for o, p in zip(orig, outs):
        P[o] = p
    done += len(orig)
    if done % 1024 < 32:
        print(f"  decode {done:,}/{len(rows):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

ed = tl = 0
per_utt = []
for (path, fr, ref), pred in zip(rows, P):
    d = editdistance.eval(pred, ref)
    ed += d; tl += len(ref)
    per_utt.append(d / max(1, len(ref)))
per = ed / max(1, tl)
import statistics
res = {"ckpt": A.ckpt, "n_utts": len(rows), "PER": round(per, 4),
       "PER_median": round(statistics.median(per_utt), 4),
       "collapse_rate_per>0.5": round(sum(1 for x in per_utt if x > 0.5) / len(per_utt), 4)}
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))

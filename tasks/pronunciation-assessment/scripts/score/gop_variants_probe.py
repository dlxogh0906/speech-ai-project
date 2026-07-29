# -*- coding: utf-8 -*-
"""음소별 점수 정의 4종을 같은 오디오·같은 정답으로 비교한다.

현행 GOP = mean_t[ log p(표준음소) − log p(최빈음소) ] 는 모델이 그 자리를 표준음소로
인식하는 순간 **정확히 0**이 된다(96.7%). 즉 전사 결과를 다시 말한 것일 뿐이라
0-100 점수로 쓸 해상도가 없다. 후보를 실측해 고른다:

  gop_maxnorm  현행 (최댓값 대비)                     ← 0에 붕괴
  post_raw     log p(표준음소) 원값                    ← 연속. 확신도 그 자체
  post_min     구간 내 최소 프레임의 log p             ← 한 프레임이라도 무너지면 감점
  margin2      log p(표준) − log p(2등)                ← 최빈이 표준일 때도 여유를 잰다

정답은 canonical ↔ TAG(전문가 전사) 정렬로 만든 위치별 오류 라벨.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, time, argparse, itertools
import numpy as np
import torch
import edlib
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_all_lar/checkpoint_best.pt")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/gop_variants_probe.json")
A = ap.parse_args()

from mdd import gop_feats_v2 as GF
from mdd import mdd_eval; mdd_eval.CKPT = A.ckpt
from mdd.mdd_eval import Decoder, load_wav, batched


def canon_labels(canon, other):
    if not other:
        return np.ones(len(canon), dtype=np.int8)
    r = edlib.align(other, canon, mode="NW", task="path")
    lab = np.zeros(len(canon), dtype=np.int8); ti = 0; n = 0
    for ch in r["cigar"]:
        if ch.isdigit(): n = n * 10 + int(ch); continue
        if ch == "=": ti += n
        elif ch in "XD": lab[ti:ti + n] = 1; ti += n
        n = 0
    return lab


tsv = open(f"{A.manifest}/test.tsv").read().splitlines()
canon = [l.split() for l in open(f"{A.manifest}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{A.manifest}/test.ltr", encoding="utf-8").read().splitlines()]
items = []
for i, line in enumerate(tsv[1:]):
    rel, fr = line.split("\t")
    if canon[i] and manual[i]:
        items.append((int(fr), i, "/" + rel))
import random; random.seed(0); random.shuffle(items); items = items[: A.n]
print(f"[표본] {len(items):,}발화")

dec = Decoder(); d = dec.d
V = {"gop_maxnorm": [], "post_raw": [], "post_min": [], "margin2": []}
Y = []
t0 = time.time(); done = 0
for idxs in batched(items):
    wavs = [load_wav(items[j][2]) for j in idxs]
    lp, Ls = GF.forward_lprobs(dec, wavs)
    for k, j in enumerate(idxs):
        fr, ci, _ = items[j]
        y = [d.index(t) for t in canon[ci]]
        if d.unk() in y: continue
        l = lp[k, :Ls[k]].numpy()
        segs = GF.viterbi_align(l, y, blank=dec.blank)
        if segs is None: continue
        g = l.argmax(-1)
        pred = [d[int(v)] for v, _ in itertools.groupby(g) if v != dec.blank]
        pred = [p for p in pred if p in GF.MD.JAMO]
        # 정답은 **전문가 전사(TAG)** 여야 한다. 모델 자기 예측으로 잡으면 순환이 된다.
        lab = canon_labels("".join(canon[ci]), "".join(manual[ci]))
        if len(lab) != len(y): continue
        srt = np.sort(l, axis=-1)                      # 2등 확률용
        second = srt[:, -2]; top = srt[:, -1]
        for si, fs in enumerate(segs):
            if not fs:
                for kk in V: V[kk].append(-15.0)
                continue
            fa = np.asarray(fs); py = l[fa, y[si]]
            V["gop_maxnorm"].append(float((py - top[fa]).mean()))
            V["post_raw"].append(float(py.mean()))
            V["post_min"].append(float(py.min()))
            # 최빈이 표준음소면 2등과의 차이, 아니면 1등과의 차이(음수)
            ref = np.where(g[fa] == y[si], second[fa], top[fa])
            V["margin2"].append(float((py - ref).mean()))
        Y.extend(lab.tolist())
    done += len(idxs)
    if done % 2000 < len(idxs):
        print(f"  {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

Y = np.asarray(Y, dtype=int)
print(f"\n[대조] 음소 위치 {len(Y):,} · 실제 오류 {Y.sum():,} ({Y.mean():.2%})\n")
print(f"{'정의':16}{'ROC-AUC':>10}{'PR-AUC':>9}{'최빈값 쏠림':>12}{'서로 다른 값':>13}{'해상도':>8}")
res = {}
for k, v in V.items():
    x = np.asarray(v, dtype=np.float32)
    auc = roc_auc_score(Y, -x); apr = average_precision_score(Y, -x)
    r2 = np.round(x, 2); vals, cnts = np.unique(r2, return_counts=True)
    zero = float(cnts.max() / len(r2)); uniq = len(vals)   # 한 값에 얼마나 뭉쳐 있나
    res[k] = {"auc": round(float(auc), 4), "pr_auc": round(float(apr), 4),
              "zero_frac": round(zero, 4), "n_unique": uniq}
    print(f"{k:16}{auc:>10.4f}{apr:>9.4f}{zero:>12.1%}{uniq:>13,}"
          f"{('❌' if zero > 0.5 else '✅'):>8}")
print(f"\n무작위 기준 PR-AUC = {Y.mean():.4f}")
json.dump({"n_pos": int(len(Y)), "err_rate": round(float(Y.mean()), 4), "variants": res},
          open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")

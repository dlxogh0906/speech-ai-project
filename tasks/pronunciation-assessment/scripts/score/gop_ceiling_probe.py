# -*- coding: utf-8 -*-
"""왜 어떤 방법을 써도 PR-AUC 0.4 근처에서 멈추는가 — 실패를 원인별로 분해한다.

가설: 병목은 GOP 수식이 아니라 **인식기가 오발음을 표준발음으로 되돌려 적는 것**이다
      (B-2 검증에서 오발음 보존율 0.514 = 절반만 보존).
      모델이 애초에 '표준음소가 들렸다'고 확신하면, 그 확률에서 파생되는 GOP도 높게 나온다.
      → 정의를 아무리 바꿔도 못 잡는다.

진짜 오류 위치를 두 갈래로 가른다:
  A. 모델도 이상을 감지  (전사가 표준과 다름)  → GOP가 잘 잡아야 정상
  B. 모델이 자동 교정    (전사가 표준과 같음)  → GOP가 구조적으로 못 잡음

B의 비중이 크고 A에서 성능이 좋다면, 남은 개선 여지는 GOP가 아니라 **인식기 학습**에 있다.
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
import edlib
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_all_lar/checkpoint_best.pt")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/gop_ceiling_probe.json")
A = ap.parse_args()

from mdd import gop_feats_v2 as GF
from mdd import mdd_eval; mdd_eval.CKPT = A.ckpt
from mdd.mdd_eval import Decoder, load_wav, batched


def align_lab(canon, other):
    """canonical 위치별로 other 와 어긋나는지(1/0)."""
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
M2, YT, YM = [], [], []          # margin2, TAG오류(정답), 모델도감지
t0 = time.time(); done = 0
for idxs in batched(items):
    wavs = [load_wav(items[j][2]) for j in idxs]
    lp, Ls = GF.forward_lprobs(dec, wavs)
    for k, j in enumerate(idxs):
        ci = items[j][1]
        y = [d.index(t) for t in canon[ci]]
        if d.unk() in y: continue
        l = lp[k, :Ls[k]].numpy()
        segs = GF.viterbi_align(l, y, blank=dec.blank)
        if segs is None: continue
        g = l.argmax(-1)
        pred = [d[int(v)] for v, _ in itertools.groupby(g) if v != dec.blank]
        pred = [p for p in pred if p in GF.MD.JAMO]
        yt = align_lab("".join(canon[ci]), "".join(manual[ci]))     # 전문가 기준 오류
        ym = align_lab("".join(canon[ci]), "".join(pred))           # 모델이 감지한 이상
        if len(yt) != len(y): continue
        srt = np.sort(l, axis=-1); second = srt[:, -2]; top = srt[:, -1]
        for si, fs in enumerate(segs):
            if not fs:
                M2.append(-15.0)
            else:
                fa = np.asarray(fs); py = l[fa, y[si]]
                ref = np.where(g[fa] == y[si], second[fa], top[fa])
                M2.append(float((py - ref).mean()))
        YT += yt.tolist(); YM += ym.tolist()
    done += len(idxs)
    if done % 4000 < len(idxs):
        print(f"  {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

M2 = np.asarray(M2, np.float32); YT = np.asarray(YT, int); YM = np.asarray(YM, int)
n = len(YT); err = YT == 1
detected = err & (YM == 1)          # A. 모델도 이상을 감지
missed = err & (YM == 0)            # B. 모델이 자동 교정
print(f"\n[전체] 음소 위치 {n:,} · 전문가 기준 오류 {err.sum():,} ({err.mean():.2%})")
print(f"  A. 모델도 감지  {detected.sum():>6,} ({detected.sum()/err.sum():.1%} of 오류)")
print(f"  B. 자동 교정됨  {missed.sum():>6,} ({missed.sum()/err.sum():.1%} of 오류)  ← GOP가 구조적으로 못 봄")

def score(mask_pos):
    """정상 전체 + 지정한 오류 부분집합만으로 재채점."""
    sel = (YT == 0) | mask_pos
    y = (YT[sel] == 1).astype(int); x = M2[sel]
    return roc_auc_score(y, -x), average_precision_score(y, -x), y.mean(), int(sel.sum())

print(f"\n{'평가 대상':30}{'ROC-AUC':>10}{'PR-AUC':>9}{'무작위':>9}{'배수':>7}")
for name, mask in (("전체 오류 (현재 보고 수치)", err),
                   ("A. 모델이 감지한 오류만", detected),
                   ("B. 자동 교정된 오류만", missed)):
    a, p, base, ns = score(mask)
    print(f"{name:30}{a:>10.4f}{p:>9.4f}{base:>9.4f}{p/base:>7.1f}x")

res = {"n_pos": int(n), "n_err": int(err.sum()),
       "detected": int(detected.sum()), "missed": int(missed.sum()),
       "auto_correct_rate": round(float(missed.sum() / err.sum()), 4)}
for name, mask in (("all", err), ("detected", detected), ("missed", missed)):
    a, p, base, ns = score(mask)
    res[name] = {"auc": round(float(a), 4), "pr_auc": round(float(p), 4),
                 "baseline": round(float(base), 4)}
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n[saved] {A.out}")

# -*- coding: utf-8 -*-
"""재학습(손실 가중)으로 건질 여지가 실제로 있는가 — '지워진 59%'의 음향 정보를 확인한다.

인식기가 오발음을 표준으로 되돌려 적는 자리에서, **전문가가 들은 음소가 확률 순위 몇 등인가**를 본다.

  2~3등에 몰려 있다  → 음향 정보는 이미 있는데 결정 경계가 표준 쪽으로 치우친 것.
                       손실 가중/재학습으로 끌어올릴 수 있다. (여지 있음)
  하위권에 흩어져 있다 → 인코더가 그 소리를 애초에 구분하지 못한다.
                       가중치를 바꿔도 없는 정보가 생기지 않는다. (여지 없음)

비교 기준으로 '모델도 감지한 자리'의 순위 분포를 함께 낸다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import MDD_OUTPUTS

import sys, os, json, time, argparse, itertools, collections
import numpy as np
import edlib
from mdd.mdd_eval import Decoder, load_wav, batched

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/gop_headroom_probe.json")
A = ap.parse_args()

from mdd import gop_feats_v2 as GF


def canon_map(canon, other):
    """canonical 위치 → other 에서 대응하는 문자(탈락이면 None), 일치 여부."""
    m = [None] * len(canon); ok = [False] * len(canon)
    if not other:
        return m, ok
    r = edlib.align(other, canon, mode="NW", task="path")
    ti = qi = 0; n = 0
    for ch in r["cigar"]:
        if ch.isdigit(): n = n * 10 + int(ch); continue
        if ch == "=":
            for t in range(n): m[ti + t] = other[qi + t]; ok[ti + t] = True
            ti += n; qi += n
        elif ch == "X":
            for t in range(n): m[ti + t] = other[qi + t]
            ti += n; qi += n
        elif ch == "D":                      # canonical 에만 있음 = 탈락
            ti += n
        else:                                # I = other 에만 있음 = 삽입
            qi += n
        n = 0
    return m, ok


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
RANK = {"missed": [], "detected": []}      # 전문가가 들은 음소의 확률 순위
DEL = {"missed": 0, "detected": 0}         # 탈락(대응 음소 없음)은 순위 정의 불가
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
        tag_at, tag_ok = canon_map("".join(canon[ci]), "".join(manual[ci]))
        _, mdl_ok = canon_map("".join(canon[ci]), "".join(pred))
        if len(tag_at) != len(y): continue
        # 구간 평균 log prob 기준 순위 (프레임별이 아니라 음소 단위로)
        for si, fs in enumerate(segs):
            if tag_ok[si] or not fs:          # 전문가 기준 정상이면 관심 없음
                continue
            grp = "detected" if not mdl_ok[si] else "missed"
            heard = tag_at[si]
            if heard is None:                 # 탈락 — 대응 음소가 없다
                DEL[grp] += 1; continue
            hi = d.index(heard)
            if hi == d.unk(): continue
            fa = np.asarray(fs); mean_lp = l[fa].mean(0)
            RANK[grp].append(int((mean_lp > mean_lp[hi]).sum()) + 1)   # 1 = 최상위
    done += len(idxs)
    if done % 4000 < len(idxs):
        print(f"  {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

print()
res = {}
for grp, label in (("detected", "A. 모델도 감지한 오류"), ("missed", "B. 자동 교정된 오류")):
    r = np.asarray(RANK[grp])
    if not len(r): continue
    top3 = float((r <= 3).mean()); top5 = float((r <= 5).mean())
    res[grp] = {"n": int(len(r)), "n_deletion": DEL[grp], "median_rank": int(np.median(r)),
                "top1": round(float((r == 1).mean()), 4), "top3": round(top3, 4),
                "top5": round(top5, 4), "mean_rank": round(float(r.mean()), 2)}
    print(f"[{label}] 치환 {len(r):,}건 (탈락 {DEL[grp]:,}건은 순위 정의 불가)")
    print(f"   전문가가 들은 음소의 확률 순위: 중앙값 {int(np.median(r))}위 · 평균 {r.mean():.1f}위")
    print(f"   1위 {(r==1).mean():.1%} · 3위 이내 {top3:.1%} · 5위 이내 {top5:.1%}")
    hist = collections.Counter(np.clip(r, 1, 11).tolist())
    print("   분포: " + " ".join(f"{('11+' if q==11 else q)}위 {hist[q]/len(r):.0%}"
                                for q in sorted(hist) if hist[q] / len(r) >= 0.03))
    print()
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")

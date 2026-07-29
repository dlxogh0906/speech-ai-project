# -*- coding: utf-8 -*-
"""① 디코딩 편향 조정 — '경계 문제인가, 표현 문제인가'를 학습 없이 가른다.

배경(실측): 인식기가 오발음의 59%를 표준발음으로 되돌려 적는다. 그런데 그 자리에서
전문가가 들은 음소는 **78%가 확률 2등**이다. 정보는 있는데 1등을 표준음소에 내주고 있다.

여기서는 강제정렬된 음소 구간마다 **표준음소 쪽에 페널티 δ**를 주고 다시 판정한다.
δ를 키우면 2등이 1등으로 뒤집힌다. 학습은 전혀 하지 않는다(추론 시 로짓 조정).

  δ를 올렸을 때 오류F1·보존율이 함께 오르면 → **경계 문제**. 재학습 없이 해결 가능.
  정밀도만 무너지고 F1이 안 오르면     → **표현 문제**. 보조 프레임 손실로 재학습해야 함.

※ 이 스크립트는 --ckpt 로 어떤 체크포인트에도 붙는다. **배포할 모델 기준으로 재야** 의미가 있다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, time, argparse, itertools
import numpy as np
import edlib

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_all_lar/checkpoint_best.pt")
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--n", type=int, default=8000, help="표본 발화 수 (GPU 경합 시 줄일 것)")
ap.add_argument("--deltas", default="0,0.5,1,1.5,2,3,4,6,8")
ap.add_argument("--tag", default="tag_all_lar", help="결과 라벨 (모델 구분용)")
ap.add_argument("--out", default=None)
A = ap.parse_args()
OUT = A.out or f"{MDD_OUTPUTS}/decode_bias_{A.tag}.json"

from mdd import mdd_eval; mdd_eval.CKPT = A.ckpt
from mdd.mdd_eval import Decoder, load_wav, batched
from mdd import gop_feats_v2 as GF


def canon_map(canon, other):
    """canonical 위치 → other 대응 문자(탈락 None) / 일치 여부."""
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
        elif ch == "D": ti += n
        else: qi += n
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
DELTAS = [float(x) for x in A.deltas.split(",")]
print(f"[모델] {A.ckpt}")
print(f"[표본] {len(items):,}발화 · δ = {DELTAS}\n")

dec = Decoder(); d = dec.d
JAMO = GF.MD.JAMO
# 위치별로: 표준음소 평균logp, 최선 대안 평균logp, 대안 음소, 정답(전문가) 오류 여부, 전문가가 들은 음소
C, B, BSYM, YT, HEARD = [], [], [], [], []
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
        heard, tag_ok = canon_map("".join(canon[ci]), "".join(manual[ci]))
        if len(tag_ok) != len(y): continue
        for si, fs in enumerate(segs):
            if not fs:
                continue
            m = l[np.asarray(fs)].mean(0).copy()
            cscore = float(m[y[si]])
            m[y[si]] = -1e9
            m[dec.blank] = -1e9                       # blank 는 음소가 아니다
            bi = int(m.argmax())
            C.append(cscore); B.append(float(m[bi])); BSYM.append(d[bi])
            YT.append(0 if tag_ok[si] else 1)
            HEARD.append(heard[si])
    done += len(idxs)
    if done % 4000 < len(idxs):
        print(f"  {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

C = np.asarray(C); B = np.asarray(B); YT = np.asarray(YT, int)
margin = C - B                                        # 클수록 표준음소 확신
print(f"\n[대조] 음소 위치 {len(YT):,} · 전문가 기준 오류 {YT.sum():,} ({YT.mean():.2%})\n")

print(f"{'δ':>5}{'오류로 판정':>12}{'정밀도':>9}{'재현율':>9}{'오류F1':>9}"
      f"{'대안이 정답':>12}{'보존율 근사':>12}")
res = {"ckpt": A.ckpt, "n_pos": int(len(YT)), "err_rate": round(float(YT.mean()), 4), "sweep": []}
best = None
for dl in DELTAS:
    pred = (margin < dl).astype(int)                  # 표준음소 확신이 δ 미만이면 '틀렸다'로 판정
    tp = int((pred & YT).sum()); fp = int(pred.sum()) - tp; fn = int(YT.sum()) - tp
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    # 오류로 판정한 자리에서, 우리가 고른 대안이 전문가가 들은 음소와 실제로 같은가
    sel = (pred == 1) & (YT == 1)
    agree = float(np.mean([BSYM[i] == HEARD[i] for i in np.nonzero(sel)[0]])) if sel.sum() else 0.0
    keep = r * agree                                  # 오류를 잡고 + 실현형까지 맞춘 비율
    res["sweep"].append({"delta": dl, "P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
                         "alt_correct": round(agree, 4), "preserve": round(keep, 4),
                         "n_flagged": int(pred.sum())})
    if best is None or f1 > best[1]: best = (dl, f1)
    print(f"{dl:>5}{int(pred.sum()):>12,}{p:>9.4f}{r:>9.4f}{f1:>9.4f}{agree:>12.1%}{keep:>12.4f}")

print(f"\n최고 오류F1 = {best[1]:.4f} (δ={best[0]})   ※ 현행 δ=0 대비 "
      f"{best[1] - res['sweep'][0]['F1']:+.4f}")
print("판정: F1이 δ와 함께 유의하게 오르면 **경계 문제**(재학습 불필요), "
      "정밀도만 무너지면 **표현 문제**(보조 프레임 손실 필요)")
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {OUT}")

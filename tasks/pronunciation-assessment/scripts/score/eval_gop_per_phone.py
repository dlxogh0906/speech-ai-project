# -*- coding: utf-8 -*-
"""4-1 검증: 음소별 GOP가 '이 음소를 틀렸다'를 실제로 짚어내는가.

체크리스트 항목: "인식기 음소별 확률 재사용해 음소별 0-100점 산출".
계산·저장은 이미 되고 있다(gop_feats_v2 의 `gops`, canonical 음소마다 1개).
여기서 처음으로 **정답과 대조**한다.

정답: canonical(표준발음) ↔ TAG(전문가가 실제로 들은 음소) 를 정렬해
      canonical 위치별로 치환/탈락이면 오류(1), 일치면 정상(0).
      → GOP 가 낮을수록 오류여야 한다.

비교 대상 2가지 (같은 위치·같은 정답):
  A. 전사 기반   canonical ↔ 모델 예측 정렬 → 다르면 오류    (현행 MDD 방식, 이진)
  B. GOP 기반    canonical 위치의 GOP < 임계 → 오류          (연속값)
  A+B. 둘 중 하나라도 / 둘 다
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
from shared.paths import MDD_OUTPUTS

import sys, os, json, argparse, collections
import numpy as np
import edlib
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=f"{MDD_OUTPUTS}/gop_feats_v2_test.jsonl")
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_gop_per_phone.json")
A = ap.parse_args()


def canon_labels(canon, other):
    """canonical 각 위치가 other 대비 오류인지. 반환 길이 = len(canon).
    other 에만 있는 음소(삽입)는 canonical 위치가 없으므로 무시한다."""
    if not other:                       # 아무것도 못 알아들음 = 전 위치 탈락
        return np.ones(len(canon), dtype=np.int8)
    r = edlib.align(other, canon, mode="NW", task="path")   # query=other, target=canon
    lab = np.zeros(len(canon), dtype=np.int8)
    ti = 0; n = 0
    for ch in r["cigar"]:
        if ch.isdigit():
            n = n * 10 + int(ch); continue
        if ch == "=":
            ti += n
        elif ch == "X":
            lab[ti:ti + n] = 1; ti += n
        elif ch == "D":          # target(canon)에만 있음 = 탈락
            lab[ti:ti + n] = 1; ti += n
        # "I" = query(other)에만 있음 = 삽입 → canonical 위치 없음
        n = 0
    assert ti == len(canon), (ti, len(canon), r["cigar"])
    return lab


# --- 자기검증: cigar 해석이 맞는지 ---
assert list(canon_labels("abc", "abd")) == [0, 0, 1], "치환"
assert list(canon_labels("abc", "ab")) == [0, 0, 1], "탈락"
assert list(canon_labels("abc", "abxc")) == [0, 0, 0], "삽입은 무시"
assert list(canon_labels("abc", "abc")) == [0, 0, 0], "일치"

tsv = open(f"{A.manifest}/test.tsv").read().splitlines()
canon = [l.split() for l in open(f"{A.manifest}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{A.manifest}/test.ltr", encoding="utf-8").read().splitlines()]
bystem = {}
for i, line in enumerate(tsv[1:]):
    bystem[os.path.basename(line.split("\t")[0])[:-4]] = i

G, Y, P = [], [], []          # GOP, 정답(1=오류), 전사기반 예측(1=오류)
n_utt = n_skip = 0
for line in open(A.feats, encoding="utf-8"):
    r = json.loads(line)
    i = bystem.get(r["stem"])
    if i is None or "gops" not in r:
        n_skip += 1; continue
    c, m = canon[i], manual[i]
    if len(r["gops"]) != len(c) or not c or not m:
        n_skip += 1; continue
    y = canon_labels("".join(c), "".join(m))
    p = canon_labels("".join(c), r.get("pred") or "")
    G.append(np.asarray(r["gops"], dtype=np.float32)); Y.append(y); P.append(p)
    n_utt += 1

Gs, Ys = G, Y                      # 발화별 리스트 보존(제품 시나리오용)
G = np.concatenate(G); Y = np.concatenate(Y).astype(int); P = np.concatenate(P).astype(int)
print(f"[대조] 발화 {n_utt:,} (건너뜀 {n_skip:,}) · 음소 위치 {len(Y):,}")
print(f"       실제 오류 {Y.sum():,} ({Y.mean():.2%}) · GOP 범위 {G.min():.2f}~{G.max():.2f}\n")


def prf(pred, y=Y):
    tp = int((pred & y).sum()); fp = int(pred.sum()) - tp; fn = int(y.sum()) - tp
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


auc = roc_auc_score(Y, -G)                     # GOP 낮을수록 오류 → 부호 반전
ap_ = average_precision_score(Y, -G)
print(f"[B. GOP 단독] ROC-AUC {auc:.4f} · PR-AUC {ap_:.4f} (무작위 기준 {Y.mean():.4f})")

best = max(((prf(( G < t).astype(int))[2], t) for t in np.percentile(G, np.arange(1, 60))), key=lambda x: x[0])
pB, rB, fB = prf((G < best[1]).astype(int))
pA, rA, fA = prf(P)
pO, rO, fO = prf(((P == 1) | (G < best[1])).astype(int))
pN, rN, fN = prf(((P == 1) & (G < best[1])).astype(int))

print(f"\n{'검출 방식':28}{'정밀도':>9}{'재현율':>9}{'F1':>9}")
print(f"{'A. 전사 기반 (현행 MDD)':28}{pA:>9.4f}{rA:>9.4f}{fA:>9.4f}")
print(f"{f'B. GOP 임계 ({best[1]:.2f})':28}{pB:>9.4f}{rB:>9.4f}{fB:>9.4f}")
print(f"{'A 또는 B (재현율 우선)':28}{pO:>9.4f}{rO:>9.4f}{fO:>9.4f}")
print(f"{'A 그리고 B (정밀도 우선)':28}{pN:>9.4f}{rN:>9.4f}{fN:>9.4f}")

# --- GOP 분포 (0에 질량이 몰려 있어 백분위 사상이 왜곡되는 원인) ---
zero = float((G >= -1e-6).mean())
print(f"\n[GOP 분포] 정확히 0(= 그 구간에서 표준음소가 최빈) 비율 {zero:.1%}")
print("  " + " · ".join(f"p{q}={np.percentile(G, q):.2f}" for q in (1, 5, 10, 25, 50)))

# --- 0-100 점수화: 정상 음소 분포 대비 백분위(동점은 중간순위) ---
ref = np.sort(G[Y == 0])
lo_i = np.searchsorted(ref, G, side="left"); hi_i = np.searchsorted(ref, G, side="right")
score100 = (lo_i + hi_i) / 2.0 / len(ref) * 100
print(f"\n[0-100 환산] 정상 음소 분포 백분위 (동점=중간순위)")
print(f"{'점수 구간':18}{'음소 수':>12}{'실제 오류율':>12}{'전체 오류 중':>12}")
for lo, hi, nm in ((0, 20, "0-20 (심각)"), (20, 40, "20-40"), (40, 60, "40-60"),
                   (60, 80, "60-80"), (80, 101, "80-100 (양호)")):
    m = (score100 >= lo) & (score100 < hi)
    if m.sum():
        print(f"{nm:18}{int(m.sum()):>12,}{Y[m].mean():>12.2%}{Y[m].sum()/Y.sum():>12.1%}")

# --- 제품 관점: 발화당 '가장 나쁜 K개'만 짚어줄 때의 적중률 ---
print(f"\n[제품 시나리오] 발화마다 GOP 최저 K개만 지적할 때")
print(f"{'K':>4}{'지적 위치':>12}{'그 중 진짜 오류':>16}{'전체 오류 중 포착':>17}")
off = 0
for K in (1, 2, 3):
    hit = tot = 0
    off = 0
    for g, y in zip(Gs, Ys):
        k = min(K, len(g))
        idx = np.argsort(g)[:k]
        hit += int(y[idx].sum()); tot += k
    print(f"{K:>4}{tot:>12,}{hit/max(1,tot):>16.2%}{hit/Y.sum():>17.1%}")

res = {"n_utt": n_utt, "n_pos": int(len(Y)), "err_rate": round(float(Y.mean()), 4),
       "gop_auc": round(float(auc), 4), "gop_pr_auc": round(float(ap_), 4),
       "thr": round(float(best[1]), 3),
       "A_transcript": {"P": round(pA, 4), "R": round(rA, 4), "F1": round(fA, 4)},
       "B_gop": {"P": round(pB, 4), "R": round(rB, 4), "F1": round(fB, 4)},
       "A_or_B": {"P": round(pO, 4), "R": round(rO, 4), "F1": round(fO, 4)},
       "A_and_B": {"P": round(pN, 4), "R": round(rN, 4), "F1": round(fN, 4)}}
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n[saved] {A.out}")

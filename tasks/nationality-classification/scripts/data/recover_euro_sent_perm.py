# -*- coding: utf-8 -*-
"""유럽 041 캐시 행 순열 복구 — 2B(전체+sent_mask) ↔ 300m(문장전용) 발화 단위 대응.

배경: 두 추출이 같은 171,241개 문장 발화를 담지만 저장 순서가 다르다
(큰 블록들이 자리바꿈된 6조각 구조). KD 등 행 단위 대응이 필요한 작업을 위해
탐욕적 최장 블록 매칭으로 순열을 복구하고 3중으로 검증한다:
  ① 전단사 + 화자열 완전 일치 (필수)
  ② y 라벨 배열 일치 (필수)
  ③ (옵션 --fingerprint) 동일 인코더 지문 대조 — 2B sent_16s 캐시(학생과 행 순서 동일)의
     layer31 벡터와 cos 비교로 '발화 단위' 대응까지 실측 확인.
     실측치: 매칭쌍 cos 0.9993 vs 동일화자 오짝 0.9565 vs 랜덤 0.9353.

출력: $SPEECH_AI_DATA/taeho_layers_euro/euro_sent_row_perm.npy
의미: 학생(euro_sent) i행 = 교사(sent_mask 적용 후) perm[i]행

    $PY tasks/nationality-classification/scripts/data/recover_euro_sent_perm.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse
from collections import defaultdict
import numpy as np


def recover(ts, ss):
    """탐욕 최장 블록 매칭: 학생열 ss 를 교사열 ts 의 미사용 연속 구간들로 덮는다.

    두 열이 '조각별 연속' 재배열 관계라면 조각 수만큼의 반복으로 끝난다.
    반환: perm (ts[perm] == ss), pieces [(학생시작, 교사시작, 길이)]
    """
    N = len(ss)
    pos = defaultdict(list)
    for i, v in enumerate(ts):
        pos[v].append(i)
    used = np.zeros(N, bool)
    perm = np.full(N, -1, np.int64)
    pieces = []
    i = 0
    while i < N:
        best_len, best_f = 0, -1
        for f in pos[ss[i]]:
            if used[f]:
                continue
            L = 0
            while i + L < N and f + L < N and not used[f + L] and ts[f + L] == ss[i + L]:
                L += 1
            if L > best_len:
                best_len, best_f = L, f
        if best_f < 0:
            raise RuntimeError(f"i={i} 에서 매치 실패 — 두 캐시가 같은 발화 집합이 아님")
        perm[i:i + best_len] = np.arange(best_f, best_f + best_len)
        used[best_f:best_f + best_len] = True
        pieces.append((i, best_f, best_len))
        i += best_len
    return perm, pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=f"{DATA_ROOT}/taeho_layers_euro/train.npz")
    ap.add_argument("--mask", default=f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
    ap.add_argument("--student", default=f"{DATA_ROOT}/taeho300_layers_euro_sent/train.npz")
    ap.add_argument("--out", default=f"{DATA_ROOT}/taeho_layers_euro/euro_sent_row_perm.npy")
    ap.add_argument("--fingerprint", action="store_true",
                    help="2B sent_16s 캐시로 발화 단위 지문 검증까지 수행 (표본 cos 대조)")
    A = ap.parse_args()

    t = np.load(A.teacher, allow_pickle=True)
    s = np.load(A.student, allow_pickle=True)
    mask = np.load(A.mask)
    ts = np.asarray(t["speaker"], dtype=object)[mask]
    ss = np.asarray(s["speaker"], dtype=object)
    assert len(ts) == len(ss), f"행수 불일치: 교사(마스크후) {len(ts):,} vs 학생 {len(ss):,}"

    perm, pieces = recover(ts, ss)

    # ── 검증 ① 전단사 + 화자열 ──
    assert np.array_equal(np.sort(perm), np.arange(len(perm))), "전단사 실패"
    assert (ts[perm] == ss).all(), "화자열 불일치"
    # ── 검증 ② y 라벨 ──
    ty = np.asarray(t["y"])[mask]
    assert np.array_equal(ty[perm], np.asarray(s["y"])), "y 라벨 불일치"

    print(f"조각 {len(pieces)}개 · 최소 {min(p[2] for p in pieces)}행")
    for st, tf, L in pieces:
        print(f"  학생 {st:>7,} ← 교사 {tf:>7,}  ({L:,}행)")

    if A.fingerprint:
        # ── 검증 ③ 발화 단위 지문 (동일 2B 인코더의 다른 추출본과 cos 대조) ──
        fp = f"{DATA_ROOT}/taeho_layers_euro_sent_16s/train.npz"
        f16 = np.load(fp, allow_pickle=True)
        assert np.array_equal(np.asarray(f16["speaker"], dtype=object), ss), \
            "sent_16s 캐시가 학생 행 순서와 다름 — 지문 검증 불가"
        li_t = t["layers"].tolist().index(31); li_f = f16["layers"].tolist().index(31)
        rng = np.random.RandomState(0)
        idx = rng.choice(len(ss), 3000, replace=False)
        Xt = t["X"][:, li_t, :].astype(np.float32)[mask][perm[idx]]
        Xf = f16["X"][:, li_f, :].astype(np.float32)[idx]
        cos = (Xt * Xf).sum(1) / (np.linalg.norm(Xt, axis=1) * np.linalg.norm(Xf, axis=1) + 1e-9)
        rnd = (Xt * np.roll(Xf, 1, 0)).sum(1) / (
            np.linalg.norm(Xt, axis=1) * np.linalg.norm(np.roll(Xf, 1, 0), axis=1) + 1e-9)
        print(f"[지문] 매칭쌍 cos 평균 {cos.mean():.4f} (min {cos.min():.4f}) vs 랜덤쌍 {rnd.mean():.4f}")
        assert cos.mean() > rnd.mean() + 0.03, "지문 분리가 불충분 — 발화 단위 대응 의심"

    np.save(A.out, perm)
    print(f"[saved] {A.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""DTW 로 원본 시각(음절 경계)을 녹음 위로 옮긴다 — 계획서 6.4 방식. 강제 정렬과 맞춰 보는 쪽이다.

특징 두 가지:
  - MFCC 13차(25 ms 창, 10 ms 간격) + 발화별 평균·분산 정규화, 유클리드 거리 — 계획서 그대로.
  - 정렬 모델(w2v-BERT) 중간층 은닉(20 ms 간격), 코사인 거리 — 화자·음색이 바뀌어도 발음 내용이 비슷하게 보인다.
경계 투영은 "원본 프레임 r−1 에 마지막으로 붙은 녹음 프레임"과 "r 에 처음 붙은 녹음 프레임" 사이의 가장자리다
(r 에 붙은 프레임들의 평균을 쓰면 끼어든 쉼 한가운데로 경계가 간다). 교차 확인(analyze.crosscheck)은 쉼을 뺀
말소리 프레임끼리만 맞춘다(project_keep).
"""
import numpy as np

MFCC_HOP, MFCC_OFF = 0.01, -0.005        # center=True: 프레임 k 의 왼쪽 가장자리 = k·10ms − 5ms


def mfcc(x, sr=16000):
    import torch
    import torchaudio
    m = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13,
                                   melkwargs=dict(n_fft=400, win_length=400, hop_length=160, n_mels=40, center=True))
    f = m(torch.from_numpy(np.asarray(x, dtype=np.float32))[None])[0].T.numpy().astype(np.float64)
    return (f - f.mean(0)) / (f.std(0) + 1e-6)


def dtw_path(A, B, metric="euclid"):
    """A[n,d] · B[m,d] → 경로 [(i, j)] (0,0)→(n-1,m-1). 대칭 DTW(대각·위·옆 한 칸)."""
    if metric == "cos":
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        C = 1.0 - An @ Bn.T
    else:
        C = np.sqrt(np.maximum(0.0, (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T))
    n, m = C.shape
    D = np.empty((n, m))
    D[0] = np.cumsum(C[0])
    for i in range(1, n):
        diag = np.concatenate([[np.inf], D[i - 1][:-1]])
        tmp = C[i] + np.minimum(D[i - 1], diag)
        S = np.cumsum(C[i])
        D[i] = S + np.minimum.accumulate(tmp - S)        # 옆 스텝: D[i,j] = min_k(tmp[k] + S[j] − S[k])
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
            i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
        path.append((i, j))
    return np.array(path[::-1])


def _span_frames(span, hop, off, n):
    a = int(np.floor((span[0] - off) / hop))
    b = int(np.ceil((span[1] - off) / hop))
    return max(0, a), min(n, max(a + 2, b))


def project_feats(F_ref, F_take, times, hop, off, ref_span=None, take_span=None, metric="cos"):
    """원본 시각들(프레임 가장자리로 반올림) → 녹음 시각들. span 이 있으면 그 구간(말소리)만 맞춘다."""
    ra, rb = _span_frames(ref_span, hop, off, len(F_ref)) if ref_span else (0, len(F_ref))
    ta, tb = _span_frames(take_span, hop, off, len(F_take)) if take_span else (0, len(F_take))
    path = dtw_path(F_ref[ra:rb], F_take[ta:tb], metric)
    n = rb - ra
    jmin = np.full(n, np.iinfo(np.int64).max)
    jmax = np.full(n, -1)
    np.minimum.at(jmin, path[:, 0], path[:, 1])
    np.maximum.at(jmax, path[:, 0], path[:, 1])
    out = []
    for t in times:
        r = int(round((t - off) / hop)) - ra                  # 가장자리 r = 프레임 r−1 과 r 사이
        if r <= 0:
            e = float(jmin[0])
        elif r >= n:
            e = float(jmax[n - 1] + 1)
        else:
            e = (jmax[r - 1] + 1 + jmin[r]) / 2.0
        out.append(round((ta + e) * hop + off, 4))
    return out


def _kept(spans, hop, off, n):
    """시간 구간들 → 그 안에 중심이 드는 프레임 번호(오름차순)."""
    centers = np.arange(n) * hop + off + hop / 2.0
    m = np.zeros(n, dtype=bool)
    for a, b in spans:
        m |= (centers >= a) & (centers < b)
    return np.flatnonzero(m)


def project_keep(F_ref, F_take, times, hop, off, ref_keep, take_keep, metric="cos"):
    """쉼을 뺀 말소리 프레임끼리만 DTW → 원본 시각들을 녹음 위 (왼쪽, 오른쪽) 구간으로.

    원본엔 1.7초 쉼이 있고 따라 한 녹음엔 없으면(가장 흔한 실수다), 쉼까지 넣은 DTW 는 그 침묵을 말소리에 억지로
    짝지으며 앞뒤 경계를 끌고 간다. 쉼은 compare 의 쉼 항목이 따로 보니 DTW 는 말소리만 맞춘다.
    keep: 남길 시간 구간들(말소리 구간을 쉼에서 끊은 것). 녹음 쪽 빠진 쉼 자리에 떨어지면 그 쉼 전체가 구간이 된다."""
    ir, it = _kept(ref_keep, hop, off, len(F_ref)), _kept(take_keep, hop, off, len(F_take))
    path = dtw_path(F_ref[ir], F_take[it], metric)
    n = len(ir)
    jmin = np.full(n, np.iinfo(np.int64).max)
    jmax = np.full(n, -1)
    np.minimum.at(jmin, path[:, 0], path[:, 1])
    np.maximum.at(jmax, path[:, 0], path[:, 1])
    cr = ir * hop + off + hop / 2.0
    out = []
    for t in times:
        r = int(np.searchsorted(cr, t))                      # 압축 공간 가장자리 = 중심이 t 앞인 원본 프레임 수
        if r <= 0:
            e = float(jmin[0])
        elif r >= n:
            e = float(jmax[n - 1] + 1)
        else:
            e = (jmax[r - 1] + 1 + jmin[r]) / 2.0
        k = int(np.floor(e))
        if e != k:                                           # 압축 프레임 k 한가운데
            tt = round(float(it[min(k, len(it) - 1)] * hop + off + hop / 2.0), 4)
            out.append((tt, tt))
            continue
        left = (it[k - 1] + 1) * hop + off if k > 0 else it[0] * hop + off
        right = it[k] * hop + off if k < len(it) else (it[-1] + 1) * hop + off
        out.append((round(float(min(left, right)), 4), round(float(max(left, right)), 4)))
    return out


def project(x_ref, x_take, times, ref_span=None, take_span=None, sr=16000):
    """MFCC-DTW 투영(계획서 6.4)."""
    return project_feats(mfcc(x_ref, sr), mfcc(x_take, sr), times, MFCC_HOP, MFCC_OFF, ref_span, take_span, "euclid")

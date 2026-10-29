# -*- coding: utf-8 -*-
"""녹음 앞뒤·중간의 빈 구간 잘라내기 — 추론 전에 쓴다.

왜 필요한가(실측 2026-09-23, 웹 녹음 42건):
    소리가 있는 구간의 비율과 음소 점수의 상관이 **+0.62** 였다.
    같은 "안녕하세요" 가 3.2초 녹음(소리 61%)에서는 점수 74, 8.4초 녹음(소리 36%)에서는 **1** 이었고
    자리 10개 중 9개가 "발음 안 함" 으로 판정됐다.
    CTC 강제정렬은 무음 구간에도 음소를 배치하므로, 빈 구간이 길수록 확신도가 무너진다.

방식은 단순한 에너지 기반이다(모델 불필요, 1ms 미만).
    30ms 프레임 RMS 가 임계 이상인 곳만 '소리' 로 보고, 앞뒤를 자른다.
    중간의 긴 침묵은 `max_gap` 까지만 남긴다 — 문장 사이 쉼은 유창성 판단에 쓰이므로 통째로 지우지 않는다.
임계는 **발화 최대 RMS 의 비율**로 잡아, 녹음 음량이 작아도 따라간다.
"""

import numpy as np

FRAME_SEC = 0.03
REL_THRESH = 0.06        # 프레임 RMS 가 (최대 RMS × 이 값) 이상이면 소리로 본다
ABS_FLOOR = 8e-4         # 전부 무음인 녹음에서 잡음을 소리로 세지 않기 위한 하한
KEEP_SEC = 0.15          # 잘라낸 뒤 앞뒤로 남기는 여유
MAX_GAP = 0.5            # 중간 침묵을 이보다 길게 두지 않는다


def frame_rms(wav, sr, frame_sec=FRAME_SEC):
    f = max(1, int(sr * frame_sec))
    n = len(wav) // f
    if n == 0:
        return np.zeros(0), f
    x = wav[: n * f].reshape(n, f).astype(np.float64)
    return np.sqrt((x * x).mean(axis=1)), f


def voiced_mask(wav, sr):
    r, f = frame_rms(wav, sr)
    if r.size == 0:
        return np.zeros(0, bool), f
    th = max(float(r.max()) * REL_THRESH, ABS_FLOOR)
    return r >= th, f


def trim(wav, sr, max_gap=MAX_GAP, keep=KEEP_SEC):
    """앞뒤 무음 제거 + 중간 침묵 단축. (잘린 wav, 정보) 를 돌려준다.

    소리가 하나도 없으면 원본을 그대로 돌려준다(무음 판정은 호출 쪽 몫).
    """
    m, f = voiced_mask(wav, sr)
    info = {"orig_sec": round(len(wav) / sr, 2), "voiced_ratio": round(float(m.mean()), 3) if m.size else 0.0}
    if not m.any():
        info.update(sec=info["orig_sec"], trimmed=False)
        return wav, info
    idx = np.flatnonzero(m)
    pad = int(keep / FRAME_SEC)
    lo, hi = max(0, idx[0] - pad), min(len(m) - 1, idx[-1] + pad)
    gap = int(max_gap / FRAME_SEC)
    keep_frames, run = [], 0
    for i in range(lo, hi + 1):
        if m[i]:
            run = 0
        else:
            run += 1
            if run > gap:            # 긴 침묵은 max_gap 까지만 남긴다
                continue
        keep_frames.append(i)
    out = np.concatenate([wav[i * f:(i + 1) * f] for i in keep_frames]) if keep_frames else wav[lo * f:(hi + 1) * f]
    info.update(sec=round(len(out) / sr, 2), trimmed=True,
                cut_sec=round((len(wav) - len(out)) / sr, 2))
    return out.astype(np.float32), info

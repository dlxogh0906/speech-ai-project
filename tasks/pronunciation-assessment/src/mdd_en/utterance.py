# -*- coding: utf-8 -*-
"""문장 하나의 특징 — 문장 점수(정확도·유창성·운율·종합) 회귀기의 입력.

발음 쪽: 음소·단어 점수의 평균·최저·분위수·낮은 점수 비율
시간 쪽: 강제정렬 구간에서 말한 시간·조음 속도(음소/초)·단어 사이 쉼(0.3초·0.5초 이상 개수·합·최장)
        ·앞뒤 무음 · 빠뜨린 음소 비율 · 인식이 표준과 다른 비율
CTC 정렬은 소리마다 한두 프레임만 짚어서(뾰족함) 단어 **사이 간격**만 쉼으로 본다.
"""

import numpy as np

from mdd_en.recognizer import POST_FRAME_SEC

UTT_COLS = ["n_phones", "n_words", "ph_mean", "ph_min", "ph_p10", "ph_p25", "ph_std", "ph_lt50", "ph_lt80",
            "w_mean", "w_min", "w_p25", "speech_sec", "art_rate", "word_rate", "pause3_n", "pause3_sec",
            "pause5_n", "pause_max", "lead_sec", "trail_sec", "del_frac", "mism_frac"]


def utt_features(phone_scores, word_of, spans, n_frames, deleted=None, mismatch=None,
                 frame_sec=POST_FRAME_SEC):
    """→ (len(UTT_COLS),) float32.

    phone_scores: 표준 음소마다 0~100 · word_of: 음소 → 단어 번호 · spans: 강제정렬 (시작, 끝) 프레임
    n_frames: 사후확률 프레임 수 · deleted/mismatch: 음소마다 0/1(없으면 0)
    frame_sec: 프레임 간격 — `phone_post` 는 2배로 늘린 10 ms 간격이다
    """
    s = np.asarray(phone_scores, np.float32)
    N = len(s)
    v = np.zeros(len(UTT_COLS), np.float32)
    if N == 0:
        return v
    w = np.asarray([-1 if x is None else x for x in word_of])
    words = [k for k in sorted(set(w.tolist())) if k >= 0]
    ws = np.array([s[w == k].mean() for k in words]) if words else s
    starts = [a for a, b in spans if b > a]
    ends = [b for a, b in spans if b > a]
    t0 = min(starts) if starts else 0
    t1 = max(ends) if ends else n_frames
    speech = max(t1 - t0, 1) * frame_sec
    gaps = []
    for k1, k2 in zip(words, words[1:]):
        e1 = [spans[i][1] for i in range(N) if w[i] == k1 and spans[i][1] > spans[i][0]]
        s2 = [spans[i][0] for i in range(N) if w[i] == k2 and spans[i][1] > spans[i][0]]
        if e1 and s2:
            gaps.append(max(0, min(s2) - max(e1)) * frame_sec)
    gaps = np.array(gaps) if gaps else np.zeros(1)
    p3 = gaps[gaps >= 0.3]
    pause_total = float(p3.sum())
    v[:] = [N, len(words), s.mean(), s.min(), np.percentile(s, 10), np.percentile(s, 25), s.std(),
            (s < 50).mean(), (s < 80).mean(), ws.mean(), ws.min(), np.percentile(ws, 25), speech,
            N / max(speech - pause_total, 0.2), len(words) / speech, len(p3), pause_total, (gaps >= 0.5).sum(),
            gaps.max(), t0 * frame_sec, max(n_frames - t1, 0) * frame_sec,
            float(np.mean(deleted)) if deleted is not None and len(deleted) else 0.0,
            float(np.mean(mismatch)) if mismatch is not None and len(mismatch) else 0.0]
    return v

# -*- coding: utf-8 -*-
"""음높이·세기·대역 에너지 곡선 (10 ms 간격).

원본과 녹음에 **같은 추출기**를 쓴다 — 계획서처럼 원본 RMVPE / 녹음 pYIN 으로 나누면
추출기 성향 차이(유성 판정·평활·옥타브 오류)가 그대로 "어긋남"으로 잡힌다.
음높이는 Praat 자기상관(parselmouth)을 두 번 돈다(Hirst 2011 변형): 넓은 범위로 한 번 → 사분위로
화자 범위를 좁혀 다시(하한 0.6·Q1 — 낭독용 0.75 는 성대모사의 깊은 내림을 무성으로 잘랐다).
그 뒤 국소 중앙값에서 튄 프레임을 ±12 반음 접거나 버린다.
"""
import numpy as np

HOP = 0.01


def _sound(wav, sr):
    import parselmouth
    return parselmouth.Sound(np.asarray(wav, dtype=np.float64), sampling_frequency=sr)


def pitch_track(wav, sr=16000):
    """→ (시간[s], f0[Hz], 쓴 범위(floor, ceiling)). 무성 프레임은 nan."""
    snd = _sound(wav, sr)
    p = snd.to_pitch_ac(time_step=HOP, pitch_floor=60.0, pitch_ceiling=800.0)
    f = p.selected_array["frequency"]
    rng = (60.0, 800.0)
    v = f[f > 0]
    if len(v) >= 20:
        q1, q3 = np.percentile(v, [25, 75])
        rng = (float(np.clip(0.6 * q1, 50.0, 300.0)), float(np.clip(2.0 * q3, 250.0, 1100.0)))
        p = snd.to_pitch_ac(time_step=HOP, pitch_floor=rng[0], pitch_ceiling=rng[1])
        f = p.selected_array["frequency"]
    f = f.astype(float)
    f[f <= 0] = np.nan
    return np.asarray(p.xs(), dtype=float), f, rng


def semitones(f_hz, ref_hz=100.0):
    return 12.0 * np.log2(np.asarray(f_hz, dtype=float) / ref_hz)


def _wmedian(vals, w):
    o = np.argsort(vals)
    c = np.cumsum(np.asarray(w, dtype=float)[o])
    return float(np.asarray(vals)[o][np.searchsorted(c, c[-1] / 2.0)])


def clean_octaves(st, jump=7.0, ctx=0.3, fold_tol=3.0, far_tol=2.0, cont=4.0, edge=5, min_run=3, iso_tol=2.0):
    """옥타브 오류(음높이를 두 배·절반으로 잡은 구간) 정리.

    진짜 억양은 이어지게 움직이고, 추적 오류는 한 프레임 사이에 ±12 반음을 뛴다. 그래서
      1. 유성 프레임을 무성 틈과 `jump` 반음 넘는 급점프에서 조각으로 나눈다.
      2. 조각의 끝과 이웃 조각의 맞닿은 끝을 견준다 — 각자 끝 `edge` 프레임에 직선을 맞춰 맞닿는 지점(최대 3 프레임
         밖까지)에서의 값끼리. 억양이 빠르게 움직이는 중에 두 배로 잡힌 조각(212→171 Hz, 실제 106→86)은 끝 프레임
         중앙값으로 견주면 점프가 줄어 보인다. 무성 틈 없이 바로 붙은 이웃(급점프)이
         있으면 그것만 본다 — 틈 건너 이웃은 다음 음절이라 진짜로 음높이가 다를 수 있다. 붙은 이웃이 없으면 `ctx` 초
         안의 틈 건너 이웃을 보되 더 엄격하게(12±`far_tol`) — 음절 사이 9~10 반음 진짜 도약을 접지 않게.
         본 이웃 중 하나라도 12±허용 반음 차이이고, 틈 건너 이웃까지 어느 이웃과도 이어지지 않으며(차이 `cont` 반음 이내인
         이웃 없음 — 붙은 이웃이 오류일 수 있다: Zeroth 낭독의 332 Hz 조각은 틈 건너 266 Hz 와 이어지는데 붙은 169 Hz(오류)
         만 보고 접었다), ±12 로 접으면 발화 중앙값에도 가까워질 때만 접는다 — 맞는 쪽을 잘못 접지 않게. 단 양쪽 이웃이
         모두 같은 방향 옥타브 차이면(사이에 낀 조각) 중앙값과 상관없이 접는다 — 음역이 넓은 화자는 중앙값이 가운데 걸린다.
         붙은 이웃이 있는 조각(근거가 센 것)부터, 그 안에선 짧은 조각부터 고치고 두 번 돈다 — 틈 건너 이웃만 보는 조각이
         아직 안 고친 오류 조각을 기준으로 삼지 않게(「느그 서장」 TTS '제': 2배로 잡힌 이웃을 기준으로 맞는 90 Hz 조각을
         180 Hz 로 접었다).
      3. 그래도 어느 이웃과도 안 이어지는 조각이 발화 중앙값보다 12±`iso_tol` 반음 **높고** 유성 프레임의 절반이 안 되면
         버린다(못 잼) — 배음이 센 목소리에서 틈 뒤 음절 하나를 통째로 2배로 잡은 것(「살아있네」 TTS '네' 216 Hz)이 이렇다.
         접지 않는 건 낭독에서 진짜로 높게 시작한 짧은 조각(스펙트럼에 반 주파수 홀수 배음이 없다)도 같은 모양이라서다.
         낮은 쪽은 진짜 긁는 소리(성대 튀김, 「하와이」 '가' 60 Hz)일 수 있어 그대로 둔다.
      0. `min_run` 프레임보다 짧은 유성 조각은 먼저 버린다(잡음에서 생긴 가짜 음높이 — 이웃 판정에도 안 쓴다).
    앞뒤 몇 프레임만 보던 첫 규칙은 80 ms 넘게 이어진 오류를, 조각 전체 중앙값과 견주던 둘째 규칙은 억양이
    움직이는 중에 뛴 오류(TTS·갈라지는 소리 끝의 86→178 Hz 같은)를 못 고쳤다."""
    st = np.array(st, dtype=float)
    n = len(st)
    voiced = ~np.isnan(st)
    segs, i = [], 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        while j < n and voiced[j] and abs(st[j] - st[j - 1]) <= jump:
            j += 1
        segs.append([i, j])
        i = j
    out = st.copy()
    for a, b in segs:
        if b - a < min_run:
            out[a:b] = np.nan                                               # 너무 짧은 조각은 이웃 판정에도 안 쓴다
    segs = [s for s in segs if s[1] - s[0] >= min_run]
    W = int(round(ctx / HOP))

    def at_edge(a, b, head, x):
        idx = np.arange(a, min(b, a + edge)) if head else np.arange(max(a, b - edge), b)
        v = out[idx]
        if len(idx) < 3:
            return float(np.median(v))
        k, c = np.polyfit(idx.astype(float), v, 1)
        ref = idx[0] if head else idx[-1]
        return float(k * (ref + np.clip(x - ref, -3.0, 3.0)) + c)

    def neighbours(k):
        """→ (붙은 이웃과의 차들, 틈 건너 이웃과의 차들) — 이 조각 끝 − 이웃 끝(반음)."""
        a, b = segs[k]
        adj, far = [], []
        if k > 0 and segs[k - 1][1] >= a - W:
            pa, pb = segs[k - 1]
            x = (pb - 1 + a) / 2.0
            (adj if pb == a else far).append(at_edge(a, b, True, x) - at_edge(pa, pb, False, x))
        if k + 1 < len(segs) and segs[k + 1][0] <= b + W:
            na, nb = segs[k + 1]
            x = (b - 1 + na) / 2.0
            (adj if na == b else far).append(at_edge(a, b, False, x) - at_edge(na, nb, True, x))
        return adj, far

    def octave(d, tol):
        """차 d 가 몇 옥타브 관계인가(12±tol 반음 배수가 아니면 0)."""
        m = int(round(d / 12.0))
        return m if m and abs(d - 12.0 * m) <= tol else 0

    touch = [(k > 0 and segs[k - 1][1] == a) or (k + 1 < len(segs) and segs[k + 1][0] == b) for k, (a, b) in enumerate(segs)]
    lens = [b - a for a, b in segs]
    for _ in range(2):
        if not segs:
            break
        allmed = _wmedian([float(np.median(out[a:b])) for a, b in segs], lens)
        for k in sorted(range(len(segs)), key=lambda q: (not touch[q], lens[q])):   # 붙은 이웃 있는 것 → 짧은 것부터
            a, b = segs[k]
            adj, far = neighbours(k)
            if not adj and not far:
                continue
            if any(abs(d) <= cont for d in adj + far):
                continue                                                    # 어느 이웃과는 이어진다 → 진짜 억양
            oa, of = [octave(d, fold_tol) for d in adj], [octave(d, far_tol) for d in far]
            hits = [m for m in (oa if adj else of) if m]                    # 붙은 이웃이 있으면 그것으로 판정
            if not hits or len(set(hits)) != 1:
                continue
            m = hits[0]
            med = float(np.median(out[a:b]))
            sandwich = len(oa + of) >= 2 and all(x == m for x in oa + of)
            if sandwich or abs(med - 12.0 * m - allmed) < abs(med - allmed):
                out[a:b] -= 12.0 * m
    if segs:
        allmed = _wmedian([float(np.median(out[a:b])) for a, b in segs], lens)
        for k, (a, b) in enumerate(segs):
            adj, far = neighbours(k)
            if any(abs(d) <= cont for d in adj + far) or 2 * lens[k] >= sum(lens):
                continue
            if abs(float(np.median(out[a:b])) - allmed - 12.0) <= iso_tol:  # 한 옥타브 위에 혼자 떠 있는 조각
                out[a:b] = np.nan
    return out


def intensity_track(wav, sr=16000, min_pitch=75.0):
    """→ (시간[s], dB). Praat Intensity(평균 빼지 않음). 디지털 무음은 0 dB 로 자른다."""
    it = _sound(wav, sr).to_intensity(minimum_pitch=min_pitch, time_step=HOP, subtract_mean=False)
    return np.asarray(it.xs(), dtype=float), np.maximum(it.values[0].astype(float), 0.0)


def band_energy(wav, sr=16000, lo=300.0, hi=2500.0, win=0.025):
    """→ (시간[s], dB). 모음 대역 에너지 — 음절 사이 골짜기(자음·쉼)를 찾는 데 쓴다."""
    x = np.asarray(wav, dtype=float)
    n, h = int(win * sr), int(HOP * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    nf = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(nf)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    band = spec[:, (fr >= lo) & (fr <= hi)].sum(axis=1)
    return (np.arange(nf) * h + n / 2.0) / sr, 10.0 * np.log10(band + 1e-10)


def silence_mask(db):
    """세기 분포로 무음 문턱을 잡는다 → (무음 여부, 문턱 dB). 잡음 바닥과 최대값 사이 30% 또는 최대−35 dB."""
    lo, hi = np.percentile(db, [5, 98])
    thr = max(lo + 0.3 * (hi - lo), hi - 35.0)
    return db < thr, float(thr)


def runs(mask):
    """참인 구간들 → [(시작 인덱스, 끝 인덱스(제외))]."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

# -*- coding: utf-8 -*-
"""검증용 변형 — 정답을 아는 어긋남을 녹음에 넣는다(Praat PSOLA, parselmouth).

  - 방해 변형(nuisance): 목소리·전체 음높이·전체 빠르기·음량·잡음만 바꾼다. 지적이 **없어야** 한다.
  - 주입 변형(injected): 음절 하나의 음높이·억양 방향·길이·앞 쉼·세기를 바꾼다. **그 음절을 그 종류로** 짚어야 한다.
  - 녹음 조건(채널): 잔향·코덱·웅성거림·찌그러짐·좁은 대역·자동 음량 — 폰·방·카페 녹음 흉내(scripts/eval/audit_channel.py).
  - 지적대로 고치기(follow)·재녹음 흔들림(retake_jitter): 교정 루프가 끝나는지 본다(scripts/eval/audit_loop.py).
방해·주입 변형은 시간 사상(원래 시각 → 바뀐 시각)의 꺾은선 매듭 `tmap=(src, dst)` 을 같이 돌려준다 —
경계 오차(정렬 vs DTW)를 재는 정답이다.
"""
import numpy as np

SR = 16000
PRAAT_SEED = 0     # _seed 의 기본값 — 점검 스크립트가 바꿔 반복 실행의 흔들림을 본다(audit_loop --seed)


def _sound(x, sr=SR):
    import parselmouth
    return parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)


def _values(snd):
    return np.ascontiguousarray(snd.values[0], dtype=np.float32)


def _seed(seed=None):
    """Praat 난수를 고정한다 — PSOLA 가 무성 구간을 늘릴 때 조각을 무작위로 골라, 그대로 두면 같은 입력도 부를 때마다
    다른 파형이 나온다(교정 루프 점검이 실행마다 4~5쌍씩 달라졌다)."""
    from parselmouth.praat import run
    run("random_initializeWithSeedUnsafelyButPredictably (%d)" % (PRAAT_SEED if seed is None else seed))


def _trapezoid(t, t0, t1, ramp):
    """[t0, t1] 에서 1, 양끝 ramp 초 동안 0↔1."""
    w = np.clip((t - (t0 - ramp)) / ramp, 0, 1) * np.clip(((t1 + ramp) - t) / ramp, 0, 1)
    return w


def psola(x, pitch_fn=None, dur_points=None, prange=(75.0, 600.0), sr=SR):
    """PSOLA 재합성. pitch_fn(시각들, f0들) → 새 f0들. dur_points: [(시각, 길이배수)]."""
    from parselmouth.praat import call
    snd = _sound(x, sr)
    manip = call(snd, "To Manipulation", 0.01, float(prange[0]), float(prange[1]))
    if pitch_fn is not None:
        pt = call(manip, "Extract pitch tier")
        n = int(call(pt, "Get number of points"))
        ts = np.array([call(pt, "Get time from index", i) for i in range(1, n + 1)])
        fs = np.array([call(pt, "Get value at index", i) for i in range(1, n + 1)])
        new = pitch_fn(ts, fs) if n else fs
        npt = call("Create PitchTier", "p", snd.xmin, snd.xmax)
        for t, f in zip(ts, new):
            call(npt, "Add point", float(t), float(f))
        call([npt, manip], "Replace pitch tier")
    if dur_points:
        dt = call("Create DurationTier", "d", snd.xmin, snd.xmax)
        for t, f in dur_points:
            call(dt, "Add point", float(t), float(f))
        call([dt, manip], "Replace duration tier")
    _seed()
    return _values(call(manip, "Get resynthesis (overlap-add)"))


def ident_map(x, sr=SR):
    T = len(x) / sr
    return (np.array([0.0, T]), np.array([0.0, T]))


# ── 방해 변형 ────────────────────────────────────────────────

def nuisance(x, kind, prange, rng, high, sr=SR):
    """→ (변형 파형, tmap). kind: resynth · pitch+3 · pitch-3 · tempo0.8 · tempo1.25 · gender · noise20 · gain-15 · combo
    high: 원래 목소리가 높은 편인가(기준 음높이 ≥ 165 Hz) — gender 변형의 방향."""
    T = len(x) / sr
    if kind == "resynth":
        return psola(x, prange=prange), ident_map(x, sr)
    if kind.startswith("pitch"):
        s = float(kind[5:])
        return psola(x, pitch_fn=lambda t, f: f * 2 ** (s / 12.0), prange=prange), ident_map(x, sr)
    if kind.startswith("tempo"):
        k = float(kind[5:])
        y = psola(x, dur_points=[(0.0, k), (T, k)], prange=prange)
        return y, (np.array([0.0, T]), np.array([0.0, T * k]))
    if kind == "gender":
        return change_gender(x, prange, high, sr), ident_map(x, sr)
    if kind == "noise20":
        return add_noise(psola(x, prange=prange), 20.0, rng), ident_map(x, sr)
    if kind == "gain-15":
        return psola(x, prange=prange) * np.float32(10 ** (-15 / 20.0)), ident_map(x, sr)
    if kind == "combo":
        y = change_gender(x, prange, high, sr)
        y = psola(y, dur_points=[(0.0, 1.15), (T, 1.15)], prange=shifted_range(prange, high))
        return add_noise(y, 25.0, rng), (np.array([0.0, T]), np.array([0.0, T * 1.15]))
    raise KeyError(kind)


def shifted_range(prange, high):
    lo, hi = prange
    return (lo * 0.6, hi * 0.6) if high else (lo * 1.7, hi * 1.7)


def change_gender(x, prange, high, sr=SR):
    """높은 목소리 → 낮게(포먼트 ×0.85, 음높이 ×0.6), 낮은 목소리 → 높게(포먼트 ×1.18, 음높이 ×1.7). 길이는 그대로."""
    from parselmouth.praat import call
    import parselmouth
    snd = _sound(x, sr)
    med = call(snd.to_pitch_ac(time_step=0.01, pitch_floor=prange[0], pitch_ceiling=prange[1]),
               "Get quantile", 0, 0, 0.5, "Hertz")
    if not np.isfinite(med):
        med = 150.0
    fr, pf = (0.85, 0.6) if high else (1.18, 1.7)
    _seed()
    out = call(snd, "Change gender", float(prange[0]), float(prange[1]), fr, float(med * pf), 1.0, 1.0)
    y = _values(out)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def add_noise(x, snr_db, rng):
    x = np.asarray(x, dtype=np.float32)
    p = float(np.mean(x[np.abs(x) > 0.02 * np.max(np.abs(x))] ** 2)) if np.any(x) else 1e-6
    n = rng.standard_normal(len(x)).astype(np.float32) * np.float32(np.sqrt(p / 10 ** (snr_db / 10.0)))
    return x + n


# ── 주입 변형 (음절 하나) ───────────────────────────────────

DEFAULT_MAG = {"pitch_up": 4.0, "pitch_down": 4.0, "contour": 5.0, "longer": 1.8, "shorter": 0.5,
               "pause_add": 0.4, "weaker": 10.0}


def inject(x, kind, syl, prange, sr=SR, ref_delta=None, mag=None):
    """syl: analyze() 의 음절 dict. → (파형, tmap, 정답 dict(type, dir, mag)).

    mag(크기): 음높이 ±반음 · 억양 목표 변화 반음(평평한 음절에 넣는 오름) · 길이 배수 · 쉼 초 · 세기 −dB.
    억양은 원본이 이미 오르내리면 그 반대 방향으로 max(mag−1, |원래 변화|) 만큼 — 목표가 '반대 방향'이라서다."""
    m = DEFAULT_MAG[kind] if mag is None else float(mag)
    t0, t1 = syl["start"], syl["end"]
    T = len(x) / sr
    if kind in ("pitch_up", "pitch_down"):
        s = m if kind == "pitch_up" else -m
        y = psola(x, pitch_fn=lambda t, f: f * 2 ** (s * _trapezoid(t, t0 + 0.01, t1 - 0.01, 0.02) / 12.0), prange=prange)
        return y, ident_map(x, sr), dict(type="pitch_level", dir="high" if s > 0 else "low", mag=m)
    if kind == "contour":
        d0 = ref_delta or 0.0
        target = -np.sign(d0) * max(m - 1.0, abs(d0)) if abs(d0) >= 1.5 else m
        amp = 1.5 * (target - d0)               # 측정(끝 1/3 − 앞 1/3)은 선형 경사의 2/3 만 본다

        def fn(t, f):
            ramp = amp * ((t - t0) / max(1e-3, t1 - t0) - 0.5)
            return f * 2 ** (ramp * _trapezoid(t, t0 + 0.01, t1 - 0.01, 0.015) / 12.0)
        y = psola(x, pitch_fn=fn, prange=prange)
        return y, ident_map(x, sr), dict(type="pitch_dir", dir="rise" if target > 0 else "fall", target=float(target),
                                         mag=m)
    if kind in ("longer", "shorter"):
        k = m
        pts = [(t0 - 0.005, 1.0), (t0, k), (t1, k), (t1 + 0.005, 1.0)]
        y = psola(x, dur_points=pts, prange=prange)
        src = np.array([0.0, t0, t1, T])
        dst = np.array([0.0, t0, t0 + k * (t1 - t0), T + (k - 1) * (t1 - t0)])
        return y, (src, dst), dict(type="duration", dir="long" if k > 1 else "short", mag=m)
    if kind == "pause_add":
        y = insert_silence(x, t0, m, sr)
        return y, (np.array([0.0, t0, t0 + 1e-4, T]), np.array([0.0, t0, t0 + m + 1e-4, T + m])), \
            dict(type="pause", dir="extra", mag=m)
    if kind == "weaker":
        t = np.arange(len(x)) / sr
        g = 10 ** (-m * _trapezoid(t, t0, t1, 0.015) / 20.0)
        return (x * g).astype(np.float32), ident_map(x, sr), dict(type="energy", dir="weak", mag=m)
    raise KeyError(kind)


# ── 현실 녹음 조건(채널) — 폰·방·카페에서 녹음하면 생기는 것 ─────────────────
# 시간 사상은 그대로(코덱 지연 몇 ms 는 무시)라 경계 정답이 필요 없는 지적 수·측정 오차 점검에 쓴다.

def mix_at(x, noise, snr_db):
    """말소리 활성 구간 세기(add_noise 와 같은 정의) 기준으로 noise 를 snr_db 에 맞춰 더한다."""
    x = np.asarray(x, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)[:len(x)]
    p = float(np.mean(x[np.abs(x) > 0.02 * np.max(np.abs(x))] ** 2)) if np.any(x) else 1e-6
    pn = float(np.mean(noise ** 2)) + 1e-12
    return x + noise * np.float32(np.sqrt(p / pn / 10 ** (snr_db / 10.0)))


def loop_to(x, n, rng):
    """x 를 이어 붙여 길이 n 의 임의 구간으로."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / float(len(x)))) + 1)
    s = int(rng.integers(0, len(x) - n + 1))
    return x[s:s + n]


def reverb(x, rt60, rng, drr_db=6.0, sr=SR):
    """합성 잔향 — 직접음 + 지수 감쇠 잡음 꼬리(RT60 초). drr_db = 직접음 대 잔향 에너지 비(클수록 마이크가 가깝다)."""
    from scipy.signal import fftconvolve
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    h[:int(0.003 * sr)] = 0.0                                  # 첫 반사 전 3 ms
    h *= np.sqrt(10 ** (-drr_db / 10.0) / (np.sum(h ** 2) + 1e-12))
    h[0] += 1.0
    x = np.asarray(x, dtype=np.float64)
    y = fftconvolve(x, h)[:len(x)]
    return (y * np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12)).astype(np.float32)


def codec(x, fmt="opus", kbps=24, sr=SR):
    """ffmpeg 로 압축했다 풀기 — fmt: opus(브라우저 MediaRecorder) · aac(iOS). 길이는 원래대로 맞춘다."""
    import subprocess
    from shared.paths import FFMPEG
    from .audio import decode_bytes, wav_bytes
    enc = {"opus": ["-c:a", "libopus", "-b:a", "%dk" % kbps, "-f", "ogg"],
           "aac": ["-c:a", "aac", "-b:a", "%dk" % kbps, "-f", "adts"]}[fmt]
    r = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-f", "wav", "-i", "pipe:0", *enc, "pipe:1"],
                       input=wav_bytes(x, sr), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("코덱 변환 실패: %s" % r.stderr.decode("utf-8", "replace")[-200:])
    y = decode_bytes(r.stdout, sr)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def clip(x, over_db=12.0):
    """입력이 너무 커서 찌그러진 녹음 — 최대값이 over_db 만큼 넘치게 키워 ±1 에서 자르고 원래 크기로."""
    x = np.asarray(x, dtype=np.float32)
    g = 10 ** (over_db / 20.0) / (float(np.max(np.abs(x))) + 1e-9)
    return (np.clip(x * g, -1.0, 1.0) / g).astype(np.float32)


def narrowband(x, sr=SR):
    """전화·블루투스 헤드셋 대역 — 8 kHz 표본화 + 300~3400 Hz."""
    from scipy.signal import butter, resample_poly, sosfiltfilt
    y = resample_poly(np.asarray(x, dtype=np.float64), 1, 2)
    y = sosfiltfilt(butter(4, [300.0, 3400.0], btype="band", fs=sr / 2, output="sos"), y)
    y = resample_poly(y, 2, 1)[:len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)


def agc(x, sr=SR, thr_db=-35.0, ratio=3.0, attack=0.01, release=0.3, level_db=-25.0):
    """폰·브라우저 자동 음량(AGC) 흉내 — 활성 구간 세기를 level_db 로 맞춘 뒤 5 ms 세기 엔벨로프가 문턱을 넘으면
    ratio:1 로 누르고, 다시 level_db 로 맞춘다. 음절 사이 세기 차가 줄어든다."""
    x = np.asarray(x, dtype=np.float64)

    def norm(y):
        a = y[np.abs(y) > 0.02 * np.max(np.abs(y))]
        return y * 10 ** (level_db / 20.0) / (np.sqrt(np.mean(a ** 2)) + 1e-12)
    x = norm(x)
    hop = int(0.005 * sr)
    nf = int(np.ceil(len(x) / float(hop)))
    lvl = 10 * np.log10(np.mean(np.pad(x, (0, nf * hop - len(x))).reshape(nf, hop) ** 2, axis=1) + 1e-10)
    a_att, a_rel = np.exp(-hop / float(sr) / attack), np.exp(-hop / float(sr) / release)
    env, e = np.empty(nf), lvl[0]
    for i, v in enumerate(lvl):
        a = a_att if v > e else a_rel
        e = a * e + (1 - a) * v
        env[i] = e
    gain = np.where(env > thr_db, -(env - thr_db) * (1 - 1.0 / ratio), 0.0)
    g = np.interp(np.arange(len(x)), np.arange(nf) * hop + hop / 2.0, 10 ** (gain / 20.0))
    return norm(x * g).astype(np.float32)


def insert_silence(x, t0, dur, sr=SR):
    """t0 초에 dur 초 무음을 끼운다(양쪽 5 ms 페이드)."""
    pos = int(round(t0 * sr))
    gap, fade = int(dur * sr), int(0.005 * sr)
    a, b = x[:pos].copy(), x[pos:].copy()
    if len(a) > fade:
        a[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    if len(b) > fade:
        b[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    return np.concatenate([a, np.zeros(gap, dtype=np.float32), b])


def remove_span(x, t0, t1, sr=SR):
    """[t0, t1) 초를 잘라 낸다(쉼 줄이기 — 잘린 자리 양쪽 5 ms 페이드)."""
    a, b = x[:int(round(t0 * sr))].copy(), x[int(round(t1 * sr)):].copy()
    fade = int(0.005 * sr)
    if len(a) > fade:
        a[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    if len(b) > fade:
        b[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    return np.concatenate([a, b]).astype(np.float32)


# ── 지적대로 고치기 — 교정 루프 점검("시킨 대로 하면 통과하나") ──────────────

def _dur_points(spans):
    """[(시작, 끝, 길이배수)] (시간순·안 겹침) → DurationTier 점. 바깥 가장자리만 5 ms 경사, 이웃 음절끼리는 바로 잇는다."""
    pts = []
    for j, (s, e, k) in enumerate(spans):
        if not pts or pts[-1][0] < s - 0.006:
            pts.append((s - 0.005, 1.0))
        pts += [(s, k), (e - 0.001, k)]
        if j + 1 == len(spans) or spans[j + 1][0] > e + 0.006:
            pts.append((e + 0.005, 1.0))
    return pts


def follow(y, take, iss, prange, amount=1.0, sr=SR):
    """compare 의 지적 하나를 녹음에 반영한다 — 사용자가 시킨 대로 고쳐 다시 녹음했다고 치는 것.
    take: 이 녹음의 analyze 결과(음절 시각) · amount: 고친 정도(1 = 잰 차이만큼 다, 0.7 = 70% 만)."""
    t = iss["type"]
    T = len(y) / sr
    if t == "tempo":
        k = (1.0 / iss["diff"]) ** amount
        return psola(y, dur_points=[(0.0, k), (T, k)], prange=prange)
    S = take["syllables"]
    sp = [(S[m["i"]], m) for m in (iss.get("members") or [iss])]
    if t == "pitch_level":
        def fn(tt, f):
            s = sum(-m["diff"] * amount * _trapezoid(tt, x["start"] + 0.01, x["end"] - 0.01, 0.02) for x, m in sp)
            return f * 2 ** (s / 12.0)
        return psola(y, pitch_fn=fn, prange=prange)
    if t == "pitch_dir":
        def fn(tt, f):
            s = 0.0
            for x, m in sp:
                amp = 1.5 * (m["ref"] - m["take"]) * amount       # inject(contour) 와 같은 환산 — 잰 변화는 경사의 2/3
                s = s + amp * ((tt - x["start"]) / max(1e-3, x["end"] - x["start"]) - 0.5) * \
                    _trapezoid(tt, x["start"] + 0.01, x["end"] - 0.01, 0.015)
            return f * 2 ** (s / 12.0)
        return psola(y, pitch_fn=fn, prange=prange)
    if t == "duration":
        spans = sorted((x["start"], x["end"], (1.0 / m["diff"]) ** amount) for x, m in sp)
        return psola(y, dur_points=_dur_points(spans), prange=prange)
    if t == "energy":
        tt = np.arange(len(y)) / sr
        g = sum(-m["diff"] * amount * _trapezoid(tt, x["start"], x["end"], 0.015) for x, m in sp)
        return (y * 10 ** (g / 20.0)).astype(np.float32)
    if t == "pause":
        i, d = iss["i"], iss["diff"] * amount                      # d = 녹음 쉼 − 원본 쉼
        if d < 0:
            return insert_silence(y, S[i]["start"], -d, sr)
        a, b = S[i - 1]["end"], S[i]["start"]
        cut = min(d, max(0.0, b - a - 0.02))
        return remove_span(y, (a + b - cut) / 2.0, (a + b + cut) / 2.0, sr)
    raise KeyError(t)


def retake_jitter(y, take, rng, prange, pitch_sd=0.7, dur_sd=0.08, energy_sd=1.5, sr=SR):
    """다시 녹음할 때마다 생기는 흔들림 — 음절마다 음높이(반음)·길이(log)·세기(dB)를 정규분포로 조금씩."""
    S = take["syllables"]
    ps, ds, es = rng.normal(0, pitch_sd, len(S)), np.exp(rng.normal(0, dur_sd, len(S))), rng.normal(0, energy_sd, len(S))

    def fn(tt, f):
        s = sum(ps[j] * _trapezoid(tt, x["start"] + 0.01, x["end"] - 0.01, 0.02) for j, x in enumerate(S))
        return f * 2 ** (s / 12.0)
    tt = np.arange(len(y)) / sr
    g = sum(es[j] * _trapezoid(tt, x["start"], x["end"], 0.015) for j, x in enumerate(S))
    y = (y * 10 ** (g / 20.0)).astype(np.float32)
    return psola(y, pitch_fn=fn, dur_points=_dur_points([(x["start"], x["end"], float(ds[j])) for j, x in enumerate(S)]),
                 prange=prange)


def map_time(tmap, t):
    return np.interp(t, tmap[0], tmap[1])


def compose(m1, m2):
    """먼저 m1, 그다음 m2 를 적용한 사상."""
    src = np.unique(np.concatenate([m1[0], np.interp(m2[0], m1[1], m1[0])]))
    return src, np.interp(np.interp(src, m1[0], m1[1]), m2[0], m2[1])

# -*- coding: utf-8 -*-
"""라벨과 무관한 채널 증강 — 학습 시 on-the-fly, numpy 만 쓴다.

왜: 국적 KD 모델(run5)이 **채널을 국적 단서로** 배웠다. 한국어 학습 데이터가 스튜디오·대화체(KsponSpeech · 001 녹음실)뿐이고
외국인 429,556발화는 브라우저 녹음(잡음 억제 · AGC · 앞뒤 무음)이라, 노트북 마이크로 녹음한 한국인이 미국으로 나왔다
(`results/11_채널편향_한국어.md`). 한국어에만 오프라인 증강(잡음 추가 · 저역통과)을 건 재학습은 그 증강으로 만든 test 만 맞히고
실제 녹음은 못 고쳤다. 여기서는 **모든 클래스**에 같은 분포의 변형을 걸어 채널이 라벨을 예측하지 못하게 한다.

구성(각각 독립 확률): 앞뒤 무음 다듬기/덧붙이기(디지털 무음 · 룸톤 · 자기 무음) · 주파수 응답(고역/저역 차단 · 기울기 · 피크/노치) ·
압축/AGC · 잡음 게이트(브라우저 잡음 억제 흉내 — 말 사이를 거의 디지털 무음으로) · 배경 잡음(백색/분홍/갈색) · 이득.
코덱(opus/aac)은 없다 — ffmpeg 호출은 로더에서 너무 느리다. 오프라인 증강 행(`kor_aug`)이 그 몫을 맡는다.

    from country_classifier.wav_augment import augment
    w2 = augment(w, np.random.default_rng(seed))        # w: float32 16 kHz mono numpy
"""
import numpy as np

SR = 16000
FRAME = 480          # 30 ms


def _frame_rms(w):
    n = w.size // FRAME
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.square(w[: n * FRAME].reshape(n, FRAME), dtype=np.float64).mean(1))


def _speech_bounds(w, thr=1.5e-3):
    """데모(nationality-kd-demo)와 같은 절대 문턱으로 말소리 구간의 (시작, 끝) 샘플. 없으면 None."""
    v = np.where(_frame_rms(w) >= thr)[0]
    if v.size == 0:
        return None
    return int(v[0] * FRAME), int(min(w.size, (v[-1] + 1) * FRAME))


def _voiced_rms(w):
    v = w[np.abs(w) > 1e-4]
    return float(np.sqrt(np.mean(np.square(v, dtype=np.float64)))) if v.size else 0.0


def _envelope(g_frames, n):
    env = np.repeat(g_frames, FRAME)[:n]
    if env.size < n:
        env = np.concatenate([env, np.full(n - env.size, env[-1] if env.size else 1.0)])
    return env


def _colored_noise(n, rng, kind):
    x = rng.standard_normal(n).astype(np.float32)
    if kind == "white" or n < 4:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    X *= (1.0 / np.sqrt(f)) if kind == "pink" else (1.0 / f)       # 분홍: 1/f 파워, 갈색: 1/f² 파워
    y = np.fft.irfft(X, n).astype(np.float32)
    return y / (float(np.sqrt(np.mean(np.square(y)))) + 1e-9)


def _edges(w, rng):
    """앞뒤 무음을 다듬거나(trim) 덧붙인다(pad). 덧붙이는 무음의 성격을 무작위로 — 디지털 무음 · 룸톤 · 발화 자신의 앞 무음."""
    b = _speech_bounds(w)
    mode = rng.choice(["trim", "pad", "both"], p=[0.35, 0.45, 0.20])
    src = w
    if b is not None and mode in ("trim", "both"):
        m = int(rng.uniform(0.05, 0.3) * SR)
        w = w[max(0, b[0] - m): min(w.size, b[1] + m)]
    if mode in ("pad", "both"):
        kind = rng.choice(["digital", "room", "own"], p=[0.4, 0.35, 0.25])
        if kind == "own" and not (b is not None and b[0] >= int(0.2 * SR)):
            kind = "room"
        lead, trail = int(rng.uniform(0.1, 1.2) * SR), int(rng.uniform(0.1, 0.8) * SR)
        level = 10 ** (-rng.uniform(40, 62) / 20)

        def seg(n):
            if kind == "digital":
                return (rng.standard_normal(n) * 10 ** (-rng.uniform(72, 92) / 20)).astype(np.float32)
            if kind == "room":
                return (_colored_noise(n, rng, rng.choice(["pink", "brown"])) * level).astype(np.float32)
            t = src[: b[0]]
            return np.tile(t, int(np.ceil(n / t.size)))[:n].astype(np.float32)
        w = np.concatenate([seg(lead), w, seg(trail)])
    return w


def _eq(w, rng):
    """마이크·브라우저 주파수 응답: 2차 고역통과(50~250 Hz) · 저역통과(2.5~7.5 kHz, 차수 무작위) · 기울기(±3 dB/oct) · 피크/노치 0~2개."""
    n = w.size
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    fs = np.maximum(f, 1e-3)
    g = 1.0 / np.sqrt(1.0 + (rng.uniform(50, 250) / fs) ** 4)
    g *= 1.0 / np.sqrt(1.0 + (f / rng.uniform(2500, 7500)) ** rng.uniform(4, 12))
    g *= (np.maximum(f, 50.0) / 1000.0) ** (rng.uniform(-3, 3) / 6.02)
    for _ in range(int(rng.integers(0, 3))):
        fc = rng.uniform(200, 6000)
        bw = fc / rng.uniform(1, 4)
        g *= 10 ** (rng.uniform(-8, 8) / 20 * np.exp(-0.5 * ((f - fc) / bw) ** 2))
    return np.fft.irfft(X * g, n).astype(np.float32)


def _compress_agc(w, rng):
    """프레임 포락선 압축(비율 2~6, 문턱 -30~-15 dB) 뒤 유성 RMS 를 -28~-16 dB 로 맞춘다(AGC)."""
    r = _frame_rms(w)
    if r.size == 0:
        return w
    env = np.convolve(_envelope(r, w.size), np.ones(FRAME) / FRAME, mode="same")
    thr = 10 ** (-rng.uniform(15, 30) / 20)
    ratio = rng.uniform(2, 6)
    gain = np.where(env > thr, (thr / np.maximum(env, 1e-9)) ** (1.0 - 1.0 / ratio), 1.0)
    w = (w * gain).astype(np.float32)
    vr = _voiced_rms(w)
    if vr > 1e-6:
        w = w * min(40.0, 10 ** (-rng.uniform(16, 28) / 20) / vr)
    return np.clip(w, -1, 1).astype(np.float32)


def _gate(w, rng):
    """잡음 게이트 — 브라우저 잡음 억제의 결과처럼 말 사이·앞뒤를 20~60 dB 낮춘다(거의 디지털 무음)."""
    r = _frame_rms(w)
    if r.size == 0:
        return w
    thr = (np.percentile(r, 95) + 1e-9) * 10 ** (-rng.uniform(25, 40) / 20)
    att = 10 ** (-rng.uniform(20, 60) / 20)
    g = np.where(r >= thr, 1.0, att)
    g = np.maximum(g, np.roll(g, 1))
    g = np.maximum(g, np.roll(g, -1))                       # 한 프레임 hold
    env = np.convolve(_envelope(g, w.size), np.ones(160) / 160, mode="same")      # 10 ms 전이
    return (w * env).astype(np.float32)


def _add_noise(w, rng):
    kind = rng.choice(["white", "pink", "brown"], p=[0.3, 0.5, 0.2])
    snr = rng.uniform(8, 40)
    vr = _voiced_rms(w) or 1e-3
    return (w + _colored_noise(w.size, rng, kind) * vr * 10 ** (-snr / 20)).astype(np.float32)


def augment(w, rng, p_edges=0.8, p_eq=0.7, p_comp=0.5, p_gate=0.4, p_noise=0.5, max_sec=None):
    """w: float32 16 kHz mono. 각 단계는 독립 확률로 켜진다. max_sec 를 주면 그 길이로 자른다."""
    w = np.asarray(w, dtype=np.float32)
    if w.size < SR // 4:
        return w
    if rng.random() < p_edges:
        w = _edges(w, rng)
    if rng.random() < p_eq:
        w = _eq(w, rng)
    if rng.random() < p_comp:
        w = _compress_agc(w, rng)
    if rng.random() < p_gate:
        w = _gate(w, rng)
    if rng.random() < p_noise:
        w = _add_noise(w, rng)
    w = np.clip(w * 10 ** (rng.uniform(-12, 6) / 20), -1, 1).astype(np.float32)
    if max_sec:
        w = w[: int(max_sec * SR)]
    return np.ascontiguousarray(w)


def silence_normalize(w, thr=1.5e-3, margin=0.15, lead=0.9, trail=0.7, level_db=-64.0, seed=0):
    """추론용 입력 정규화 — 앞뒤 무음을 말소리 기준으로 다듬고 **고정 룸톤**으로 바꾼다.

    왜: 녹음 도구가 남기는 무음의 성격(브라우저 잡음 억제의 디지털 무음 vs 스튜디오 룸톤)이 국적 판정을 기울였다
    (`results/11_채널편향_한국어.md` 무음 이식 실험). 무음 구간을 항상 같은 룸톤으로 바꾸면 그 단서가 입력에서 사라진다.
    외국인 정확도는 표본 240/120 에서 ±1~2점(잡음 범위), 사용자 노트북 녹음은 2/6 → 6/6. 말소리를 못 찾으면 그대로 돌려준다.
    """
    w = np.asarray(w, dtype=np.float32)
    b = _speech_bounds(w, thr)
    if b is None:
        return w
    core = w[max(0, b[0] - int(margin * SR)): min(w.size, b[1] + int(margin * SR))]
    rng = np.random.default_rng(seed)
    m = int((lead + trail) * SR)
    tone = (_colored_noise(m, rng, "pink") * 10 ** (level_db / 20)).astype(np.float32)
    k = int(lead * SR)
    return np.ascontiguousarray(np.concatenate([tone[:k], core, tone[k:]]).astype(np.float32))


# ── 부스 프로필 (2026-09-24): 전시장 군중 잡음 · 폰 대역 · 짧은 발화 ──
_BABBLE = {"paths": [], "cache": {}}


def set_babble_pool(paths):
    """군중 잡음 재료(원어민 대화 wav 목록). 워커마다 처음 쓸 때 몇 개만 메모리에 올린다."""
    _BABBLE["paths"] = list(paths)


def _babble(n, rng):
    import wave
    paths = _BABBLE["paths"]
    if not paths:
        return _colored_noise(n, rng, "pink")
    out = np.zeros(n, np.float32)
    for _ in range(4):
        p = paths[int(rng.integers(len(paths)))]
        if p not in _BABBLE["cache"]:
            if len(_BABBLE["cache"]) > 64:
                _BABBLE["cache"].clear()
            try:
                with wave.open(p, "rb") as f:
                    a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
            except Exception:
                a = _colored_noise(SR * 4, rng, "pink")
            _BABBLE["cache"][p] = a / (np.sqrt(np.mean(np.square(a))) + 1e-9)
        a = _BABBLE["cache"][p]
        s0 = int(rng.integers(0, max(1, a.size - n))) if a.size > n else 0
        seg = a[s0:s0 + n]
        if seg.size < n:
            seg = np.tile(seg, int(np.ceil(n / max(seg.size, 1))))[:n]
        out += seg
    return out / (np.sqrt(np.mean(np.square(out))) + 1e-9)


def _phone_band(w, rng):
    n = w.size
    X = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1.0 / SR); fs = np.maximum(f, 1e-3)
    g = 1.0 / np.sqrt(1.0 + (rng.uniform(250, 350) / fs) ** 4) / np.sqrt(1.0 + (f / rng.uniform(3200, 3600)) ** 8)
    return np.fft.irfft(X * g, n).astype(np.float32)


def augment_booth(w, rng, p_babble=0.6, p_phone=0.3, p_crop=0.3, max_sec=None):
    """부스 조건 증강: 군중 잡음 3~20 dB(0.6) · 폰 대역(0.3) · 1~4초로 자르기(0.3) · 기본 채널 증강(augment) 뒤에 건다."""
    w = augment(w, rng, max_sec=max_sec)
    if rng.random() < p_crop:
        b = _speech_bounds(w)
        a, e = b if b else (0, w.size)
        n = int(rng.uniform(1.0, 4.0) * SR)
        if e - a > n:
            s0 = a + int(rng.integers(0, e - a - n + 1)); w = w[s0:s0 + n]
    if rng.random() < p_phone:
        w = _phone_band(w, rng)
    if rng.random() < p_babble:
        snr = rng.uniform(3, 20); vr = _voiced_rms(w) or 1e-3
        w = np.clip(w + _babble(w.size, rng) * vr * 10 ** (-snr / 20), -1, 1).astype(np.float32)
    return np.ascontiguousarray(w)

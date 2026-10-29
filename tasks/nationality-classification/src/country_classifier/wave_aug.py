# -*- coding: utf-8 -*-
"""학습용 파형 증강 — 이미지 증강의 음성판 (2026-10-20).

  이미지                 음성 (여기)
  RandomResizedCrop  →  crop   무작위 위치 · 무작위 길이로 자르기 (학습은 원래 녹음 앞 max_sec 초만 본다)
  Resize             →  speed  속도 f 배 (길이 1/f · 음높이 · 포먼트가 f 배 — 목소리가 바뀐다, Kaldi 식 속도 섭동)
  ColorJitter        →  gain   음량 ±dB
                        eq     무작위 쉘빙 · 피킹 필터 (음색)
                        noise  백색 · 분홍 잡음, 무작위 SNR
  CutMix             →  cutmix_batch  배치 안 다른 발화의 한 구간을 오려 붙이고 라벨을 길이 비율로 섞는다

AugmentWaveDataset 은 AudioCountryDataset 을 감싸 학습 행에만 쓴다(valid · test 는 그대로).
발화마다 (seed, epoch, 행 번호)로 난수를 정해 작업자 수와 상관없이 같은 결과가 나온다.
"""

import math

import numpy as np

OPS = ("crop", "speed", "gain", "eq", "noise")


def speed_perturb(x, f):
    """속도 f 배 — 다상 리샘플(안티에일리어싱 포함). f 는 0.01 단위로 반올림한다."""
    from scipy.signal import resample_poly
    down = int(round(100 * f))
    if down == 100 or len(x) < 2:
        return x
    g = math.gcd(100, down)
    return resample_poly(x, 100 // g, down // g).astype(np.float32)


def biquad(kind, f0, gain_db, q, sr):
    """RBJ 쿡북 쌍2차 필터 계수 (b, a) — kind: peak · lowshelf · highshelf."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    if kind == "peak":
        al = sw / (2 * q)
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    else:
        al = sw / 2 * math.sqrt(2)                        # 기울기 S = 1
        s = 2 * math.sqrt(A) * al
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
            a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s)]
            a = [(A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s]
    return np.array(b) / a[0], np.array(a) / a[0]


def random_eq(x, rng, max_db, sr):
    """저역 쉘빙 · 고역 쉘빙 · 피킹 1개를 각각 반반 확률로, 이득 ±max_db."""
    from scipy.signal import lfilter
    bands = [("lowshelf", rng.uniform(100, 400), 0.707), ("highshelf", rng.uniform(2000, 6000), 0.707),
             ("peak", rng.uniform(300, 4000), rng.uniform(0.7, 2.0))]
    for kind, f0, q in bands:
        if rng.random() < 0.5:
            b, a = biquad(kind, f0, rng.uniform(-max_db, max_db), q, sr)
            x = lfilter(b, a, x)
    return np.asarray(x, np.float32)


def add_noise(x, rng, snr_lo, snr_hi):
    """백색 또는 분홍(1/f) 잡음을 SNR U(snr_lo, snr_hi) dB 로 더한다 — 신호 세기는 발화 전체 RMS."""
    n = rng.standard_normal(len(x))
    if rng.random() < 0.5:
        X = np.fft.rfft(n)
        k = np.arange(len(X)); k[0] = 1
        n = np.fft.irfft(X / np.sqrt(k), len(x))
    p_sig = float(np.mean(x.astype(np.float64) ** 2))
    p_n = float(np.mean(n ** 2))
    if p_sig <= 0 or p_n <= 0:
        return x
    snr = rng.uniform(snr_lo, snr_hi)
    return (x + n * math.sqrt(p_sig / (p_n * 10 ** (snr / 10)))).astype(np.float32)


def augment(x, rng, cfg, sr, max_n):
    """한 발화 증강. cfg 는 dict — ops(집합) · p · speed · crop_scale · crop_min_sec · gain_db · eq_db · noise_p · snr.
    아무 증강도 안 걸리면 x[:max_n] 그대로(원래 학습과 같다)."""
    ops, p = cfg["ops"], cfg["p"]
    f = rng.uniform(1 - cfg["speed"], 1 + cfg["speed"]) if "speed" in ops and rng.random() < p else 1.0
    f = round(f, 2)
    avail = len(x)
    full = min(int(avail / f), max_n)                    # 증강 뒤 최대 길이
    out_n, off = full, 0
    if "crop" in ops and rng.random() < p:
        min_n = int(cfg["crop_min_sec"] * sr)
        if full > min_n:
            out_n = max(min_n, int(rng.uniform(cfg["crop_scale"], 1.0) * full))
        off = int(rng.integers(0, max(1, avail - int(math.ceil(out_n * f)) + 1)))
    seg = x[off: off + int(math.ceil(out_n * f))]
    if f != 1.0:
        seg = speed_perturb(seg, f)
    seg = seg[:out_n]
    if "gain" in ops and rng.random() < p:
        seg = seg * np.float32(10 ** (rng.uniform(-cfg["gain_db"], cfg["gain_db"]) / 20))
        peak = float(np.abs(seg).max()) if len(seg) else 0.0
        if peak > 0.99:
            seg = seg * np.float32(0.99 / peak)
    if "eq" in ops and rng.random() < p:
        seg = random_eq(seg, rng, cfg["eq_db"], sr)
    if "noise" in ops and rng.random() < cfg["noise_p"]:
        seg = add_noise(seg, rng, *cfg["snr"])
    return np.ascontiguousarray(seg, dtype=np.float32)


class AugmentWaveDataset:
    """AudioCountryDataset(자르지 않은 것, max_duration_sec=0)을 감싸 augment() 를 건다. set_epoch() 로 epoch 마다 다른 증강."""

    def __init__(self, base, cfg, max_sec, sr=16000, seed=0):
        self.base, self.cfg, self.max_n, self.sr, self.seed = base, cfg, int(max_sec * sr), sr, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import torch
        item = self.base[idx]
        if item.get("_skip"):
            return item
        rng = np.random.default_rng([self.seed, self.epoch, int(idx)])
        x = augment(item["waveform"].numpy(), rng, self.cfg, self.sr, self.max_n)
        return dict(item, waveform=torch.from_numpy(x))


def cutmix_batch(wav, pm, rng, prob, alpha=1.0, fade=160, min_seg=1600):
    """배치 CutMix — 확률 prob 로, 발화 i 의 한 구간을 짝 j(무작위 순열)의 같은 길이 구간으로 바꾼다.

    잘라 붙일 비율 r ~ 1 − Beta(alpha, alpha) 를 배치마다 하나 뽑고, 구간 길이 = r · min(길이 i, 길이 j).
    붙인 경계는 fade 표본(10 ms)만큼 겹쳐 잇는다. 반환 (wav, perm, lam) — lam[i] = 발화 i 원래 라벨의 몫
    (마스크 평균 풀링이라 길이 비율이 곧 섞인 비율이다). 안 걸리면 lam = 1 · perm = 항등.
    rng = random.Random. wav · pm 은 같은 장치의 (B, S) 텐서, pm True = 패딩.
    """
    import torch
    B = wav.shape[0]
    lam = torch.ones(B, device=wav.device)
    perm = torch.arange(B, device=wav.device)
    if B < 2 or prob <= 0 or rng.random() >= prob:
        return wav, perm, lam
    r = 1.0 - rng.betavariate(alpha, alpha)
    order = list(range(B)); rng.shuffle(order)
    perm = torch.tensor(order, device=wav.device)
    lengths = (~pm).sum(1).tolist()
    out = wav.clone()
    for i, j in enumerate(order):
        Li, Lj = lengths[i], lengths[j]
        s = int(r * min(Li, Lj))
        if i == j or s < min_seg:
            continue
        a, b = rng.randint(0, Li - s), rng.randint(0, Lj - s)
        w = torch.ones(s, device=wav.device, dtype=wav.dtype)
        k = min(fade, s // 4)
        if k > 0:
            ramp = torch.linspace(0, 1, k, device=wav.device, dtype=wav.dtype)
            w[:k] = ramp; w[-k:] = ramp.flip(0)
        out[i, a:a + s] = (1 - w) * wav[i, a:a + s] + w * wav[j, b:b + s]
        lam[i] = 1.0 - (float(w.sum()) / Li)
    return out, perm, lam

# -*- coding: utf-8 -*-
"""국적 분류에 음원 분리(htdemucs)가 필요한가 — 시험용 음성을 만든다(잡음 섞기 → htdemucs 목소리 분리).

backbone_compare 의 test 에서 클래스 고르게 --n 개를 뽑아, 조건마다 16 kHz wav 와 목록 csv 를 만든다.
  clean                 원본
  clean_sep             원본 → htdemucs 목소리   (깨끗한 음성을 분리하면 억양 단서가 상하는지)
  music{,v}_SNR         배경음악(FMA; music = 연주곡, musicv = 노래 있는 곡) 섞음, SNR dB
  music{,v}_SNR_sep     → htdemucs 목소리
  babble_SNR(_sep)      다른 사람 말소리 4명(Kspon 학습분 한국어)
  noise_SNR(_sep)       생활 잡음(MUSAN noise)
잡음 원본은 DATA_ROOT/noise_pools/{music,musicv,babble,noise}/*.wav (16 kHz, 출처 sources.json).
분류는 backbone_compare.py evalset 이 한다(다른 환경). 이 스크립트는 demucs 가 깔린 환경에서 돈다.

    <demucs env python> .../noise_separation_prep.py --n 2000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import collections
import csv
import json
import random
import time

import numpy as np

from shared import paths as P

SR = 16000


def read(path):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def pick_noise(pool, n, rnd):
    """pool 에서 길이 n 조각(짧으면 이어 붙임)."""
    out = np.zeros(0, np.float32)
    while len(out) < n:
        x = read(rnd.choice(pool))
        if len(x) > n:
            s = rnd.randrange(0, len(x) - n)
            x = x[s:s + n]
        out = np.concatenate([out, x])
    return out[:n]


def mix(speech, noise, snr):
    ps = float(np.mean(speech ** 2)) + 1e-10
    pn = float(np.mean(noise ** 2)) + 1e-10
    y = speech + noise * np.sqrt(ps / (pn * 10 ** (snr / 10.0)))
    peak = float(np.abs(y).max())
    y = y / peak * 0.95 if peak > 0.99 else y
    return y.astype(np.float32)                                             # numpy 2 는 float64 스칼라를 곱하면 float64 가 된다


class Separator(object):
    def __init__(self, name, device):
        import torch
        from demucs.pretrained import get_model
        self.m = get_model(name).to(device).eval()
        self.device = device
        self.vi = self.m.sources.index("vocals")

    def __call__(self, x):
        import torch
        import torchaudio.functional as AF
        from demucs.apply import apply_model
        w = AF.resample(torch.from_numpy(np.asarray(x, np.float32)), SR, self.m.samplerate)
        w = w[None].repeat(2, 1)                                            # 모노 → 스테레오
        ref = w.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(self.m, ((w - mu) / sd)[None], device=self.device, split=True, overlap=0.25, progress=False)[0]
        v = (out[self.vi] * sd + mu).mean(0).cpu()
        y = AF.resample(v, self.m.samplerate, SR).numpy().astype(np.float32)
        return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]            # 리샘플 왕복 ±1 샘플 맞춤


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--snrs", default="10,5,0")
    ap.add_argument("--kinds", default="music,musicv,babble,noise")
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    import torch
    root = P.NAT_OUTPUTS / "backbone_compare_20261017"
    out = root / "noise_sep"
    out.mkdir(exist_ok=True)
    rnd = random.Random(a.seed)

    train_labels = {r["label"] for r in csv.DictReader(open(str(root / "manifest_train.csv"), encoding="utf-8"))}
    test = [r for r in csv.DictReader(open(str(root / "manifest_test.csv"), encoding="utf-8")) if r["label"] in train_labels]
    by = collections.defaultdict(list)
    for r in test:
        by[r["label"]].append(r)
    per = max(1, a.n // len(by))
    rows = []
    for lab in sorted(by):                                                  # 클래스 고르게
        rnd.shuffle(by[lab])
        rows += by[lab][:per]
    rnd.shuffle(rows)
    rows = rows[:a.n]

    pool_root = P.DATA_ROOT / "noise_pools"                                  # 출처는 noise_pools/sources.json
    pools = {k: sorted((pool_root / k).glob("*.wav")) for k in ("music", "musicv", "babble", "noise")}
    print("[prep] 시험 %d개(클래스 %d) · 음악 %d · 노래 %d · 말소리 %d · 잡음 %d" % (
        len(rows), len(by), len(pools["music"]), len(pools["musicv"]), len(pools["babble"]), len(pools["noise"])), flush=True)

    sep = Separator(a.model, torch.device("cuda"))
    conds = ["clean"] + ["%s_%s" % (k, s) for k in a.kinds.split(",") for s in a.snrs.split(",")]
    writers = {}
    for c in conds + [c + "_sep" for c in conds]:
        (out / c).mkdir(exist_ok=True)
        f = open(str(out / ("%s.csv" % c)), "w", encoding="utf-8", newline="")
        w = csv.DictWriter(f, fieldnames=["audio_path", "label", "speaker", "domain", "task", "duration_sec"])
        w.writeheader()
        writers[c] = (f, w)
    t0 = time.time()
    for i, r in enumerate(rows):
        x = read(r["audio_path"])[: int(a.max_sec * SR)]
        for c in conds:
            if c == "clean":
                y = x
            else:
                kind, snr = c.rsplit("_", 1)
                if kind == "babble":
                    nz = sum(pick_noise(pools["babble"], len(x), rnd) for _ in range(4))
                else:
                    nz = pick_noise(pools[kind], len(x), rnd)
                y = mix(x, nz, float(snr))
            for cc, z in ((c, y), (c + "_sep", sep(y))):
                p = out / cc / ("%05d.wav" % i)
                sf.write(str(p), z, SR, subtype="PCM_16")
                writers[cc][1].writerow({"audio_path": str(p), "label": r["label"], "speaker": r["speaker"], "domain": r["domain"],
                                         "task": r["task"], "duration_sec": "%.3f" % (len(z) / float(SR))})
        if i % 100 == 0:
            el = time.time() - t0
            print("[prep] %d/%d (%.0fs, 남은 %.0f분)" % (i, len(rows), el, el / (i + 1) * (len(rows) - i - 1) / 60), flush=True)
    for f, _ in writers.values():
        f.close()
    (out / "conditions.json").write_text(json.dumps({"conditions": conds, "n": len(rows), "model": a.model, "seed": a.seed,
                                                     "snrs": a.snrs, "kinds": a.kinds}, ensure_ascii=False, indent=1))
    print("[prep] 끝 %.1f분" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()

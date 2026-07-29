#!/usr/bin/env python
"""
Speaker-perturbation augmentation for long-tail classes (offline, on raw wav).

Implements the accent-ID style perturbation (Yang et al., arXiv:2306.06524):
  1) formant frequencies scaled by beta1
  2) F0 scaled by beta2
  3) a random frequency-shaping (peaking) EQ
beta1, beta2 ~ U(1, 1.4), each flipped to its reciprocal with prob 0.5.
Goal: change the SPEAKER timbre while leaving content/accent ~intact, i.e.
manufacture pseudo-speakers for classes that lack speaker diversity.

This runs on raw wav and writes new wav files + an extended CSV manifest, so it
plugs into your existing extract_features.py (no encoder change). It does NOT
touch valid/test. Augmented rows keep the ORIGINAL speaker_id (same person).

Primary backend: parselmouth (Praat). Fallback: librosa (pitch+formant approx).

Input CSV must have at least: audio_path, speaker_id, label_name, label_id
(extra columns are passed through). Filter it to your target tail classes first,
or use --target_labels.

Example:
  python augment_tail_wavs.py \
      --in_csv data/train_tail_targets.csv \
      --out_dir $SPEECH_AI_DATA/taeho/aug_wavs \
      --out_csv data/train_augmented.csv \
      --target_labels 인도네시아어,광동어,말레이어,스페인어,타갈로그어,네팔어,터키어 \
      --n_aug 8 --max_per_class 4000

Then: append out_csv to your train manifest, extract features for the new rows,
append those features to train.npz, and re-run the linear/decoupling probe.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import csv
import math
import os
import random

import numpy as np
from scipy.signal import sosfilt

try:
    import soundfile as sf
except ImportError:
    sf = None

SR = 16000


# --------------------------------------------------------------------- backends
def _draw_ratio(lo=1.0, hi=1.4):
    r = random.uniform(lo, hi)
    if random.random() < 0.5:
        r = 1.0 / r
    return r


def perturb_praat(in_path, formant_ratio, pitch_ratio):
    """Formant + F0 scaling via Praat 'Change gender'. Returns (mono_float32, SR)."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(in_path)
    if snd.n_channels > 1:
        snd = call(snd, "Convert to mono")
    if snd.sampling_frequency != SR:
        snd = call(snd, "Resample", SR, 50)
    # Change gender: (pitch_floor, pitch_ceiling, formant_shift, new_pitch_median,
    #                 pitch_range_factor, duration_factor). new_pitch_median=0 keeps
    #                 the median; pitch_range_factor acts as the F0 scaling.
    out = call(snd, "Change gender", 75, 600, formant_ratio, 0.0, pitch_ratio, 1.0)
    x = np.asarray(out.values).reshape(-1).astype(np.float32)
    return x, SR


def perturb_librosa(in_path, formant_ratio, pitch_ratio):
    """Fallback: pitch shift (F0) + resample trick to nudge formants. Coarser."""
    import librosa
    y, _ = librosa.load(in_path, sr=SR, mono=True)
    # F0: shift by semitones equivalent to pitch_ratio
    semis = 12.0 * math.log2(max(pitch_ratio, 1e-3))
    y = librosa.effects.pitch_shift(y, sr=SR, n_steps=semis)
    # crude formant nudge: resample then stretch back to length (vocal-tract-ish)
    if abs(formant_ratio - 1.0) > 1e-3:
        y2 = librosa.resample(y, orig_sr=SR, target_sr=int(SR * formant_ratio))
        y = librosa.effects.time_stretch(y2, rate=formant_ratio)
        y = librosa.util.fix_length(y, size=len(y))
    return y.astype(np.float32), SR


# ------------------------------------------------------------------------- EQ
def _peaking_sos(fs, f0, Q, gain_db):
    """RBJ cookbook peaking EQ as a single second-order section."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * Q)
    cosw = math.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cosw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw
    a2 = 1 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def random_eq(x, fs, n_bands=3):
    for _ in range(n_bands):
        f0 = random.uniform(200, min(6000, fs / 2 - 100))
        Q = random.uniform(0.7, 3.0)
        gain = random.uniform(-6.0, 6.0)
        x = sosfilt(_peaking_sos(fs, f0, Q, gain), x).astype(np.float32)
    # guard against clipping
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x


# ----------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True, help="where augmented wavs are written")
    ap.add_argument("--out_csv", required=True, help="manifest of augmented rows")
    ap.add_argument("--target_labels", default=None,
                    help="comma-separated label_name list; default = all rows in in_csv")
    ap.add_argument("--n_aug", type=int, default=8, help="augmented copies per source wav")
    ap.add_argument("--max_per_class", type=int, default=4000,
                    help="cap on TOTAL augmented rows per class")
    ap.add_argument("--backend", choices=["praat", "librosa"], default="praat")
    ap.add_argument("--no_eq", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio_col", default="audio_path")
    ap.add_argument("--label_col", default="label_name")
    args = ap.parse_args()

    if sf is None:
        raise SystemExit("soundfile required: pip install soundfile")
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    targets = set(args.target_labels.split(",")) if args.target_labels else None
    perturb = perturb_praat if args.backend == "praat" else perturb_librosa

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else []
    if "is_augmented" not in fieldnames:
        fieldnames = list(fieldnames) + ["is_augmented"]

    made = {}
    out_rows = []
    for r in rows:
        lab = r[args.label_col]
        if targets is not None and lab not in targets:
            continue
        if made.get(lab, 0) >= args.max_per_class:
            continue
        src = r[args.audio_col]
        if not os.path.exists(src):
            print(f"[skip] missing {src}")
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        for i in range(args.n_aug):
            if made.get(lab, 0) >= args.max_per_class:
                break
            b1, b2 = _draw_ratio(), _draw_ratio()
            try:
                x, fs = perturb(src, b1, b2)
            except Exception as e:
                print(f"[warn] perturb failed {src}: {e}")
                break
            if not args.no_eq:
                x = random_eq(x, fs)
            out_name = f"{stem}__aug{i}.wav"
            out_path = os.path.join(args.out_dir, out_name)
            sf.write(out_path, x, fs)
            nr = dict(r)
            nr[args.audio_col] = out_path
            nr["is_augmented"] = "1"
            # speaker_id stays the ORIGINAL speaker on purpose (same person)
            out_rows.append(nr)
            made[lab] = made.get(lab, 0) + 1

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for nr in out_rows:
            for k in fieldnames:
                nr.setdefault(k, "")
            w.writerow(nr)

    print(f"[done] {len(out_rows)} augmented rows over {len(made)} classes -> {args.out_csv}")
    for lab, n in sorted(made.items(), key=lambda kv: -kv[1]):
        print(f"  {lab}: +{n}")


if __name__ == "__main__":
    main()

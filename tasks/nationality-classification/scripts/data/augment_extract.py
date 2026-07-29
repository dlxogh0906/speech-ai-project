"""Audio-augment minority-class TRAIN utterances, re-extract XLS-R features.

Augmentations (cheap, create pseudo-speaker / robustness variation):
  speed 0.9, speed 1.1 (resample -> shifts formants+pitch, classic ASR aug),
  additive gaussian noise (SNR 15 dB), time masking (zero 2 x 0.3s).
Only classes with < --max_train_per_class real train utts are augmented (the strugglers;
dominant classes already work and don't need it).

Output: runs/features_aug/train_aug.npz with X, y, speaker, aug (for later viz/training).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, MODEL_ROOT, NAT_DATA, NAT_RUNS


import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np

from country_classifier import baseline_common as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default=f"{NAT_DATA}/aihub_nationality_fine_811")
    ap.add_argument("--out", default=f"{NAT_RUNS}/features_aug/train_aug.npz")
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--max_train_per_class", type=int, default=6000,
                    help="only augment classes with fewer than this many train utts")
    ap.add_argument("--cap_per_class", type=int, default=1500,
                    help="cap source utts per class to augment (keeps runtime bounded)")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--sample_rate", type=int, default=16000)
    args = ap.parse_args()

    import torch
    import torchaudio.functional as AF

    from country_classifier.audio import load_wav_mono
    from country_classifier.model import build_model

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    # which classes to augment + which source rows
    rows = list(csv.DictReader(open(Path(args.manifest_dir) / "train.csv", encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["label_id"], []).append(r)
    rng = np.random.default_rng(0)
    targets = []
    for lid, rs in by.items():
        if len(rs) < args.max_train_per_class:
            if len(rs) > args.cap_per_class:
                rs = [rs[i] for i in rng.choice(len(rs), args.cap_per_class, replace=False)]
            targets.extend(rs)
    print(f"augmenting {len(targets):,} src utts from "
          f"{sum(1 for lid,rs in by.items() if len(rs)<args.max_train_per_class)} classes", flush=True)

    print("loading XLS-R 2B ...", flush=True)
    model = build_model(backend="fairseq_xlsr", num_classes=61, checkpoint_path=args.xlsr_ckpt,
                        fairseq_root=args.fairseq_root, freeze_encoder=True).cuda().eval()

    def augment(w, kind):
        if kind == "spd0.9":  # slower/lower
            return AF.resample(w, sr, int(sr / 0.9))
        if kind == "spd1.1":  # faster/higher
            return AF.resample(w, sr, int(sr / 1.1))
        if kind == "noise":
            p = w.pow(2).mean().clamp_min(1e-9)
            n = torch.randn_like(w) * (p / (10 ** (15 / 10))).sqrt()  # SNR 15 dB
            return w + n
        if kind == "tmask":
            w = w.clone(); L = w.numel(); seg = int(0.3 * sr)
            for _ in range(2):
                if L > seg:
                    s = int(rng.integers(0, L - seg)); w[s:s + seg] = 0
            return w
        return w

    KINDS = ["spd0.9", "spd1.1", "noise", "tmask"]
    maxlen = int(args.max_sec * sr)
    X, Y, SP, AUG = [], [], [], []
    buf_w, buf_meta = [], []

    def flush():
        if not buf_w:
            return
        n = len(buf_w); ml = max(w.numel() for w in buf_w)
        wav = torch.zeros(n, ml, device="cuda"); mask = torch.ones(n, ml, dtype=torch.bool, device="cuda")
        for i, w in enumerate(buf_w):
            wav[i, :w.numel()] = w; mask[i, :w.numel()] = False
        with torch.no_grad(), torch.autocast("cuda"):
            f, fm = model._extract_features(wav, mask)
            pooled = model._pool(f, fm)
        p = pooled.float().cpu().numpy().astype(np.float16)
        for i, (lid, spk, kind) in enumerate(buf_meta):
            X.append(p[i]); Y.append(int(lid)); SP.append(spk); AUG.append(kind)
        buf_w.clear(); buf_meta.clear()

    t0 = time.time(); done = 0
    for r in targets:
        try:
            w, _ = load_wav_mono(r["audio_path"], sr)
        except Exception:
            continue
        w = w.cuda()                       # augment on GPU (CPU resample was the bottleneck)
        if w.numel() > maxlen:
            w = w[:maxlen]
        for kind in KINDS:
            aw = augment(w, kind)
            if aw.numel() > maxlen:
                aw = aw[:maxlen]
            if aw.numel() < int(0.25 * sr):
                continue
            buf_w.append(aw); buf_meta.append((r["label_id"], r["speaker_id"], kind))
            if len(buf_w) >= args.batch_size:
                flush()
        done += 1
        if done % 500 == 0:
            print(f"  {done:,}/{len(targets):,} src done, {len(X):,} aug feats "
                  f"({done/(time.time()-t0):.1f} src/s)", flush=True)
    flush()
    X = np.array(X); Y = np.array(Y, np.int64); SP = np.array(SP, dtype=object); AUG = np.array(AUG, dtype=object)
    np.savez(args.out, X=X, y=Y, speaker=SP, aug=AUG)
    print(f"[done] {len(X):,} augmented features -> {args.out}", flush=True)
    print("per-aug:", dict(Counter(AUG.tolist())))


if __name__ == "__main__":
    main()

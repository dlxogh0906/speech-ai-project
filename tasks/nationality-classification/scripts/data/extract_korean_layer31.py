"""Sample native-Korean speakers and extract layer-31/35/39 XLS-R features.

SAME pipeline as the foreign features (extract_layer_features.py): frozen XLS-R 2B,
mean-pool over time at the chosen layers, max_sec=12 truncation, clamp[-50,50].
So the Korean feature is directly comparable to $SPEECH_AI_DATA/taeho_layers_full.

  python extract_korean_layer31.py --n_speakers 300 --cap 500 \
      --out $SPEECH_AI_DATA/taeho_layers_korean/korean.npz
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, FAIRSEQ_ROOT, MODEL_ROOT


import argparse
import csv
import sqlite3
import time
from pathlib import Path

import numpy as np

from country_classifier import baseline_common as bc

DB = f"{DATA_ROOT}/taeho/korean_native_index/korean_native_index.sqlite"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
    ap.add_argument("--n_speakers", type=int, default=300)
    ap.add_argument("--cap", type=int, default=500, help="max utts per speaker")
    ap.add_argument("--layers", type=int, nargs="+", default=[31, 35, 39])
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from country_classifier.dataset import build_dataloader
    from country_classifier.model import build_model

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # --- sample speakers (>= cap utts), then up to cap utts each ---
    # single bulk read + group in Python (per-speaker SQL was full-scanning -> too slow)
    from collections import defaultdict
    con = sqlite3.connect(args.db)
    print("[sample] reading index (one pass) ...", flush=True)
    by_spk = defaultdict(list)
    for rid, p in con.execute("select recorder_id, resolved_audio_path from utterances "
                              "where eligible_for_sampling=1"):
        if p:
            by_spk[rid].append(p)
    elig = [s for s, ps in by_spk.items() if len(ps) >= args.cap]
    print(f"[sample] {len(by_spk)} speakers total, {len(elig)} with >= {args.cap} utts", flush=True)
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(elig, args.n_speakers, replace=False).tolist()
    manifest = []
    for spk in picked:
        for p in sorted(by_spk[spk])[:args.cap]:
            manifest.append({"audio_path": p, "json_path": "", "speaker_id": spk,
                             "label_name": "한국어", "label_id": 0, "duration_sec": ""})
    csv_path = out.parent / "korean_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS); w.writeheader(); w.writerows(manifest)
    print(f"[sample] {args.n_speakers} speakers, {len(manifest):,} utts (cap {args.cap}) "
          f"-> {csv_path}", flush=True)

    # --- load model, extract ---
    print("[model] loading XLS-R 2B ...", flush=True)
    model = build_model(backend="fairseq_xlsr", num_classes=61, checkpoint_path=args.xlsr_ckpt,
                        fairseq_root=args.fairseq_root, freeze_encoder=True).cuda().eval()
    L = args.layers

    def pool(x_tbc, pad):                         # (T,B,C) -> (B,C) masked mean
        x = x_tbc.transpose(0, 1)
        if pad is None:
            return x.mean(1)
        valid = (~pad).to(x.dtype).unsqueeze(-1)
        return (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    loader = build_dataloader(csv_path, 16000, args.batch_size, shuffle=False, seed=0,
                              max_duration_sec=args.max_sec, num_workers=args.num_workers)
    feats, sp = [], []
    n, n_skip, t0 = 0, 0, time.time()
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if b is None:
                continue
            n_skip += len(b["skipped"])
            wav = b["waveforms"].cuda(); pm = b["padding_mask"].cuda()
            with torch.autocast("cuda"):
                res = model.encoder.extract_features(source=wav, padding_mask=pm, mask=False)
            fpad = res.get("padding_mask")
            pooled = torch.stack([pool(res["layer_results"][l][0], fpad) for l in L], dim=1)
            pooled = torch.nan_to_num(pooled.float(), posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
            feats.append(pooled.cpu().numpy().astype(np.float16))
            sp.extend(b["speaker_ids"]); n += wav.size(0)
            if bi % 200 == 0:
                print(f"  {n:,} done ({n/(time.time()-t0):.0f} utt/s) skipped={n_skip}", flush=True)
    X = np.concatenate(feats, 0)
    np.savez(args.out, X=X, speaker=np.array(sp, dtype=object),
             label=np.array(["한국어"] * len(sp), dtype=object), layers=np.array(L))
    print(f"[done] X={X.shape} ({len(set(sp))} speakers) skipped={n_skip} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

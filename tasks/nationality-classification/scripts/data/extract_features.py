"""Cache frozen XLS-R 2B mean-pooled features for each split's CSV manifest.

Because the encoder is frozen, its features are fixed, so extracting them once and
then fitting a linear head is mathematically identical to end-to-end frozen training
-- just far cheaper. Output per split: <out_dir>/<split>.npz with
  X        float16 [N, D]   mean-pooled features
  y        int64   [N]      label id
  speaker  <U..    [N]      speaker id
(N excludes utterances skipped by the loader as unreadable/too short.)

    python scripts/extract_features.py \
        --manifest_dir $NATIONALITY_DATA/aihub_nationality_fine_811 \
        --out_dir $NATIONALITY_RUNS/features_fine_811 \
        --splits test valid train --batch_size 24 --max_sec 12 --num_workers 8
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, MODEL_ROOT


import argparse
import time
from pathlib import Path

import numpy as np

from country_classifier import baseline_common as bc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "valid", "train"])
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--min_sec", type=float, default=0.25)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=None, help="cap utts per split (debug)")
    ap.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="print progress every N dataloader batches",
    )
    ap.add_argument(
        "--resume_chunks",
        action="store_true",
        help=(
            "write per-batch chunk files while extracting and resume unfinished splits "
            "from those chunks; final output is still <split>.npz"
        ),
    )
    args = ap.parse_args()

    import json

    import torch
    from country_classifier.dataset import build_dataloader
    from country_classifier.model import build_model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, label2id, id2label = bc.build_label_maps(args.manifest_dir)
    num_classes = len(labels)
    (out_dir / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    (out_dir / "id2label.json").write_text(json.dumps(id2label, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    print(f"loading XLS-R 2B ({num_classes} classes) ...")
    t0 = time.time()
    model = build_model(backend="fairseq_xlsr", num_classes=num_classes,
                        checkpoint_path=args.xlsr_ckpt, fairseq_root=args.fairseq_root,
                        freeze_encoder=True).cuda().eval()
    print(f"  loaded in {time.time() - t0:.0f}s")

    for split in args.splits:
        csv_path = Path(args.manifest_dir) / f"{split}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue
        out_npz = out_dir / f"{split}.npz"
        if out_npz.exists():
            print(f"[skip] {out_npz} already exists")
            continue
        chunk_dir = out_dir / f"_{split}_chunks"
        done_batches = set()
        if args.resume_chunks and chunk_dir.exists():
            for p in chunk_dir.glob("batch_*.npz"):
                try:
                    done_batches.add(int(p.stem.split("_")[1]))
                except Exception:
                    continue
            if done_batches:
                print(f"[resume] {split}: found {len(done_batches):,} saved chunks")

        if args.limit:  # debug: truncate via a temp csv
            import csv as _csv
            rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))[: args.limit]
            tmp = out_dir / f"_{split}_limited.csv"
            with tmp.open("w", encoding="utf-8", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=bc.CSV_FIELDS)
                w.writeheader(); w.writerows(rows)
            csv_path = tmp

        loader = build_dataloader(csv_path, args.sample_rate, args.batch_size, shuffle=False,
                                  seed=0, min_duration_sec=args.min_sec,
                                  max_duration_sec=args.max_sec, num_workers=args.num_workers)
        feats, ys, spks = [], [], []
        n, n_skip, t1 = 0, 0, time.time()
        if args.resume_chunks:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(chunk_dir.glob("batch_*.npz")):
                d = np.load(p, allow_pickle=True)
                n += int(d["y"].shape[0])
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if args.resume_chunks and bi in done_batches:
                    if args.log_every > 0 and bi % args.log_every == 0:
                        rate = n / max(1e-6, time.time() - t1)
                        print(f"  [{split}] {n:,} resumed ({rate:.0f} utt/s) chunks={len(done_batches):,}", flush=True)
                    continue
                if batch is None:
                    continue
                n_skip += len(batch["skipped"])
                wav = batch["waveforms"].cuda(non_blocking=True)
                mask = batch["padding_mask"].cuda(non_blocking=True)
                with torch.autocast("cuda"):
                    f, fmask = model._extract_features(wav, mask)
                    pooled = model._pool(f, fmask)
                feat_np = pooled.float().cpu().numpy().astype(np.float16)
                y_np = batch["labels"].numpy().astype(np.int64)
                spk_np = np.array(batch["speaker_ids"], dtype=object)
                if args.resume_chunks:
                    np.savez(chunk_dir / f"batch_{bi:08d}.npz", X=feat_np, y=y_np, speaker=spk_np)
                else:
                    feats.append(feat_np)
                    ys.extend(y_np.tolist())
                    spks.extend(batch["speaker_ids"])
                n += wav.size(0)
                if args.log_every > 0 and bi % args.log_every == 0:
                    rate = n / max(1e-6, time.time() - t1)
                    print(f"  [{split}] {n:,} done ({rate:.0f} utt/s) skipped={n_skip}", flush=True)
        if args.resume_chunks:
            chunk_paths = sorted(chunk_dir.glob("batch_*.npz"))
            feats, ys, spks = [], [], []
            for p in chunk_paths:
                d = np.load(p, allow_pickle=True)
                feats.append(d["X"])
                ys.append(d["y"].astype(np.int64))
                spks.append(d["speaker"].astype(object))
            X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), np.float16)
            y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0,), np.int64)
            spk_all = np.concatenate(spks, axis=0) if spks else np.array([], dtype=object)
        else:
            X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), np.float16)
            y_all = np.array(ys, np.int64)
            spk_all = np.array(spks, dtype=object)
        np.savez(out_npz, X=X, y=y_all, speaker=spk_all)
        print(f"[done] {split}: X={X.shape} skipped={n_skip} "
              f"-> {out_npz} ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()

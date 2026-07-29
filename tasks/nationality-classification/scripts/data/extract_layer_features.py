"""Extract mean-pooled XLS-R features at SELECTED transformer layers (not just the top).

Research (Tier 1-1): accent info concentrates in mid-upper layers while speaker identity
dominates early layers. The frozen baseline used only the top layer. Here we cache
per-layer pooled features so we can (a) probe each layer and (b) learn a weighted sum,
to see if a better layer choice beats the top-layer macro top-5 (=0.50).

Output per split: <out>/<split>.npz with X [N, L, 1920] float16, y, speaker, and `layers`.
Train is capped per class (layer sweep needs less data); valid/test are full.
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
from shared.paths import DATA_ROOT, FAIRSEQ_ROOT, MODEL_ROOT, NAT_DATA


import argparse
import csv
from pathlib import Path

import numpy as np

from country_classifier import baseline_common as bc


def cap_csv(src, dst, cap, seed=42):
    rng = np.random.default_rng(seed)
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["label_id"], []).append(r)
    out = []
    for _, rs in by.items():
        if cap and len(rs) > cap:
            rs = [rs[i] for i in rng.choice(len(rs), cap, replace=False)]
        out.extend(rs)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS); w.writeheader(); w.writerows(out)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default=f"{NAT_DATA}/aihub_nationality_fine_811")
    ap.add_argument("--out_dir", default=f"{DATA_ROOT}/taeho_layers")
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 15, 23, 31, 35, 39, 43, 47])
    ap.add_argument("--train_cap", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["test", "valid", "train"],
                    help="which splits to (re)extract; e.g. --splits train")
    ap.add_argument("--shard_size", type=int, default=0,
                    help="for the train split: extract in resumable shards of this many rows "
                         "(saves train_partNNN.npz incrementally, then concats). 0=off")
    args = ap.parse_args()

    import torch
    from country_classifier.dataset import build_dataloader
    from country_classifier.model import build_model

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layers.json").write_text(str(args.layers))
    print(f"loading XLS-R checkpoint: {args.xlsr_ckpt}", flush=True)
    model = build_model(backend="fairseq_xlsr", num_classes=61, checkpoint_path=args.xlsr_ckpt,
                        fairseq_root=args.fairseq_root, freeze_encoder=True).cuda().eval()
    L = args.layers

    def pool(x_tbc, pad):  # x: (T,B,C) -> (B,C) masked mean
        x = x_tbc.transpose(0, 1)                     # (B,T,C)
        if pad is None:
            return x.mean(1)
        valid = (~pad).to(x.dtype).unsqueeze(-1)
        return (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    import time

    def extract_one(csv_path, tag):
        """Run the frozen encoder over one csv; return (X[N,L,1920] f16, y, speaker)."""
        loader = build_dataloader(csv_path, 16000, args.batch_size, shuffle=False, seed=0,
                                  max_duration_sec=args.max_sec, num_workers=args.num_workers)
        feats, ys, sp = [], [], []
        done = 0; t0 = time.time()
        with torch.no_grad():
            for bi, b in enumerate(loader):
                if b is None:
                    continue
                wav = b["waveforms"].cuda(); pm = b["padding_mask"].cuda()
                with torch.autocast("cuda"):
                    res = model.encoder.extract_features(source=wav, padding_mask=pm, mask=False)
                fpad = res.get("padding_mask")
                lr = res["layer_results"]
                pooled = torch.stack([pool(lr[l][0], fpad) for l in L], dim=1)  # (B,L,C)
                pooled = torch.nan_to_num(pooled.float(), posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
                feats.append(pooled.cpu().numpy().astype(np.float16))
                ys.extend(b["labels"].tolist()); sp.extend(b["speaker_ids"])
                done += wav.size(0)
                if bi % 200 == 0:
                    print(f"  [{tag}] {done:,} ({done/(time.time()-t0):.0f} utt/s)", flush=True)
        X = np.concatenate(feats, 0) if feats else np.zeros((0, len(L), 1920), np.float16)
        return X, np.array(ys, np.int64), np.array(sp, dtype=object)

    for split in args.splits:
        src = Path(args.manifest_dir) / f"{split}.csv"
        if not src.exists():
            continue
        tmp = out_dir / f"_{split}.csv"
        cap = args.train_cap if split == "train" else 0
        cap_csv(src, tmp, cap)

        if args.shard_size > 0 and split == "train":
            # resumable shards: split rows, extract each (skip finished), then concat.
            rows = list(csv.DictReader(open(tmp, encoding="utf-8")))
            n_sh = (len(rows) + args.shard_size - 1) // args.shard_size
            print(f"[train] {len(rows):,} rows -> {n_sh} shards of {args.shard_size:,}", flush=True)
            for k in range(n_sh):
                part = out_dir / f"train_part{k:03d}.npz"
                if part.exists():
                    print(f"  [shard {k:03d}] exists -> skip (resume)", flush=True)
                    continue
                chunk = rows[k * args.shard_size:(k + 1) * args.shard_size]
                cpath = out_dir / f"_train_shard{k:03d}.csv"
                with open(cpath, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS); w.writeheader(); w.writerows(chunk)
                X, ys, sp = extract_one(cpath, f"shard {k:03d}")
                np.savez(part, X=X, y=ys, speaker=sp, layers=np.array(L))
                cpath.unlink(missing_ok=True)
                print(f"  [shard {k:03d}] saved X={X.shape} -> {part.name}", flush=True)
            # concat shards -> train.npz
            Xs, Ys, Sp = [], [], []
            for k in range(n_sh):
                d = np.load(out_dir / f"train_part{k:03d}.npz", allow_pickle=True)
                Xs.append(d["X"]); Ys.append(d["y"]); Sp.append(d["speaker"])
            X = np.concatenate(Xs); y = np.concatenate(Ys); sp = np.concatenate(Sp)
            np.savez(out_dir / "train.npz", X=X, y=y, speaker=sp, layers=np.array(L))
            print(f"[done] train: X={X.shape} -> {out_dir/'train.npz'} (from {n_sh} shards)", flush=True)
            tmp.unlink(missing_ok=True)
        else:
            X, ys, sp = extract_one(tmp, split)
            np.savez(out_dir / f"{split}.npz", X=X, y=ys, speaker=sp, layers=np.array(L))
            print(f"[done] {split}: X={X.shape} -> {out_dir/f'{split}.npz'}", flush=True)
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

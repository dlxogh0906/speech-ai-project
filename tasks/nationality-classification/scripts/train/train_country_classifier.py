"""Train the XLS-R country-classification baseline (mean-pool + linear head).

Reuses $HOME/country_classifier (dataset / model / metrics / utils). The fairseq
manifests under $NATIONALITY_DATA are materialised into CSV manifests on the fly.

See README_BASELINE.md for the full run sequence. Smoke run example:

    python scripts/train_country_classifier.py \
        --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_quarter_valid_test \
        --xlsr_ckpt $SPEECH_AI_MODELS/xlsr2_2B_1000k.pt \
        --output_dir $NATIONALITY_RUNS/xlsr_country_baseline_smoke \
        --batch_size 1 --grad_accum_steps 8 --epochs 1 --lr 1e-4 --max_sec 10 \
        --freeze_encoder true --amp --num_workers 2 --limit_train 200 --limit_valid 100
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
from shared.paths import FAIRSEQ_ROOT, MODEL_ROOT


import argparse
import csv
import time
from pathlib import Path

from country_classifier import baseline_common as bc


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--backend", default="fairseq_xlsr", choices=["fairseq_xlsr", "tiny"])
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--audio_root", default=bc.DEFAULT_AUDIO_ROOT)
    ap.add_argument("--no_audio_index", dest="audio_index", action="store_false",
                    help="resolve audio by literal path instead of a basename index")
    ap.set_defaults(audio_index=True)
    ap.add_argument("--alias_config", default=None)
    ap.add_argument("--csv_manifests", action="store_true",
                    help="read {train,valid}.csv directly from --manifest_dir "
                         "(e.g. the speaker_811 split); audio paths already resolved")
    # training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--min_sec", type=float, default=0.25)
    ap.add_argument("--freeze_encoder", type=bc.str2bool, default=True)
    ap.add_argument("--classifier_hidden_dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_best_by", default="top1", choices=["top1", "loss"])
    # smoke knobs
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_valid", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="train on synthetic wavs (when real audio is not on this machine)")
    ap.add_argument("--log_every", type=int, default=50)
    return ap.parse_args()


def _limited_csv(src, limit, dest):
    """Copy header + first `limit` rows of a CSV to dest; return path to use."""
    if not limit:
        return src
    src = Path(src)
    with src.open("r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    dest = Path(dest)
    dest.write_text("".join(lines[:1] + lines[1:1 + limit]), encoding="utf-8")
    return dest


def build_csvs(args, label2id, labels, out_dir, audio_index=None):
    """Return (train_csv, valid_csv). Synthetic, prebuilt CSVs, or materialised."""
    if args.csv_manifests:
        md = Path(args.manifest_dir)
        tr = _limited_csv(md / "train.csv", args.limit_train, out_dir / "train.csv")
        va = _limited_csv(md / "valid.csv", args.limit_valid, out_dir / "valid.csv")
        print(f"[data] using prebuilt CSV manifests from {md}")
        return tr, va
    if args.synthetic:
        tr = bc.generate_synthetic_csv(out_dir / "synth_train", labels, label2id,
                                       n_per_split=args.limit_train or 200,
                                       sample_rate=args.sample_rate, seconds=min(args.max_sec, 4.0),
                                       seed=args.seed)
        va = bc.generate_synthetic_csv(out_dir / "synth_valid", labels, label2id,
                                       n_per_split=args.limit_valid or 100,
                                       sample_rate=args.sample_rate, seconds=min(args.max_sec, 4.0),
                                       seed=args.seed + 1)
        return tr, va
    tr = out_dir / "train.csv"
    va = out_dir / "valid.csv"
    n_tr, _ = bc.prepare_split_csv(args.manifest_dir, "train", tr, label2id,
                                   limit=args.limit_train, audio_root=args.audio_root,
                                   sample_rate=args.sample_rate, audio_index=audio_index)
    n_va, _ = bc.prepare_split_csv(args.manifest_dir, "valid", va, label2id,
                                   limit=args.limit_valid, audio_root=args.audio_root,
                                   sample_rate=args.sample_rate, audio_index=audio_index)
    print(f"[data] materialised train={n_tr} valid={n_va} CSV rows")
    return tr, va


def run_validation(model, loader, num_classes, device, amp):
    import torch
    from country_classifier.metrics import TopKAccuracy
    from sklearn.metrics import f1_score

    model.eval()
    meter = TopKAccuracy(num_classes)
    all_pred, all_true = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            wav = batch["waveforms"].to(device)
            mask = batch["padding_mask"].to(device)
            y = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
                logits = model(wav, mask)
                loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
            n_batches += 1
            meter.update(logits, y)
            all_pred.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(y.cpu().tolist())
    m = meter.compute()
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0) if all_true else 0.0
    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_top1": m["top1_acc"],
        "val_top5": m["top5_acc"],
        "val_macro_f1": macro_f1,
        "val_samples": m["num_samples"],
    }


def main():
    args = parse_args()
    import torch
    from country_classifier.dataset import build_dataloader
    from country_classifier.model import build_model
    from country_classifier.utils import set_seed, write_json

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    labels, label2id, id2label = bc.build_label_maps(args.manifest_dir)
    num_classes = len(labels)
    write_json(out_dir / "label2id.json", label2id)
    write_json(out_dir / "id2label.json", id2label)
    write_json(out_dir / "config.json", vars(args))
    print(f"[init] device={device} backend={args.backend} num_classes={num_classes} labels={labels}")

    audio_index = None
    if not args.synthetic and not args.csv_manifests and args.audio_index:
        print("[data] building audio basename index ...")
        audio_index = bc.build_audio_index(args.audio_root)
        print(f"[data] indexed {len(audio_index):,} wavs under {args.audio_root}")
    train_csv, valid_csv = build_csvs(args, label2id, labels, out_dir, audio_index=audio_index)

    train_loader = build_dataloader(train_csv, args.sample_rate, args.batch_size, shuffle=True,
                                    seed=args.seed, min_duration_sec=args.min_sec,
                                    max_duration_sec=args.max_sec, num_workers=args.num_workers)
    valid_loader = build_dataloader(valid_csv, args.sample_rate, args.batch_size, shuffle=False,
                                    seed=args.seed, min_duration_sec=args.min_sec,
                                    max_duration_sec=args.max_sec, num_workers=args.num_workers)

    if args.backend == "tiny":
        model = build_model(backend="tiny", num_classes=num_classes, dropout=args.dropout)
    else:
        model = build_model(backend="fairseq_xlsr", num_classes=num_classes,
                            checkpoint_path=args.xlsr_ckpt, fairseq_root=args.fairseq_root,
                            freeze_encoder=args.freeze_encoder, dropout=args.dropout,
                            classifier_hidden_dim=args.classifier_hidden_dim)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    print(f"[init] trainable params = {n_train_params:,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_top1", "val_top5", "val_macro_f1", "sec"])

    best_metric = -1.0 if args.save_best_by == "top1" else float("inf")

    def oom_guard(exc):
        if "out of memory" in str(exc).lower():
            print("\n[OOM] CUDA out of memory. Try: lower --batch_size (already 1?), "
                  "reduce --max_sec, increase --grad_accum_steps, or keep --freeze_encoder true.")
            if device == "cuda":
                torch.cuda.empty_cache()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen, step = 0.0, 0, 0
        for it, batch in enumerate(train_loader):
            if batch is None:
                continue
            try:
                wav = batch["waveforms"].to(device)
                mask = batch["padding_mask"].to(device)
                y = batch["labels"].to(device)
                with torch.autocast(device_type="cuda", enabled=args.amp and device == "cuda"):
                    logits = model(wav, mask)
                    loss = torch.nn.functional.cross_entropy(logits, y) / args.grad_accum_steps
                scaler.scale(loss).backward()
                step += 1
                if step % args.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                oom_guard(exc)
                raise
            running += loss.item() * args.grad_accum_steps * y.size(0)
            seen += y.size(0)
            if args.log_every and step % args.log_every == 0:
                print(f"  e{epoch} it{step} loss={running / max(1, seen):.4f} seen={seen}")
        # flush a trailing partial accumulation window
        if step % args.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running / max(1, seen)
        val = run_validation(model, valid_loader, num_classes, device, args.amp)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val['val_loss']:.4f} top1={val['val_top1']:.4f} "
              f"top5={val['val_top5']:.4f} macroF1={val['val_macro_f1']:.4f} ({dt:.0f}s)")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val['val_loss']:.6f}",
                                    f"{val['val_top1']:.6f}", f"{val['val_top5']:.6f}",
                                    f"{val['val_macro_f1']:.6f}", f"{dt:.1f}"])

        cur = val["val_top1"] if args.save_best_by == "top1" else val["val_loss"]
        improved = cur > best_metric if args.save_best_by == "top1" else cur < best_metric
        ckpt_meta = {"epoch": epoch, "val": val, "labels": labels}
        bc.save_checkpoint(out_dir / "last.pt", model, label2id, id2label, vars(args), ckpt_meta)
        if improved:
            best_metric = cur
            bc.save_checkpoint(out_dir / "best.pt", model, label2id, id2label, vars(args), ckpt_meta)
            print(f"  -> new best ({args.save_best_by}={cur:.4f}) saved to best.pt")

    print(f"\n[done] checkpoints + logs in {out_dir}")


if __name__ == "__main__":
    main()

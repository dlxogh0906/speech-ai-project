"""Evaluate a trained country-classification checkpoint on a split.

Writes metrics.json (top1/top5/macro-F1/per-class acc), predictions.csv, and
confusion_matrix.csv.

    python scripts/eval_country_classifier.py \
        --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_full_valid_test_no_train_speaker_test \
        --split test --ckpt $NATIONALITY_RUNS/xlsr_country_baseline_full/best.pt \
        --output_dir $NATIONALITY_RUNS/xlsr_country_baseline_full_eval \
        --batch_size 1 --max_sec 15
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
import csv
import json
from pathlib import Path

from country_classifier import baseline_common as bc


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--audio_root", default=bc.DEFAULT_AUDIO_ROOT)
    ap.add_argument("--no_audio_index", dest="audio_index", action="store_false")
    ap.set_defaults(audio_index=True)
    ap.add_argument("--csv_manifests", action="store_true",
                    help="read {split}.csv directly from --manifest_dir")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--min_sec", type=float, default=0.25)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_samples", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from sklearn.metrics import confusion_matrix, f1_score

    from country_classifier.dataset import build_dataloader

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model, payload, labels = bc.restore_model(
        args.ckpt, fairseq_root=args.fairseq_root, xlsr_ckpt=args.xlsr_ckpt, device=device)
    label2id = payload["label2id"]
    num_classes = len(labels)
    print(f"[eval] device={device} num_classes={num_classes} labels={labels}")

    if args.synthetic:
        csv_path = bc.generate_synthetic_csv(out_dir / "synth_eval", labels, label2id,
                                             n_per_split=args.limit_samples or 100,
                                             sample_rate=args.sample_rate,
                                             seconds=min(args.max_sec, 4.0), seed=args.seed)
    elif args.csv_manifests:
        csv_path = Path(args.manifest_dir) / f"{args.split}.csv"
        print(f"[eval] using prebuilt CSV manifest {csv_path}")
    else:
        audio_index = None
        if args.audio_index:
            print("[eval] building audio basename index ...")
            audio_index = bc.build_audio_index(args.audio_root)
            print(f"[eval] indexed {len(audio_index):,} wavs")
        csv_path = out_dir / f"{args.split}.csv"
        n, _ = bc.prepare_split_csv(args.manifest_dir, args.split, csv_path, label2id,
                                    limit=args.limit_samples, audio_root=args.audio_root,
                                    sample_rate=args.sample_rate, audio_index=audio_index)
        print(f"[eval] materialised {n} CSV rows for split={args.split}")

    loader = build_dataloader(csv_path, args.sample_rate, args.batch_size, shuffle=False,
                              seed=args.seed, min_duration_sec=args.min_sec,
                              max_duration_sec=args.max_sec, num_workers=args.num_workers)

    k5 = min(5, num_classes)
    rows = []
    y_true, y_pred = [], []
    top5_hits = 0
    n_skipped = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            n_skipped += len(batch["skipped"])
            wav = batch["waveforms"].to(device)
            mask = batch["padding_mask"].to(device)
            logits = model(wav, mask)
            probs = torch.softmax(logits, dim=-1)
            top5 = probs.topk(k5, dim=-1)
            preds = logits.argmax(-1)
            for i in range(wav.size(0)):
                true_id = batch["labels"][i].item()
                pred_id = preds[i].item()
                y_true.append(true_id)
                y_pred.append(pred_id)
                t5_ids = top5.indices[i].cpu().tolist()
                t5_scores = top5.values[i].cpu().tolist()
                if true_id in t5_ids:
                    top5_hits += 1
                rows.append({
                    "utt_id": Path(batch["audio_paths"][i]).stem,
                    "audio_path": batch["audio_paths"][i],
                    "speaker_id": batch["speaker_ids"][i],
                    "true_label": labels[true_id],
                    "pred_label": labels[pred_id],
                    "top5_labels": "|".join(labels[j] for j in t5_ids),
                    "top5_scores": "|".join(f"{s:.4f}" for s in t5_scores),
                })

    n = len(y_true)
    if n == 0:
        raise SystemExit("[eval] no samples evaluated (all skipped / missing audio). "
                         "Extract audio (scripts/extract_audio.py) or pass --synthetic.")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_acc = {}
    for i, lab in enumerate(labels):
        denom = cm[i].sum()
        per_class_acc[lab] = float(cm[i, i] / denom) if denom else 0.0
    top1 = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n
    metrics = {
        "split": args.split,
        "num_samples": n,
        "num_skipped": n_skipped,
        "top1_acc": top1,
        "top5_acc": top5_hits / n,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_acc": per_class_acc,
        "labels": labels,
        "checkpoint": args.ckpt,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "audio_path", "speaker_id", "true_label",
                                          "pred_label", "top5_labels", "top5_scores"])
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + np.asarray(cm[i]).tolist())

    print(f"\n[eval] n={n} top1={metrics['top1_acc']:.4f} top5={metrics['top5_acc']:.4f} "
          f"macroF1={metrics['macro_f1']:.4f}")
    print(f"[eval] per-class acc: {per_class_acc}")
    print(f"[eval] wrote metrics.json / predictions.csv / confusion_matrix.csv -> {out_dir}")


if __name__ == "__main__":
    main()

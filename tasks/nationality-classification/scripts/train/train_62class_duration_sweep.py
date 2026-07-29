#!/usr/bin/env python3
"""Train 62-class linear probes for 8s/12s/16s cached XLS-R features.

The available duration-sweep foreign features were extracted for the 31-class
policy, so their `y` is grouped.  Speaker IDs are still present, and each
foreign speaker has a unique original 61-class mother-tongue label.  This script
restores the original 61-class target via speaker -> label mapping, then adds
native Korean as class 61.
"""

from __future__ import annotations

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS



import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_FOREIGN_FEATURES = {
    "8s": Path(f"{DATA_ROOT}/taeho/duration_sweep_policy31/sec08/features"),
    "12s": Path(f"{NAT_OUTPUTS}/duration_sweep_policy31/sec12/features"),
    "16s": Path(f"{DATA_ROOT}/taeho/duration_sweep_policy31/sec16/features"),
}
DEFAULT_KOREAN_FEATURES = {
    "8s": Path(f"{DATA_ROOT}/taeho/korean_native_features/sec08"),
    "12s": Path(f"{DATA_ROOT}/taeho/korean_native_features/sec12"),
    "16s": Path(f"{DATA_ROOT}/taeho/korean_native_features/sec16"),
}
LABEL_SOURCE_DIR = Path(f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts")
DEFAULT_OUT_DIR = Path(f"{DATA_ROOT}/taeho/probe_62class_duration_sweep_linear")


def build_speaker_to_label_id(label_source_dir: Path):
    label2id = json.loads((label_source_dir / "label2id.json").read_text(encoding="utf-8"))
    id2label = json.loads((label_source_dir / "id2label.json").read_text(encoding="utf-8"))
    assignment = json.loads(
        (label_source_dir / "label_speaker_assignment.json").read_text(encoding="utf-8")
    )
    speaker_to_label_id = {}
    conflicts = []
    for label_key, split_to_speakers in assignment.items():
        if label_key in label2id:
            label_id = int(label2id[label_key])
        else:
            label_id = int(label_key)
        for speakers in split_to_speakers.values():
            for speaker in speakers:
                previous = speaker_to_label_id.get(speaker)
                if previous is not None and previous != label_id:
                    conflicts.append((speaker, previous, label_id))
                speaker_to_label_id[speaker] = label_id
    if conflicts:
        raise RuntimeError(f"speaker label conflicts found: {conflicts[:5]}")
    labels = [id2label[str(i)] for i in range(61)] + ["한국어_모국어"]
    return speaker_to_label_id, labels


def load_foreign_split(directory: Path, split: str, speaker_to_label_id: dict[str, int]):
    data = np.load(directory / f"{split}.npz", allow_pickle=True)
    x = data["X"].astype(np.float32)
    speakers = data["speaker"].astype(str)
    missing = sorted(set(speakers) - set(speaker_to_label_id))
    if missing:
        raise RuntimeError(f"{directory}/{split}: missing speaker label ids: {missing[:10]}")
    y = np.fromiter((speaker_to_label_id[s] for s in speakers), dtype=np.int64, count=len(speakers))
    return x, y, speakers


def load_foreign_all(directory: Path, speaker_to_label_id: dict[str, int]):
    xs = []
    speakers = []
    source_splits = []
    for split in ("train", "valid", "test"):
        data = np.load(directory / f"{split}.npz", allow_pickle=True)
        x = data["X"].astype(np.float32)
        spk = data["speaker"].astype(str)
        xs.append(x)
        speakers.append(spk)
        source_splits.extend([split] * len(spk))
    x = np.concatenate(xs)
    speaker = np.concatenate(speakers)
    missing = sorted(set(speaker) - set(speaker_to_label_id))
    if missing:
        raise RuntimeError(f"{directory}: missing speaker label ids: {missing[:10]}")
    y = np.fromiter((speaker_to_label_id[s] for s in speaker), dtype=np.int64, count=len(speaker))
    return x, y, speaker, np.asarray(source_splits, dtype=object)


def load_target_split_assignment(split_source_dir: Path):
    path = split_source_dir / "speaker_assignment.json"
    assignment = json.loads(path.read_text(encoding="utf-8"))
    valid = {"train", "valid", "test"}
    bad = sorted({split for split in assignment.values() if split not in valid})
    if bad:
        raise RuntimeError(f"unexpected split names in {path}: {bad}")
    return assignment


def load_korean_split(directory: Path, split: str):
    data = np.load(directory / f"{split}.npz", allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = np.full(len(x), 61, dtype=np.int64)
    speakers = data["speaker"].astype(str)
    return x, y, speakers


def topk(logits, y, k=5):
    indices = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(indices == y[:, None], axis=1)))


def predict(model, features, device, batch_size=16384):
    import torch

    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs)


def evaluate(logits, targets, labels):
    from sklearn.metrics import f1_score

    predictions = logits.argmax(1)
    top5_indices = np.argpartition(-logits, kth=4, axis=1)[:, :5]
    per_class = []
    for class_id, label in enumerate(labels):
        mask = targets == class_id
        if not np.any(mask):
            per_class.append(
                {"class_id": class_id, "class": label, "n": 0, "top1": None, "top5": None}
            )
        else:
            per_class.append(
                {
                    "class_id": class_id,
                    "class": label,
                    "n": int(mask.sum()),
                    "top1": float(np.mean(predictions[mask] == class_id)),
                    "top5": float(np.mean(np.any(top5_indices[mask] == class_id, axis=1))),
                }
            )
    return {
        "n": int(len(targets)),
        "top1": float(np.mean(predictions == targets)),
        "top5": topk(logits, targets),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "classes_present": int(len(np.unique(targets))),
        "per_class": per_class,
    }


def train_one(duration: str, foreign_dir: Path, korean_dir: Path, out_dir: Path, args):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    speaker_to_label_id, labels = build_speaker_to_label_id(args.label_source_dir)
    duration_out = out_dir / duration
    duration_out.mkdir(parents=True, exist_ok=True)

    splits = {}
    split_speaker_counts = {}
    foreign_all = None
    target_assignment = None
    if args.split_source_dir is not None:
        target_assignment = load_target_split_assignment(args.split_source_dir)
        foreign_all = load_foreign_all(foreign_dir, speaker_to_label_id)
        all_foreign_speakers = set(foreign_all[2])
        missing_from_features = sorted(set(target_assignment) - all_foreign_speakers)
        if missing_from_features:
            raise RuntimeError(
                f"{duration}: target split speakers missing from duration features: "
                f"{missing_from_features[:10]}"
            )

    for split in ("train", "valid", "test"):
        if foreign_all is None:
            foreign_x, foreign_y, foreign_speakers = load_foreign_split(
                foreign_dir, split, speaker_to_label_id
            )
        else:
            all_x, all_y, all_speakers, _source_splits = foreign_all
            mask = np.fromiter(
                (target_assignment[speaker] == split for speaker in all_speakers),
                dtype=bool,
                count=len(all_speakers),
            )
            foreign_x = all_x[mask]
            foreign_y = all_y[mask]
            foreign_speakers = all_speakers[mask]
        korean_x, korean_y, korean_speakers = load_korean_split(korean_dir, split)
        x = np.concatenate([foreign_x, korean_x])
        y = np.concatenate([foreign_y, korean_y])
        speakers = np.concatenate([foreign_speakers, korean_speakers])
        splits[split] = (x, y)
        split_speaker_counts[split] = {
            "n": int(len(y)),
            "foreign_n": int(len(foreign_y)),
            "korean_n": int(len(korean_y)),
            "classes_present": int(len(np.unique(y))),
            "speakers": int(len(np.unique(speakers))),
            "foreign_speakers": int(len(np.unique(foreign_speakers))),
            "korean_speakers": int(len(np.unique(korean_speakers))),
        }
        print(duration, split, x.shape, split_speaker_counts[split], flush=True)

    train_x, train_y = splits["train"]
    valid_x, valid_y = splits["valid"]
    test_x, test_y = splits["test"]

    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    test_x = (test_x - mean) / std

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.from_numpy(train_x).to(device)
    target_tensor = torch.from_numpy(train_y).to(device)
    model = nn.Linear(train_x.shape[1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    log_path = duration_out / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            ["epoch", "train_loss", "val_top1", "val_top5", "val_macro_f1", "sec"]
        )

    best = {"top5": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        permutation = torch.randperm(len(target_tensor), device=device)
        loss_sum = 0.0
        for start in range(0, len(target_tensor), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_tensor[indices]), target_tensor[indices])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(indices)

        valid_logits = predict(model, valid_x, device)
        valid_predictions = valid_logits.argmax(1)
        row = [
            epoch,
            loss_sum / len(train_y),
            float(np.mean(valid_predictions == valid_y)),
            topk(valid_logits, valid_y),
            float(f1_score(valid_y, valid_predictions, average="macro", zero_division=0)),
            time.time() - started,
        ]
        with log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(row)
        if row[3] > best["top5"]:
            best = {
                "top5": row[3],
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{duration} epoch={epoch} loss={row[1]:.4f} val_top1={row[2]:.4f} "
                f"val_top5={row[3]:.4f} macro_f1={row[4]:.4f}",
                flush=True,
            )

    model.load_state_dict(best["state"])
    test_logits = predict(model, test_x, device)
    overall = evaluate(test_logits, test_y, labels)
    foreign_mask = test_y != 61
    foreign = evaluate(test_logits[foreign_mask, :61], test_y[foreign_mask], labels[:61])
    result = {
        "model": "linear",
        "duration": duration,
        "classes": 62,
        "foreign_features_dir": str(foreign_dir),
        "korean_features_dir": str(korean_dir),
        "label_source_dir": str(args.label_source_dir),
        "split_source_dir": str(args.split_source_dir) if args.split_source_dir else None,
        "label_restore_rule": "foreign y restored from speaker -> original 61-class label",
        "split_rule": (
            "foreign split rebuilt from split_source_dir/speaker_assignment.json"
            if args.split_source_dir
            else "foreign split used as stored in each duration feature directory"
        ),
        "best_epoch": best["epoch"],
        "best_val_top5": best["top5"],
        "split_speaker_counts": split_speaker_counts,
        "overall": overall,
        "foreign_only_61way": foreign,
    }
    (duration_out / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (duration_out / "classwise.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_id", "class", "n", "top1", "top5"])
        writer.writeheader()
        writer.writerows(overall["per_class"])
    torch.save(
        {
            "head": best["state"],
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "labels": labels,
        },
        duration_out / "linear_best.pt",
    )
    print(duration, "overall", {k: v for k, v in overall.items() if k != "per_class"}, flush=True)
    print(duration, "foreign_only_61way", {k: v for k, v in foreign.items() if k != "per_class"}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="+", default=["8s", "12s", "16s"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label-source-dir", type=Path, default=LABEL_SOURCE_DIR)
    parser.add_argument(
        "--split-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional 61-class split directory whose speaker_assignment.json should "
            "be applied to the duration features. Use this to force 8s/12s/16s to "
            "share the target_counts speaker-disjoint split."
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for duration in args.durations:
        result = train_one(
            duration,
            DEFAULT_FOREIGN_FEATURES[duration],
            DEFAULT_KOREAN_FEATURES[duration],
            args.out_dir,
            args,
        )
        results.append(result)

    summary_rows = []
    for result in results:
        korean = next(
            item for item in result["overall"]["per_class"] if item["class"] == "한국어_모국어"
        )
        summary_rows.append(
            {
                "duration": result["duration"],
                "test_n": result["overall"]["n"],
                "classes_present": result["overall"]["classes_present"],
                "overall_top1": result["overall"]["top1"],
                "overall_top5": result["overall"]["top5"],
                "overall_macro_f1": result["overall"]["macro_f1"],
                "foreign_top1": result["foreign_only_61way"]["top1"],
                "foreign_top5": result["foreign_only_61way"]["top5"],
                "foreign_macro_f1": result["foreign_only_61way"]["macro_f1"],
                "korean_top1": korean["top1"],
                "korean_top5": korean["top5"],
                "best_epoch": result["best_epoch"],
                "best_val_top5": result["best_val_top5"],
            }
        )
    with (args.out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

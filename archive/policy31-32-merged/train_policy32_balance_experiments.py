#!/usr/bin/env python3
"""Compare Policy32 12-second baseline with two class-balancing strategies."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("$NATIONALITY_OUTPUTS/feature_policy32_othergpu_split_12s"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("$NATIONALITY_OUTPUTS/policy32_othergpu_split_12s_probe"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("$SPEECH_AI_DATA/taeho/policy32_balance_experiments_12s"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-cap", type=int, default=20_000)
    parser.add_argument("--speaker-cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(path):
    data = np.load(path, allow_pickle=True)
    result = {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
    }
    if "speaker" in data.files:
        result["speaker"] = data["speaker"].astype(str)
    return result


def balanced_cap_indices(y, speakers, class_cap, speaker_cap, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        class_speakers = speakers[class_indices]
        speaker_limited = []
        for speaker in np.unique(class_speakers):
            indices = class_indices[class_speakers == speaker]
            if len(indices) > speaker_cap:
                indices = rng.choice(indices, size=speaker_cap, replace=False)
            speaker_limited.extend(indices.tolist())
        speaker_limited = np.asarray(speaker_limited, dtype=np.int64)
        if len(speaker_limited) > class_cap:
            speaker_limited = rng.choice(
                speaker_limited,
                size=class_cap,
                replace=False,
            )
        selected.extend(speaker_limited.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def topk_accuracy(logits, y, k):
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(topk == y[:, None], axis=1)))


def predict(head, X, device, batch_size=16_384):
    import torch

    outputs = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            outputs.append(head(batch).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate(logits, y, labels):
    from sklearn.metrics import f1_score

    predicted = logits.argmax(axis=1)
    top5 = np.argpartition(-logits, kth=4, axis=1)[:, :5]
    per_class = []
    for class_id, label in enumerate(labels):
        mask = y == class_id
        if not np.any(mask):
            per_class.append(
                {
                    "label_id": class_id,
                    "class": label,
                    "n": 0,
                    "top1": None,
                    "top5": None,
                }
            )
            continue
        per_class.append(
            {
                "label_id": class_id,
                "class": label,
                "n": int(mask.sum()),
                "top1": float(np.mean(predicted[mask] == class_id)),
                "top5": float(np.mean(np.any(top5[mask] == class_id, axis=1))),
            }
        )
    return {
        "n": int(len(y)),
        "top1": float(np.mean(predicted == y)),
        "top5": topk_accuracy(logits, y, 5),
        "macro_f1": float(
            f1_score(y, predicted, average="macro", zero_division=0)
        ),
        "classes_present": int(len(np.unique(y))),
        "per_class": per_class,
    }


def train_experiment(
    name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    labels,
    args,
    class_weights=None,
):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    output_dir = args.out_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std

    train_x = torch.from_numpy(X_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    head = nn.Linear(X_train.shape[1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            ["epoch", "train_loss", "val_top1", "val_top5", "val_macro_f1", "sec"]
        )

    best = {"top5": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        head.train()
        permutation = torch.randperm(len(train_y), device=device)
        total_loss = 0.0
        for start in range(0, len(train_y), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)

        validation_logits = predict(head, X_valid, device)
        validation_pred = validation_logits.argmax(axis=1)
        validation_top1 = float(np.mean(validation_pred == y_valid))
        validation_top5 = topk_accuracy(validation_logits, y_valid, 5)
        validation_f1 = float(
            f1_score(
                y_valid,
                validation_pred,
                average="macro",
                zero_division=0,
            )
        )
        row = [
            epoch,
            total_loss / len(train_y),
            validation_top1,
            validation_top5,
            validation_f1,
            time.time() - started,
        ]
        with log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(row)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[{name} e{epoch}] loss={row[1]:.4f} "
                f"val_top1={validation_top1:.4f} "
                f"val_top5={validation_top5:.4f} "
                f"macroF1={validation_f1:.4f}",
                flush=True,
            )
        if validation_top5 > best["top5"]:
            best = {
                "top5": validation_top5,
                "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                },
            }

    head.load_state_dict(best["state"])
    test_logits = predict(head, X_test, device)
    overall = evaluate(test_logits, y_test, labels)
    foreign_mask = y_test != 31
    foreign = evaluate(
        test_logits[foreign_mask, :31],
        y_test[foreign_mask],
        labels[:31],
    )
    result = {
        "experiment": name,
        "train_samples": int(len(y_train)),
        "train_class_counts": {
            labels[class_id]: int(np.sum(y_train == class_id))
            for class_id in range(len(labels))
        },
        "class_weights": (
            {
                labels[class_id]: float(class_weights[class_id])
                for class_id in range(len(labels))
            }
            if class_weights is not None
            else None
        ),
        "best_epoch": best["epoch"],
        "best_val_top5": best["top5"],
        "overall": overall,
        "foreign_only": foreign,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "classwise.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["label_id", "class", "n", "top1", "top5"],
        )
        writer.writeheader()
        writer.writerows(overall["per_class"])
    torch.save(
        {
            "head": best["state"],
            "mu": mean,
            "sd": std,
            "labels": labels,
            "kind": "linear",
            "experiment": name,
        },
        output_dir / "linear_best.pt",
    )
    print(
        f"[{name} TEST] top1={overall['top1']:.4f} "
        f"top5={overall['top5']:.4f} macroF1={overall['macro_f1']:.4f}",
        flush=True,
    )
    del train_x, train_y, head, optimizer
    torch.cuda.empty_cache()
    return result


def load_baseline(args, labels):
    metrics = json.loads(
        (args.baseline_dir / "detailed_test_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    classwise = np.genfromtxt(
        args.baseline_dir / "foreign_only_classwise.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    per_class = []
    for row in classwise:
        per_class.append(
            {
                "label_id": int(row["label_id"]),
                "class": str(row["class"]),
                "n": int(row["n"]),
                "top1": (
                    None if np.isnan(float(row["top1"])) else float(row["top1"])
                ),
                "top5": (
                    None if np.isnan(float(row["top5"])) else float(row["top5"])
                ),
            }
        )
    korean = metrics["korean_only"]
    per_class.append(
        {
            "label_id": 31,
            "class": labels[31],
            "n": int(korean["n"]),
            "top1": float(korean["top1"]),
            "top5": float(korean["top5"]),
        }
    )
    return {
        "experiment": "baseline",
        "train_samples": 1_004_483,
        "best_epoch": 47,
        "overall": {
            **metrics["all_test"],
            "macro_f1": metrics["all_test"]["macro_f1_default"],
            "classes_present": 30,
            "per_class": per_class,
        },
        "foreign_only": {
            **metrics["foreign_only"],
            "macro_f1": metrics["foreign_only"]["macro_f1_default"],
            "classes_present": 29,
        },
    }


def write_summary(results, output_dir):
    rows = []
    for result in results:
        rows.append(
            {
                "experiment": result["experiment"],
                "train_samples": result["train_samples"],
                "overall_top1": result["overall"]["top1"],
                "overall_top5": result["overall"]["top5"],
                "overall_macro_f1": result["overall"]["macro_f1"],
                "foreign_top1": result["foreign_only"]["top1"],
                "foreign_top5": result["foreign_only"]["top5"],
                "foreign_macro_f1": result["foreign_only"]["macro_f1"],
                "best_epoch": result["best_epoch"],
            }
        )
    with (output_dir / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Policy32 12s Balance Experiment",
        "",
        "| experiment | train samples | overall top1 | overall top5 | overall macro F1 | foreign top1 | foreign top5 | foreign macro F1 | best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['train_samples']:,} | "
            f"{row['overall_top1']:.4f} | {row['overall_top5']:.4f} | "
            f"{row['overall_macro_f1']:.4f} | {row['foreign_top1']:.4f} | "
            f"{row['foreign_top5']:.4f} | {row['foreign_macro_f1']:.4f} | "
            f"{row['best_epoch']} |"
        )
    (output_dir / "RESULT_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    by_experiment = {result["experiment"]: result for result in results}
    classwise_rows = []
    baseline_classes = by_experiment["baseline"]["overall"]["per_class"]
    cap_classes = by_experiment["balanced_cap"]["overall"]["per_class"]
    loss_classes = by_experiment["balanced_loss"]["overall"]["per_class"]
    for baseline, cap, loss in zip(
        baseline_classes,
        cap_classes,
        loss_classes,
    ):
        classwise_rows.append(
            {
                "label_id": baseline["label_id"],
                "class": baseline["class"],
                "test_samples": baseline["n"],
                "baseline_top1": baseline["top1"],
                "baseline_top5": baseline["top5"],
                "balanced_cap_top1": cap["top1"],
                "balanced_cap_top5": cap["top5"],
                "balanced_loss_top1": loss["top1"],
                "balanced_loss_top5": loss["top5"],
            }
        )
    with (output_dir / "classwise_comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(classwise_rows[0]),
        )
        writer.writeheader()
        writer.writerows(classwise_rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = (
        args.features_dir.joinpath("labels.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    print("loading cached features...", flush=True)
    train = load_split(args.features_dir / "train.npz")
    valid = load_split(args.features_dir / "valid.npz")
    test = load_split(args.features_dir / "test.npz")

    cap_indices = balanced_cap_indices(
        train["y"],
        train["speaker"],
        args.class_cap,
        args.speaker_cap,
        args.seed,
    )
    cap_counts = np.bincount(
        train["y"][cap_indices],
        minlength=len(labels),
    )
    full_counts = np.bincount(train["y"], minlength=len(labels))
    loss_weights = 1.0 / np.sqrt(np.maximum(full_counts, 1))
    loss_weights /= loss_weights.mean()

    config = {
        **vars(args),
        "balanced_cap_order": "speaker cap first, then class cap",
        "balanced_loss_formula": "weight[c] = 1/sqrt(train_count[c]), normalized to mean 1",
        "full_train_samples": int(len(train["y"])),
        "cap_train_samples": int(len(cap_indices)),
        "full_class_counts": dict(zip(labels, full_counts.astype(int).tolist())),
        "cap_class_counts": dict(zip(labels, cap_counts.astype(int).tolist())),
        "loss_weights": dict(zip(labels, loss_weights.tolist())),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    baseline = load_baseline(args, labels)
    cap_result = train_experiment(
        "balanced_cap",
        train["X"][cap_indices],
        train["y"][cap_indices],
        valid["X"],
        valid["y"],
        test["X"],
        test["y"],
        labels,
        args,
    )
    loss_result = train_experiment(
        "balanced_loss",
        train["X"],
        train["y"],
        valid["X"],
        valid["y"],
        test["X"],
        test["y"],
        labels,
        args,
        class_weights=loss_weights,
    )
    results = [baseline, cap_result, loss_result]
    write_summary(results, args.out_dir)
    (args.out_dir / "all_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary: {args.out_dir / 'RESULT_SUMMARY.md'}")


if __name__ == "__main__":
    main()

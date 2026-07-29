#!/usr/bin/env python3
"""Train a 62-class linear probe: original 61 labels plus native Korean."""

from __future__ import annotations

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_OUTPUTS



import csv
import json
import time
from pathlib import Path

import numpy as np


FOREIGN_DIR = Path(f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts")
KOREAN_DIR = Path(f"{DATA_ROOT}/taeho/korean_native_features/sec12")
OUT_DIR = Path(f"{DATA_ROOT}/taeho/probe_62class_korean_12s_linear")
EPOCHS = 60
BATCH_SIZE = 4096
SEED = 42


def load_xy(directory: Path, split: str):
    data = np.load(directory / f"{split}.npz", allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int64)


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


def main():
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_by_id = json.loads((FOREIGN_DIR / "id2label.json").read_text(encoding="utf-8"))
    labels = [labels_by_id[str(i)] for i in range(61)] + ["한국어_모국어"]

    splits = {}
    for split in ("train", "valid", "test"):
        foreign_x, foreign_y = load_xy(FOREIGN_DIR, split)
        korean_x, _ = load_xy(KOREAN_DIR, split)
        korean_y = np.full(len(korean_x), 61, dtype=np.int64)
        splits[split] = (
            np.concatenate([foreign_x, korean_x]),
            np.concatenate([foreign_y, korean_y]),
        )
        print(split, splits[split][0].shape, flush=True)

    train_x, train_y = splits["train"]
    valid_x, valid_y = splits["valid"]
    test_x, test_y = splits["test"]

    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    test_x = (test_x - mean) / std

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.from_numpy(train_x).to(device)
    target_tensor = torch.from_numpy(train_y).to(device)
    model = nn.Linear(train_x.shape[1], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    log_path = OUT_DIR / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            ["epoch", "train_loss", "val_top1", "val_top5", "val_macro_f1", "sec"]
        )

    best = {"top5": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()
        permutation = torch.randperm(len(target_tensor), device=device)
        loss_sum = 0.0
        for start in range(0, len(target_tensor), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
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
                f"epoch={epoch} loss={row[1]:.4f} val_top1={row[2]:.4f} "
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
        "classes": 62,
        "best_epoch": best["epoch"],
        "best_val_top5": best["top5"],
        "overall": overall,
        "foreign_only_61way": foreign,
    }
    (OUT_DIR / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT_DIR / "classwise.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["class_id", "class", "n", "top1", "top5"]
        )
        writer.writeheader()
        writer.writerows(overall["per_class"])
    torch.save(
        {
            "head": best["state"],
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "labels": labels,
        },
        OUT_DIR / "linear_best.pt",
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("overall", "foreign_only_61way")}, ensure_ascii=False))
    print("overall", {k: v for k, v in overall.items() if k != "per_class"})
    print("foreign_only_61way", {k: v for k, v in foreign.items() if k != "per_class"})


if __name__ == "__main__":
    main()

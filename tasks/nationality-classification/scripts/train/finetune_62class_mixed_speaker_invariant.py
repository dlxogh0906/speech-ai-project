#!/usr/bin/env python3
"""Conservative 62-class fine-tuning from the strong linear baseline.

Uses 80% natural-distribution samples and 20% class/speaker-balanced samples.
A zero-initialized residual adapter preserves the baseline at epoch 0.
"""

from __future__ import annotations

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT



import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np

from train_62class_speaker_invariant import (
    BASELINE_CKPT,
    BalancedSpeakerBatchSampler,
    conditional_speaker_loss,
    cross_speaker_supcon,
    embedding_diagnostics,
    evaluate,
    load_split,
    make_labels,
    predict as _unused_predict,
    save_classwise,
    topk_accuracy,
)


OUT_DIR = Path(f"{DATA_ROOT}/taeho/probe_62class_mixed_speaker_invariant_12s")
EPOCHS = 12
STEPS_PER_EPOCH = 300
NATURAL_BATCH = 832
SUPCON_WEIGHT = 0.10
ADVERSARIAL_WEIGHT = 0.02
DISTILL_WEIGHT = 0.50
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--supcon-weight", type=float, default=SUPCON_WEIGHT)
    parser.add_argument("--adversarial-weight", type=float, default=ADVERSARIAL_WEIGHT)
    parser.add_argument("--distill-weight", type=float, default=DISTILL_WEIGHT)
    return parser.parse_args()


def predict(model, features, mean, std, device, batch_size=16384, return_embedding=False):
    import torch

    logits, embeddings = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                (features[start : start + batch_size].astype(np.float32) - mean) / std
            ).to(device)
            embedding, output = model(batch)
            logits.append(output.float().cpu().numpy())
            if return_embedding:
                embeddings.append(embedding.float().cpu().numpy())
    logits = np.concatenate(logits)
    return (logits, np.concatenate(embeddings)) if return_embedding else logits


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from sklearn.metrics import f1_score

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = make_labels()
    train, valid, test = load_split("train"), load_split("valid"), load_split("test")
    baseline = torch.load(BASELINE_CKPT, map_location="cpu")
    mean, std = baseline["mean"].astype(np.float32), baseline["std"].astype(np.float32)

    speaker_names = sorted(set(train["speaker"]))
    speaker_to_id = {speaker: index for index, speaker in enumerate(speaker_names)}
    train_speaker_ids = np.asarray(
        [speaker_to_id[speaker] for speaker in train["speaker"]], dtype=np.int64
    )
    class_speaker_columns = {}
    speaker_local_targets = np.zeros(len(speaker_names), dtype=np.int64)
    for class_id in range(62):
        names = sorted(set(train["speaker"][train["y"] == class_id]))
        columns = [speaker_to_id[name] for name in names]
        class_speaker_columns[class_id] = columns
        for local_id, global_id in enumerate(columns):
            speaker_local_targets[global_id] = local_id

    sampler = BalancedSpeakerBatchSampler(
        train["y"], train["speaker"], 2, 2, STEPS_PER_EPOCH, SEED
    )

    class ResidualModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(1920, 256),
                nn.GELU(),
                nn.Linear(256, 1920),
            )
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
            self.projector = nn.Sequential(
                nn.Linear(1920, 256),
                nn.LayerNorm(256),
            )
            self.classifier = nn.Linear(1920, 62)
            self.classifier.load_state_dict(baseline["head"])

        def forward(self, inputs):
            adapted = inputs + self.adapter(inputs)
            projection = self.projector(adapted)
            return projection, self.classifier(adapted)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = ResidualModel().to(device)
    speaker_head = nn.Linear(256, len(speaker_names)).to(device)
    frozen_baseline = nn.Linear(1920, 62).to(device)
    frozen_baseline.load_state_dict(baseline["head"])
    frozen_baseline.eval()
    for parameter in frozen_baseline.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.adapter.parameters(), "lr": 2e-4},
            {"params": model.projector.parameters(), "lr": 2e-4},
            {"params": model.classifier.parameters(), "lr": 5e-5},
            {"params": speaker_head.parameters(), "lr": 2e-4},
        ],
        weight_decay=1e-4,
    )
    local_targets = torch.from_numpy(speaker_local_targets).to(device)
    rng = np.random.default_rng(SEED)

    initial_logits = predict(model, valid["X"], mean, std, device)
    initial_predictions = initial_logits.argmax(1)
    initial_macro = float(
        f1_score(valid["y"], initial_predictions, average="macro", zero_division=0)
    )
    best = {
        "macro_f1": initial_macro,
        "epoch": 0,
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "speaker_head": {
            k: v.detach().cpu().clone() for k, v in speaker_head.state_dict().items()
        },
    }
    print(
        f"[epoch 0 baseline] val_top1={np.mean(initial_predictions == valid['y']):.4f} "
        f"val_top5={topk_accuracy(initial_logits, valid['y']):.4f} "
        f"macro_f1={initial_macro:.4f}",
        flush=True,
    )

    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            [
                "epoch", "loss", "ce", "supcon", "speaker", "distill",
                "val_top1", "val_top5", "val_macro_f1", "sec",
            ]
        )

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        speaker_head.train()
        totals = np.zeros(5)
        seen = 0
        reversal_scale = args.adversarial_weight * min(1.0, epoch / 6.0)
        for balanced_indices in sampler.batches(epoch):
            natural_indices = rng.choice(len(train["y"]), NATURAL_BATCH, replace=False)
            indices = np.concatenate([natural_indices, balanced_indices])
            x = torch.from_numpy(
                (train["X"][indices].astype(np.float32) - mean) / std
            ).to(device)
            y = torch.from_numpy(train["y"][indices]).to(device)
            speaker = torch.from_numpy(train_speaker_ids[indices]).to(device)
            balanced_start = len(natural_indices)

            optimizer.zero_grad(set_to_none=True)
            projection, logits = model(x)
            ce = functional.cross_entropy(logits, y)
            supcon = cross_speaker_supcon(
                projection[balanced_start:],
                y[balanced_start:],
                speaker[balanced_start:],
                0.1,
            )
            if args.adversarial_weight > 0:
                speaker_loss = conditional_speaker_loss(
                    projection[balanced_start:],
                    y[balanced_start:],
                    speaker[balanced_start:],
                    speaker_head,
                    class_speaker_columns,
                    local_targets,
                    reversal_scale,
                )
            else:
                speaker_loss = projection.sum() * 0.0
            with torch.no_grad():
                teacher_logits = frozen_baseline(x[:balanced_start])
            temperature = 2.0
            distill = functional.kl_div(
                functional.log_softmax(logits[:balanced_start] / temperature, dim=1),
                functional.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            loss = (
                ce
                + args.supcon_weight * supcon
                + speaker_loss
                + args.distill_weight * distill
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_n = len(indices)
            totals += np.asarray(
                [loss.item(), ce.item(), supcon.item(), speaker_loss.item(), distill.item()]
            ) * batch_n
            seen += batch_n

        valid_logits = predict(model, valid["X"], mean, std, device)
        predictions = valid_logits.argmax(1)
        val_top1 = float(np.mean(predictions == valid["y"]))
        val_top5 = topk_accuracy(valid_logits, valid["y"])
        val_macro = float(
            f1_score(valid["y"], predictions, average="macro", zero_division=0)
        )
        row = [
            epoch, *(totals / seen).tolist(), val_top1, val_top5, val_macro,
            time.time() - started,
        ]
        with log_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(row)
        if val_macro > best["macro_f1"]:
            best = {
                "macro_f1": val_macro,
                "epoch": epoch,
                "model": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "speaker_head": {
                    k: v.detach().cpu().clone()
                    for k, v in speaker_head.state_dict().items()
                },
            }
        print(
            f"[e{epoch}] loss={row[1]:.4f} val_top1={val_top1:.4f} "
            f"val_top5={val_top5:.4f} macro_f1={val_macro:.4f}",
            flush=True,
        )

    model.load_state_dict(best["model"])
    test_logits, embeddings = predict(
        model, test["X"], mean, std, device, return_embedding=True
    )
    overall = evaluate(test_logits, test["y"], labels)
    foreign_mask = test["y"] != 61
    foreign = evaluate(
        test_logits[foreign_mask, :61], test["y"][foreign_mask], labels[:61]
    )
    diagnostics = embedding_diagnostics(
        embeddings, test["y"], test["speaker"], SEED
    )
    result = {
        "experiment": {
            "name": "mixed_80natural_20balanced",
            "supcon_weight": args.supcon_weight,
            "adversarial_weight": args.adversarial_weight,
            "distill_weight": args.distill_weight,
        },
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["macro_f1"],
        "overall": overall,
        "foreign_only_61way": foreign,
        "embedding_diagnostics": diagnostics,
    }
    (args.out_dir / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_classwise(args.out_dir / "classwise.csv", overall["per_class"])
    torch.save(
        {
            "model": best["model"],
            "speaker_head": best["speaker_head"],
            "mean": mean,
            "std": std,
            "labels": labels,
        },
        args.out_dir / "best.pt",
    )
    print(
        "overall", {k: v for k, v in overall.items() if k != "per_class"},
        flush=True,
    )
    print(
        "foreign", {k: v for k, v in foreign.items() if k != "per_class"},
        diagnostics,
        flush=True,
    )


if __name__ == "__main__":
    main()

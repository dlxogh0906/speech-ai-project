#!/usr/bin/env python3
"""Speaker-balanced 62-class experiments on cached XLS-R embeddings.

Experiments:
1. balanced_ce: class -> speaker -> utterance balanced batches.
2. balanced_supcon: add cross-speaker supervised contrastive loss.
3. balanced_supcon_adv: add language-conditional adversarial speaker loss.

The adversarial speaker head only compares speakers within the same language,
so it suppresses within-language speaker identity without directly erasing
between-language information.
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
from collections import defaultdict
from pathlib import Path

import numpy as np


FOREIGN_DIR = Path(f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts")
KOREAN_DIR = Path(f"{DATA_ROOT}/taeho/korean_native_features/sec12")
DEFAULT_OUT = Path(f"{DATA_ROOT}/taeho/probe_62class_speaker_invariant_12s")
BASELINE_CKPT = Path(f"{DATA_ROOT}/taeho/probe_62class_korean_12s_linear/linear_best.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--speakers-per-class", type=int, default=2)
    parser.add_argument("--utterances-per-speaker", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--supcon-weight", type=float, default=0.15)
    parser.add_argument("--adversarial-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(split):
    foreign = np.load(FOREIGN_DIR / f"{split}.npz", allow_pickle=True)
    korean = np.load(KOREAN_DIR / f"{split}.npz", allow_pickle=True)
    return {
        "X": np.concatenate([foreign["X"], korean["X"]]),
        "y": np.concatenate(
            [
                foreign["y"].astype(np.int64),
                np.full(len(korean["X"]), 61, dtype=np.int64),
            ]
        ),
        "speaker": np.concatenate(
            [foreign["speaker"].astype(str), korean["speaker"].astype(str)]
        ),
    }


def make_labels():
    mapping = json.loads((FOREIGN_DIR / "id2label.json").read_text(encoding="utf-8"))
    return [mapping[str(index)] for index in range(61)] + ["한국어_모국어"]


class BalancedSpeakerBatchSampler:
    def __init__(
        self,
        labels,
        speakers,
        speakers_per_class,
        utterances_per_speaker,
        steps,
        seed,
    ):
        self.speakers_per_class = speakers_per_class
        self.utterances_per_speaker = utterances_per_speaker
        self.steps = steps
        self.seed = seed
        nested = defaultdict(lambda: defaultdict(list))
        for index, (class_id, speaker) in enumerate(zip(labels, speakers)):
            nested[int(class_id)][speaker].append(index)
        self.by_class = {
            class_id: {
                speaker: np.asarray(indices, dtype=np.int64)
                for speaker, indices in speaker_map.items()
            }
            for class_id, speaker_map in nested.items()
        }
        self.class_ids = sorted(self.by_class)

    def batches(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        for _ in range(self.steps):
            batch = []
            for class_id in self.class_ids:
                speaker_map = self.by_class[class_id]
                speaker_names = np.asarray(list(speaker_map))
                count = min(self.speakers_per_class, len(speaker_names))
                selected_speakers = rng.choice(speaker_names, count, replace=False)
                for speaker in selected_speakers:
                    choices = speaker_map[str(speaker)]
                    selected = rng.choice(
                        choices,
                        self.utterances_per_speaker,
                        replace=len(choices) < self.utterances_per_speaker,
                    )
                    batch.extend(selected.tolist())
            batch = np.asarray(batch, dtype=np.int64)
            rng.shuffle(batch)
            yield batch


def topk_accuracy(logits, targets, k=5):
    indices = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(indices == targets[:, None], axis=1)))


def evaluate(logits, targets, labels):
    from sklearn.metrics import f1_score

    predictions = logits.argmax(1)
    top5 = np.argpartition(-logits, kth=4, axis=1)[:, :5]
    rows = []
    for class_id, label in enumerate(labels):
        mask = targets == class_id
        rows.append(
            {
                "class_id": class_id,
                "class": label,
                "n": int(mask.sum()),
                "top1": (
                    float(np.mean(predictions[mask] == class_id))
                    if np.any(mask)
                    else None
                ),
                "top5": (
                    float(np.mean(np.any(top5[mask] == class_id, axis=1)))
                    if np.any(mask)
                    else None
                ),
            }
        )
    return {
        "n": int(len(targets)),
        "top1": float(np.mean(predictions == targets)),
        "top5": topk_accuracy(logits, targets),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "classes_present": int(len(np.unique(targets))),
        "per_class": rows,
    }


def cross_speaker_supcon(embeddings, labels, speaker_ids, temperature):
    import torch
    import torch.nn.functional as functional

    embeddings = functional.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positives = (
        (labels[:, None] == labels[None, :])
        & (speaker_ids[:, None] != speaker_ids[None, :])
        & ~identity
    )
    valid = positives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    similarity = similarity.masked_fill(identity, -1e9)
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positives).sum(dim=1) / positive_count
    return loss[valid].mean()


def gradient_reverse(inputs, scale):
    import torch

    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values):
            ctx.scale = scale
            return values.view_as(values)

        @staticmethod
        def backward(ctx, gradients):
            return -ctx.scale * gradients

    return Reverse.apply(inputs)


def conditional_speaker_loss(
    embeddings,
    class_ids,
    global_speaker_ids,
    speaker_head,
    class_speaker_columns,
    speaker_local_targets,
    reversal_scale,
):
    import torch
    import torch.nn.functional as functional

    reversed_embeddings = gradient_reverse(embeddings, reversal_scale)
    all_logits = speaker_head(reversed_embeddings)
    losses = []
    for class_id in torch.unique(class_ids).tolist():
        columns = class_speaker_columns[int(class_id)]
        if len(columns) < 2:
            continue
        row_mask = class_ids == class_id
        column_tensor = torch.as_tensor(columns, dtype=torch.long, device=embeddings.device)
        logits = all_logits[row_mask][:, column_tensor]
        targets = speaker_local_targets[global_speaker_ids[row_mask]]
        losses.append(functional.cross_entropy(logits, targets))
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def predict(model, features, mean, std, device, batch_size=16384, return_embedding=False):
    import torch

    logits = []
    embeddings = []
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
    result = np.concatenate(logits)
    if return_embedding:
        return result, np.concatenate(embeddings)
    return result


def embedding_diagnostics(embeddings, labels, speakers, seed):
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > 200:
            indices = rng.choice(indices, 200, replace=False)
        selected.extend(indices.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    reduced = PCA(
        n_components=min(50, len(selected) - 1, embeddings.shape[1]),
        random_state=seed,
    ).fit_transform(embeddings[selected])
    selected_labels = labels[selected]
    selected_speakers = speakers[selected]
    class_score = silhouette_score(
        reduced,
        selected_labels,
        sample_size=min(5000, len(selected)),
        random_state=seed,
    )
    speaker_counts = {
        speaker: int(np.sum(selected_speakers == speaker))
        for speaker in np.unique(selected_speakers)
    }
    speaker_mask = np.asarray(
        [speaker_counts[speaker] >= 2 for speaker in selected_speakers]
    )
    speaker_score = silhouette_score(
        reduced[speaker_mask],
        selected_speakers[speaker_mask],
        sample_size=min(5000, int(speaker_mask.sum())),
        random_state=seed,
    )
    return {
        "sample_n": int(len(selected)),
        "class_silhouette": float(class_score),
        "speaker_silhouette": float(speaker_score),
    }


def save_classwise(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["class_id", "class", "n", "top1", "top5"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    labels = make_labels()
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")
    baseline = torch.load(BASELINE_CKPT, map_location="cpu")
    mean = baseline["mean"].astype(np.float32)
    std = baseline["std"].astype(np.float32)

    speaker_names = sorted(set(train["speaker"]))
    speaker_to_id = {speaker: index for index, speaker in enumerate(speaker_names)}
    train_speaker_ids = np.asarray(
        [speaker_to_id[speaker] for speaker in train["speaker"]], dtype=np.int64
    )
    class_speaker_columns = {}
    speaker_local_targets = np.zeros(len(speaker_names), dtype=np.int64)
    for class_id in range(len(labels)):
        class_speakers = sorted(
            {
                speaker
                for speaker, label in zip(train["speaker"], train["y"])
                if int(label) == class_id
            }
        )
        columns = [speaker_to_id[speaker] for speaker in class_speakers]
        class_speaker_columns[class_id] = columns
        for local_id, global_id in enumerate(columns):
            speaker_local_targets[global_id] = local_id

    sampler = BalancedSpeakerBatchSampler(
        train["y"],
        train["speaker"],
        args.speakers_per_class,
        args.utterances_per_speaker,
        args.steps_per_epoch,
        args.seed,
    )

    class ProjectionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(1920, args.hidden_dim),
                nn.LayerNorm(args.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(args.hidden_dim, args.embedding_dim),
            )
            self.classifier = nn.Linear(args.embedding_dim, len(labels))

        def forward(self, inputs):
            embedding = self.projector(inputs)
            return embedding, self.classifier(embedding)

    experiments = [
        {"name": "balanced_ce", "supcon": 0.0, "adversarial": 0.0},
        {
            "name": "balanced_supcon",
            "supcon": args.supcon_weight,
            "adversarial": 0.0,
        },
        {
            "name": "balanced_supcon_conditional_adv",
            "supcon": args.supcon_weight,
            "adversarial": args.adversarial_weight,
        },
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speaker_local_targets_tensor = torch.from_numpy(speaker_local_targets).to(device)
    all_results = []

    for experiment_index, experiment in enumerate(experiments):
        output_dir = args.out_dir / experiment["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed + experiment_index)
        model = ProjectionClassifier().to(device)
        speaker_head = nn.Linear(args.embedding_dim, len(speaker_names)).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(speaker_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        best = {"macro_f1": -1.0, "epoch": 0, "model": None, "speaker_head": None}
        log_path = output_dir / "train_log.csv"
        with log_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(
                [
                    "epoch",
                    "train_loss",
                    "ce_loss",
                    "supcon_loss",
                    "speaker_loss",
                    "val_top1",
                    "val_top5",
                    "val_macro_f1",
                    "sec",
                ]
            )

        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train()
            speaker_head.train()
            sums = np.zeros(4, dtype=np.float64)
            seen = 0
            reversal_scale = experiment["adversarial"] * min(1.0, epoch / 10.0)
            for indices in sampler.batches(epoch):
                batch_x = torch.from_numpy(
                    (
                        train["X"][indices].astype(np.float32)
                        - mean
                    )
                    / std
                ).to(device)
                batch_y = torch.from_numpy(train["y"][indices]).to(device)
                batch_speaker = torch.from_numpy(train_speaker_ids[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                embedding, logits = model(batch_x)
                ce_loss = criterion(logits, batch_y)
                supcon_loss = cross_speaker_supcon(
                    embedding,
                    batch_y,
                    batch_speaker,
                    args.temperature,
                )
                if experiment["adversarial"] > 0:
                    speaker_loss = conditional_speaker_loss(
                        embedding,
                        batch_y,
                        batch_speaker,
                        speaker_head,
                        class_speaker_columns,
                        speaker_local_targets_tensor,
                        reversal_scale,
                    )
                else:
                    speaker_loss = embedding.sum() * 0.0
                loss = ce_loss + experiment["supcon"] * supcon_loss + speaker_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_n = len(indices)
                sums += np.asarray(
                    [
                        float(loss.item()),
                        float(ce_loss.item()),
                        float(supcon_loss.item()),
                        float(speaker_loss.item()),
                    ]
                ) * batch_n
                seen += batch_n

            valid_logits = predict(model, valid["X"], mean, std, device)
            valid_predictions = valid_logits.argmax(1)
            val_top1 = float(np.mean(valid_predictions == valid["y"]))
            val_top5 = topk_accuracy(valid_logits, valid["y"])
            val_macro_f1 = float(
                f1_score(
                    valid["y"],
                    valid_predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            row = [
                epoch,
                *(sums / seen).tolist(),
                val_top1,
                val_top5,
                val_macro_f1,
                time.time() - started,
            ]
            with log_path.open("a", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(row)
            if val_macro_f1 > best["macro_f1"]:
                best = {
                    "macro_f1": val_macro_f1,
                    "epoch": epoch,
                    "model": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "speaker_head": {
                        key: value.detach().cpu().clone()
                        for key, value in speaker_head.state_dict().items()
                    },
                }
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"[{experiment['name']} e{epoch}] "
                    f"loss={row[1]:.4f} val_top1={val_top1:.4f} "
                    f"val_top5={val_top5:.4f} macro_f1={val_macro_f1:.4f}",
                    flush=True,
                )

        model.load_state_dict(best["model"])
        test_logits, test_embeddings = predict(
            model, test["X"], mean, std, device, return_embedding=True
        )
        overall = evaluate(test_logits, test["y"], labels)
        foreign_mask = test["y"] != 61
        foreign = evaluate(
            test_logits[foreign_mask, :61], test["y"][foreign_mask], labels[:61]
        )
        diagnostics = embedding_diagnostics(
            test_embeddings,
            test["y"],
            test["speaker"],
            args.seed,
        )
        result = {
            "experiment": experiment,
            "best_epoch": best["epoch"],
            "best_val_macro_f1": best["macro_f1"],
            "overall": overall,
            "foreign_only_61way": foreign,
            "embedding_diagnostics": diagnostics,
        }
        (output_dir / "test_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_classwise(output_dir / "classwise.csv", overall["per_class"])
        torch.save(
            {
                "model": best["model"],
                "speaker_head": best["speaker_head"],
                "mean": mean,
                "std": std,
                "labels": labels,
                "config": vars(args),
                "experiment": experiment,
            },
            output_dir / "best.pt",
        )
        all_results.append(result)
        print(
            experiment["name"],
            {key: value for key, value in overall.items() if key != "per_class"},
            diagnostics,
            flush=True,
        )

    summary_rows = []
    for result in all_results:
        summary_rows.append(
            {
                "experiment": result["experiment"]["name"],
                "best_epoch": result["best_epoch"],
                "overall_top1": result["overall"]["top1"],
                "overall_top5": result["overall"]["top5"],
                "overall_macro_f1": result["overall"]["macro_f1"],
                "foreign_top1": result["foreign_only_61way"]["top1"],
                "foreign_top5": result["foreign_only_61way"]["top5"],
                "foreign_macro_f1": result["foreign_only_61way"]["macro_f1"],
                **result["embedding_diagnostics"],
            }
        )
    with (args.out_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.out_dir / "all_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

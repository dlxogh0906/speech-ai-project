#!/usr/bin/env python3
"""Train/evaluate 31-class heads from cached XLS-R features.

This consumes $NATIONALITY_RUNS/features_fine_811, remaps the original 61 labels
through xlsr_nationality_inference_bundle/labels/label_map.json, then trains
linear/MLP classifier heads on the cached 1920-d XLS-R mean-pooled features.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features_dir", type=Path, default=Path("$NATIONALITY_RUNS/features_fine_811"))
    p.add_argument(
        "--label_map",
        type=Path,
        default=Path("$NATIONALITY_OUTPUTS/xlsr_nationality_inference_bundle/labels/label_map.json"),
    )
    p.add_argument("--out_dir", type=Path, default=Path("$NATIONALITY_OUTPUTS/policy31_feature_probe"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--models", nargs="+", default=["linear", "mlp512"], choices=["linear", "mlp512"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ensemble", action="store_true", help="Evaluate average-logit ensemble of trained heads.")
    return p.parse_args()


def load_npz(features_dir: Path, split: str):
    d = np.load(features_dir / f"{split}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def topk_acc(logits, y, k):
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([int(yi) in row for yi, row in zip(y, topk)]))


def remap_labels(y61, source_id2label, final_label_to_id, aliases):
    mapped = []
    missing = set()
    for yi in y61:
        src = source_id2label[str(int(yi))]
        final = aliases.get(src, src)
        if final not in final_label_to_id:
            missing.add(src)
            mapped.append(-1)
        else:
            mapped.append(final_label_to_id[final])
    if missing:
        raise ValueError(f"Labels missing from policy31 label map: {sorted(missing)}")
    return np.asarray(mapped, dtype=np.int64)


def build_head(kind, dim, num_classes, dropout):
    import torch.nn as nn

    if kind == "linear":
        return nn.Linear(dim, num_classes)
    if kind == "mlp512":
        return nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    raise ValueError(kind)


def evaluate_head(head, X, y, batch=16384):
    import torch
    from sklearn.metrics import f1_score

    head.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i : i + batch]).to(next(head.parameters()).device)
            logits.append(head(xb).float().cpu())
    logits = torch.cat(logits).numpy()
    pred = logits.argmax(1)
    return {
        "logits": logits,
        "top1": topk_acc(logits, y, 1),
        "top5": topk_acc(logits, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def per_class_metrics(logits, y, labels):
    out = {}
    pred = logits.argmax(1)
    for c in range(len(labels)):
        m = y == c
        if not np.any(m):
            out[labels[c]] = {"n": 0, "top1": None}
        else:
            out[labels[c]] = {"n": int(m.sum()), "top1": float((pred[m] == c).mean())}
    return out


def save_portable_checkpoint(path, head, kind, labels, mu, sd, dropout):
    import torch

    hidden_dim = 0 if kind == "linear" else 512
    state = head.state_dict()
    if kind == "linear":
        state = {f"net.{k}": v.detach().cpu() for k, v in state.items()}
    else:
        state = {f"net.{k}": v.detach().cpu() for k, v in state.items()}
    torch.save(
        {
            "classifier": state,
            "hidden_dim": hidden_dim,
            "dropout": 0.0 if kind == "linear" else dropout,
            "labels": labels,
            "x_mean": torch.from_numpy(mu.astype(np.float32)),
            "x_std": torch.from_numpy(sd.astype(np.float32)),
        },
        path,
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn as nn
    from sklearn.metrics import confusion_matrix

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = json.loads(args.label_map.read_text(encoding="utf-8"))
    final_label_to_id = policy["label_to_id"]
    labels = [policy["id_to_label"][str(i)] for i in range(policy["num_labels"])]
    aliases = policy["label_rule"]["aliases"]
    source_id2label = json.loads((args.features_dir / "id2label.json").read_text(encoding="utf-8"))

    t0 = time.time()
    Xtr, ytr61 = load_npz(args.features_dir, "train")
    Xva, yva61 = load_npz(args.features_dir, "valid")
    Xte, yte61 = load_npz(args.features_dir, "test")
    ytr = remap_labels(ytr61, source_id2label, final_label_to_id, aliases)
    yva = remap_labels(yva61, source_id2label, final_label_to_id, aliases)
    yte = remap_labels(yte61, source_id2label, final_label_to_id, aliases)

    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    dim = Xtr.shape[1]
    num_classes = len(labels)
    meta = {
        "features_dir": str(args.features_dir),
        "label_map": str(args.label_map),
        "feature_condition": "cached XLS-R mean-pooled features, sample_rate=16000, max_sec=12",
        "device": device,
        "splits": {
            "train": {"n": int(len(ytr)), "classes_present": int(len(np.unique(ytr)))},
            "valid": {"n": int(len(yva)), "classes_present": int(len(np.unique(yva)))},
            "test": {"n": int(len(yte)), "classes_present": int(len(np.unique(yte)))},
        },
        "dim": int(dim),
        "num_classes": num_classes,
        "labels": labels,
        "load_and_standardize_sec": time.time() - t0,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps({**vars(args), **meta}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    with (args.out_dir / "class_counts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_id", "label", "train", "valid", "test"])
        for i, label in enumerate(labels):
            w.writerow([i, label, int((ytr == i).sum()), int((yva == i).sum()), int((yte == i).sum())])

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    results = {}
    trained = {}

    for kind in args.models:
        print(f"\n===== train {kind} =====", flush=True)
        head = build_head(kind, dim, num_classes, args.dropout).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lossf = nn.CrossEntropyLoss()
        log_path = args.out_dir / f"{kind}_train_log.csv"
        with log_path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_top1", "val_top5", "val_macro_f1", "sec"])

        best = {"val_top5": -1.0, "state": None, "epoch": 0}
        n = Xtr_t.shape[0]
        for epoch in range(1, args.epochs + 1):
            e0 = time.time()
            head.train()
            perm = torch.randperm(n, device=device)
            total = 0.0
            for i in range(0, n, args.batch_size):
                idx = perm[i : i + args.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = lossf(head(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * idx.numel()
            train_loss = total / n
            val = evaluate_head(head, Xva, yva)
            row = [epoch, f"{train_loss:.6f}", f"{val['top1']:.6f}", f"{val['top5']:.6f}", f"{val['macro_f1']:.6f}", f"{time.time() - e0:.2f}"]
            with log_path.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(row)
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"[{kind} e{epoch}] loss={train_loss:.4f} "
                    f"val_top1={val['top1']:.4f} val_top5={val['top5']:.4f} "
                    f"macroF1={val['macro_f1']:.4f}",
                    flush=True,
                )
            if val["top5"] > best["val_top5"]:
                best = {
                    "val_top5": val["top5"],
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "epoch": epoch,
                }

        head.load_state_dict(best["state"])
        test = evaluate_head(head, Xte, yte)
        test_metrics = {
            "model": kind,
            "best_epoch": best["epoch"],
            "best_val_top5": best["val_top5"],
            "test": {
                "n": int(len(yte)),
                "top1": test["top1"],
                "top5": test["top5"],
                "macro_f1": test["macro_f1"],
                "classes_present": int(len(np.unique(yte))),
            },
            "per_class_top1": per_class_metrics(test["logits"], yte, labels),
        }
        cm = confusion_matrix(yte, test["logits"].argmax(1), labels=list(range(num_classes)))
        np.savetxt(args.out_dir / f"{kind}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        (args.out_dir / f"{kind}_test_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        torch.save({"head": best["state"], "mu": mu, "sd": sd, "labels": labels, "kind": kind}, args.out_dir / f"{kind}_best.pt")
        save_portable_checkpoint(args.out_dir / f"{kind}_best_portable.pt", head, kind, labels, mu, sd, args.dropout)
        results[kind] = test_metrics
        trained[kind] = head
        print(
            f"[{kind} TEST] top1={test['top1']:.4f} top5={test['top5']:.4f} "
            f"macroF1={test['macro_f1']:.4f}",
            flush=True,
        )

    if args.ensemble and len(trained) > 1:
        print("\n===== evaluate ensemble average logits =====", flush=True)
        logits = None
        for head in trained.values():
            out = evaluate_head(head, Xte, yte)["logits"]
            logits = out if logits is None else logits + out
        logits /= len(trained)
        from sklearn.metrics import f1_score

        pred = logits.argmax(1)
        ens = {
            "model": "ensemble_avg_logits",
            "members": list(trained.keys()),
            "test": {
                "n": int(len(yte)),
                "top1": topk_acc(logits, yte, 1),
                "top5": topk_acc(logits, yte, 5),
                "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                "classes_present": int(len(np.unique(yte))),
            },
            "per_class_top1": per_class_metrics(logits, yte, labels),
        }
        (args.out_dir / "ensemble_avg_logits_test_metrics.json").write_text(
            json.dumps(ens, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results["ensemble_avg_logits"] = ens
        print(
            f"[ensemble TEST] top1={ens['test']['top1']:.4f} "
            f"top5={ens['test']['top5']:.4f} macroF1={ens['test']['macro_f1']:.4f}",
            flush=True,
        )

    (args.out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

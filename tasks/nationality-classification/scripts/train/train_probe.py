"""Train a linear classifier head on cached XLS-R features; report test top-5 acc.

This is the frozen-encoder baseline: linear head on fixed mean-pooled XLS-R features.
No class weighting / resampling / augmentation (plain baseline, as requested).

    python scripts/train_probe.py --features_dir $NATIONALITY_RUNS/features_fine_811 \
        --out_dir $NATIONALITY_RUNS/probe_fine_811 --epochs 50
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_split(features_dir, split):
    d = np.load(Path(features_dir) / f"{split}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def topk_acc(logits, y, k):
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([yi in row for yi, row in zip(y, topk)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=0, help="0 = linear; >0 = one hidden layer")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    id2label = json.loads((Path(args.features_dir) / "id2label.json").read_text(encoding="utf-8"))
    num_classes = len(id2label)
    labels = [id2label[str(i)] for i in range(num_classes)]

    Xtr, ytr = load_split(args.features_dir, "train")
    Xva, yva = load_split(args.features_dir, "valid")
    Xte, yte = load_split(args.features_dir, "test")
    dim = Xtr.shape[1]
    print(f"train={Xtr.shape} valid={Xva.shape} test={Xte.shape} dim={dim} classes={num_classes}")

    # standardize with train stats
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd

    if args.hidden_dim > 0:
        head = nn.Sequential(nn.Linear(dim, args.hidden_dim), nn.ReLU(),
                             nn.Dropout(args.dropout), nn.Linear(args.hidden_dim, num_classes))
    else:
        head = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(dim, num_classes))
    head = head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).to(dev); ytr_t = torch.from_numpy(ytr).to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)

    def evaluate(Xt, y):
        head.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, Xt.shape[0], 16384):
                logits.append(head(torch.from_numpy(Xt[i:i + 16384]).to(dev)).cpu())
            logits = torch.cat(logits).numpy()
        return logits

    log_path = out_dir / "probe_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_top1", "val_top5", "val_macro_f1"])

    n = Xtr_t.shape[0]
    best_top5, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(head(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        train_loss = tot / n

        vlog = evaluate(Xva, yva)
        v1, v5 = topk_acc(vlog, yva, 1), topk_acc(vlog, yva, 5)
        vf1 = f1_score(yva, vlog.argmax(1), average="macro", zero_division=0)
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{v1:.4f}", f"{v5:.4f}", f"{vf1:.4f}"])
        if epoch % 5 == 0 or epoch == 1:
            print(f"[e{epoch}] loss={train_loss:.4f} val_top1={v1:.4f} val_top5={v5:.4f} macroF1={vf1:.4f}")
        if v5 > best_top5:
            best_top5 = v5
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    # final test
    tlog = evaluate(Xte, yte)
    t1, t5 = topk_acc(tlog, yte, 1), topk_acc(tlog, yte, 5)
    tf1 = f1_score(yte, tlog.argmax(1), average="macro", zero_division=0)
    # per-class top1 (only classes present in test)
    per_class = {}
    for c in np.unique(yte):
        m = yte == c
        per_class[labels[c]] = {"n": int(m.sum()),
                                "top1": float((tlog[m].argmax(1) == c).mean())}
    metrics = {
        "num_classes": num_classes,
        "test": {"n": int(len(yte)), "top1": t1, "top5": t5, "macro_f1": float(tf1),
                 "classes_present": int(len(np.unique(yte)))},
        "best_val_top5": best_top5,
        "head": "linear" if args.hidden_dim == 0 else f"mlp_{args.hidden_dim}",
        "per_class_top1": per_class,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    torch.save({"head": best_state, "mu": mu, "sd": sd, "labels": labels}, out_dir / "head.pt")
    print("\n==================== TEST (baseline) ====================")
    print(f"  n={len(yte):,}  classes_present={len(np.unique(yte))}/{num_classes}")
    print(f"  top1={t1:.4f}  TOP5={t5:.4f}  macro_f1={tf1:.4f}")
    print(f"  -> {out_dir/'test_metrics.json'}")


if __name__ == "__main__":
    main()

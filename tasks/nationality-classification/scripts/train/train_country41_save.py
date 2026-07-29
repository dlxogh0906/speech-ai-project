"""Train 41-class (40 country + Korean) linear head on 12s layer-31 features AND SAVE.

Saves to --out_dir:
  head.pt          head weights + mu/sd + class labels + layer info (reusable for inference)
  metrics.json     overall top1/5 (micro/macro) + per-class + Korean detection
  per_class.csv    per-class top1/top5/test counts
  label2id.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS


import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foreign_dir", default=f"{DATA_ROOT}/taeho_layers_full")
    ap.add_argument("--korean_npz", default=f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
    ap.add_argument("--out_dir", default=f"{NAT_OUTPUTS}/0629_country41_12s")
    ap.add_argument("--layer_idx", type=int, default=0, help="0=layer31, 1=layer35, 2=layer39")
    ap.add_argument("--layer_name", type=int, default=31)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    LI = args.layer_idx

    # foreign speaker -> 40-country
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    Xs, sps = [], []
    for sp in ["train", "valid", "test"]:
        d = np.load(Path(args.foreign_dir) / f"{sp}.npz", allow_pickle=True)
        Xs.append(np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, LI, :])
        sps.extend([str(s) for s in d["speaker"]])
    X = np.concatenate(Xs); sps = np.array(sps); lab = np.array([c40(s) for s in sps], dtype=object)
    keep = lab != None  # noqa
    X, sps, lab = X[keep], sps[keep], lab[keep]
    # korean
    kd = np.load(args.korean_npz, allow_pickle=True)
    kX = np.clip(np.nan_to_num(kd["X"].astype(np.float32)), -50, 50)[:, LI, :]
    kspk = np.array(["KOR:" + str(s) for s in kd["speaker"]])
    X = np.concatenate([X, kX]); sps = np.concatenate([sps, kspk])
    lab = np.concatenate([lab, np.array(["한국어"] * len(kspk), dtype=object)])

    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64); KOR = c2i["한국어"]
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42); split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    print(f"{len(classes)} classes | " + " ".join(f"{s}:{len(idx[s]):,}" for s in idx), flush=True)

    dev = "cuda"
    Xtr, ytr = X[idx["train"]], y[idx["train"]]; Xva, yva = X[idx["valid"]], y[idx["valid"]]; Xte, yte = X[idx["test"]], y[idx["test"]]
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(42); head = nn.Linear(X.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)
    def logits(Xn):
        head.eval()
        with torch.no_grad(): return head(torch.from_numpy(Xn).to(dev)).cpu().numpy()
    def macro5(L, yy):
        i5 = np.argpartition(-L, 4, axis=1)[:, :5]; h = np.array([a in r for a, r in zip(yy, i5)])
        return np.mean([h[yy == c].mean() for c in sorted(set(yy.tolist()))])
    best = -1; bw = None
    for ep in range(args.epochs):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i + 4096]; F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
        m = macro5(logits(Xva), yva)
        if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(bw)

    # --- evaluate ---
    tl = logits(Xte); pred = tl.argmax(1); t5 = np.argpartition(-tl, 4, axis=1)[:, :5]
    hit1 = (pred == yte); hit5 = np.array([yi in r for yi, r in zip(yte, t5)])
    present = sorted(set(yte.tolist()), key=lambda c: -int((yte == c).sum()))
    tespk = defaultdict(set)
    for c, s in zip(yte, sps[idx["test"]]): tespk[int(c)].add(s)
    per_class = {}
    for c in present:
        m = yte == c
        per_class[classes[c]] = {"test_n": int(m.sum()), "test_speakers": len(tespk[c]),
                                 "top1": round(float(hit1[m].mean()), 4), "top5": round(float(hit5[m].mean()), 4)}
    tp = int(((pred == KOR) & (yte == KOR)).sum()); fp = int(((pred == KOR) & (yte != KOR)).sum())
    fn = int(((pred != KOR) & (yte == KOR)).sum())
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    metrics = {
        "kind": "country41_12s_layer31", "num_classes": len(classes),
        "test": {"top1_micro": round(float(hit1.mean()), 4), "top5_micro": round(float(hit5.mean()), 4),
                 "top1_macro": round(float(np.mean([hit1[yte == c].mean() for c in present])), 4),
                 "top5_macro": round(float(np.mean([hit5[yte == c].mean() for c in present])), 4),
                 "classes_present": len(present)},
        "korean": {"precision": round(prec, 4), "recall": round(rec, 4),
                   "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0,
                   "foreign_as_korean_fp": fp, "korean_missed_fn": fn},
        "per_class": per_class,
    }

    # --- SAVE ---
    torch.save({"head": head.state_dict(), "mu": mu, "sd": sd, "labels": classes,
                "c2i": c2i, "layer": args.layer_name, "kind": "country41_12s"}, out / "head.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "label2id.json").write_text(json.dumps(c2i, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_class.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "test_n", "test_speakers", "top1", "top5"])
        for c in present:
            d = per_class[classes[c]]; w.writerow([classes[c], d["test_n"], d["test_speakers"], d["top1"], d["top5"]])

    print(f"\n[saved] -> {out}")
    print(f"  test: top1_micro={metrics['test']['top1_micro']} top5_micro={metrics['test']['top5_micro']} "
          f"top5_macro={metrics['test']['top5_macro']}")
    print(f"  korean: F1={metrics['korean']['f1']} (P={metrics['korean']['precision']} R={metrics['korean']['recall']})")
    print(f"  files: head.pt, metrics.json, per_class.csv, label2id.json")


if __name__ == "__main__":
    main()

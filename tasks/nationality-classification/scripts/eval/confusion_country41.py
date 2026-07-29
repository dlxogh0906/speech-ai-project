"""Rebuild the country41 test set (same pipeline as merge_country41_euro.py, euro=1),
score it with the DEPLOYED head (0629_country41_12s/head.pt), and emit a confusion matrix.

Outputs (in --out_dir):
  confusion_counts.csv     true x pred raw counts (41x41)
  confusion_rownorm.csv    row-normalized (recall per true class)
  top_confusions.csv       per true class: top wrong target + rate
  summary printed to stdout (incl. metrics.json cross-check)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS

import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

ASIA = Path(f"{DATA_ROOT}/taeho_layers_full")
EURO = Path(f"{DATA_ROOT}/taeho_layers_euro/train.npz")
KOR = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
EURO_MANIFEST = f"{NAT_DATA}/euro_country40_manifest.csv"
HEAD = f"{NAT_OUTPUTS}/0629_country41_12s/head.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euro", type=int, default=1)
    ap.add_argument("--euro_sent_only", type=int, default=0)
    ap.add_argument("--head", default=HEAD)
    ap.add_argument("--out_dir", default=f"{NAT_OUTPUTS}/0629_country41_12s/confusion")
    args = ap.parse_args()
    head_path = args.head
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # speaker -> country mapping (identical to build script)
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}

    def c40(s):
        mt = spk_mt.get(s)
        return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    euro_spk2lab = {}
    if args.euro:
        euro_spk2lab = {r["speaker_id"]: r["label_name"]
                        for r in csv.DictReader(open(EURO_MANIFEST, encoding="utf-8"))}
    sent = (np.load(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
            if (args.euro and args.euro_sent_only) else None)

    # ---- a "source plan" = list of (speaker_array, domain, label_fn, row_filter) in build order
    def asia_label(s, i): return c40(str(s))
    def kor_label(s, i): return "한국어"
    def euro_label(s, i): return euro_spk2lab.get(str(s))
    def euro_keep(i): return (sent is None) or bool(sent[i])

    plan = []
    for sp in ["train", "valid", "test"]:
        plan.append((ASIA / f"{sp}.npz", "asia", asia_label, None))
    plan.append((KOR, "korean", kor_label, None))
    if args.euro:
        plan.append((EURO, "euro", euro_label, euro_keep))

    # ---- PASS 1: labels only (no X) -> build sps/lab/dom in exact build order
    sps, lab, dom = [], [], []
    for path, dd, labfn, keep in plan:
        z = np.load(path, allow_pickle=True)
        spk = z["speaker"]
        for i, s in enumerate(spk):
            if keep is not None and not keep(i):
                continue
            l = labfn(s, i)
            if l:
                sps.append(str(s)); lab.append(l); dom.append(dd)
        del z
    sps = np.array(sps); lab = np.array(lab, dtype=object); dom = np.array(dom)
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)

    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    is_te = split == "test"
    print(f"euro={args.euro} sent_only={args.euro_sent_only} | total {len(y):,} | classes {len(classes)} | test {int(is_te.sum()):,}")

    # cross-check class set vs deployed head
    head_ck = torch.load(head_path, map_location="cpu", weights_only=False)
    head_classes = list(head_ck["labels"])
    if classes != sorted(head_classes):
        print("!! WARNING: reconstructed class set != head labels")
        print("   only_in_recon:", sorted(set(classes) - set(head_classes)))
        print("   only_in_head :", sorted(set(head_classes) - set(classes)))
    # head uses its OWN label order; remap our indices -> head index order
    h2i = {c: i for i, c in enumerate(head_classes)}

    # ---- PASS 2: load X per source, keep only test rows (global order)
    g = 0  # global index over kept rows (matches sps/lab order)
    Xte = []
    for path, dd, labfn, keep in plan:
        z = np.load(path, allow_pickle=True)
        spk = z["speaker"]
        Xsrc = np.clip(np.nan_to_num(z["X"].astype(np.float32)), -50, 50)[:, 0, :]  # layer 31
        for i, s in enumerate(spk):
            if keep is not None and not keep(i):
                continue
            if labfn(s, i):
                if is_te[g]:
                    Xte.append(Xsrc[i])
                g += 1
        del z, Xsrc
    assert g == len(sps), (g, len(sps))
    Xte = np.stack(Xte)
    yte_name = lab[is_te]
    yte = np.array([h2i[c] for c in yte_name], np.int64)  # true labels in HEAD index space
    dte = dom[is_te]
    print(f"Xte {Xte.shape}")

    # ---- score with DEPLOYED head (its own mu/sd, weight, bias)
    mu = np.asarray(head_ck["mu"], np.float32); sd = np.asarray(head_ck["sd"], np.float32)
    W = head_ck["head"]["weight"].numpy().astype(np.float32)
    b = head_ck["head"]["bias"].numpy().astype(np.float32)
    Z = (Xte - mu) / sd
    logits = Z @ W.T + b
    pred = logits.argmax(1)
    t5 = np.argpartition(-logits, 4, axis=1)[:, :5]
    hit1 = (pred == yte); hit5 = np.array([yi in r for yi, r in zip(yte, t5)])
    present = sorted(set(yte.tolist()))
    micro1 = hit1.mean(); micro5 = hit5.mean()
    macro1 = np.mean([hit1[yte == c].mean() for c in present])
    macro5 = np.mean([hit5[yte == c].mean() for c in present])
    print(f"CHECK  micro top1={micro1:.4f} top5={micro5:.4f} | macro top1={macro1:.4f} top5={macro5:.4f} | {len(present)}/{len(head_classes)}")
    print("       (metrics.json: micro1=0.8326 top5=0.9631 macro1=0.356 macro5=0.5775)")

    # ---- confusion matrix (HEAD index order)
    N = len(head_classes)
    C = np.zeros((N, N), np.int64)
    for t, p in zip(yte, pred):
        C[t, p] += 1
    with open(out / "confusion_counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + head_classes)
        for i, c in enumerate(head_classes):
            w.writerow([c] + C[i].tolist())
    rown = C / C.sum(1, keepdims=True).clip(min=1)
    with open(out / "confusion_rownorm.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + head_classes)
        for i, c in enumerate(head_classes):
            w.writerow([c] + [round(float(x), 4) for x in rown[i]])

    # ---- top confusions per present true class
    with open(out / "top_confusions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["true", "n", "top1_acc", "top_wrong_pred", "wrong_rate", "2nd_wrong", "rate2"])
        for c in present:
            n = int(C[c].sum()); correct = int(C[c, c])
            wrong = [(head_classes[j], int(C[c, j])) for j in range(N) if j != c and C[c, j] > 0]
            wrong.sort(key=lambda x: -x[1])
            w1, r1 = (wrong[0][0], round(wrong[0][1]/n, 3)) if wrong else ("", 0)
            w2, r2 = (wrong[1][0], round(wrong[1][1]/n, 3)) if len(wrong) > 1 else ("", 0)
            w.writerow([head_classes[c], n, round(correct/n, 3), w1, r1, w2, r2])
    print(f"-> wrote confusion_counts.csv / confusion_rownorm.csv / top_confusions.csv in {out}")

    # ---- print a readable digest: worst classes & where they leak
    print("\n=== 주요 오분류 (정답률 낮은 순, n>=100) ===")
    order = sorted(present, key=lambda c: hit1[yte == c].mean())
    print(f"{'정답국가':<12}{'n':>7}{'top1':>7}  주요 오인 →")
    for c in order:
        n = int(C[c].sum())
        if n < 100:
            continue
        acc = C[c, c] / n
        wrong = [(head_classes[j], C[c, j]) for j in range(N) if j != c and C[c, j] > 0]
        wrong.sort(key=lambda x: -x[1])
        leak = ", ".join(f"{nm} {cc/n*100:.0f}%" for nm, cc in wrong[:3])
        print(f"{head_classes[c]:<12}{n:>7}{acc*100:>6.1f}%  {leak}")


if __name__ == "__main__":
    main()

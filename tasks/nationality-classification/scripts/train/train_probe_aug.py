"""Train linear probe on REAL train features + AUGMENTED minority features; eval test.

Tests empirically whether audio augmentation of minority classes lifts macro top-5.
Compares: baseline (real only) vs real+aug, reporting micro/macro top-5 and per-class
top-5 for the augmented (minority) classes specifically.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import NAT_RUNS


import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    fd = Path(f"{NAT_RUNS}/features_fine_811")
    id2label = json.loads((fd / "id2label.json").read_text()); C = len(id2label)
    labels = [id2label[str(i)] for i in range(C)]
    dev = "cuda"

    def load(s):
        d = np.load(fd / f"{s}.npz", allow_pickle=True)
        return d["X"].astype(np.float32), d["y"].astype(np.int64)
    Xtr, ytr = load("train"); Xva, yva = load("valid"); Xte, yte = load("test")
    aug = np.load(f"{NAT_RUNS}/features_aug/train_aug.npz", allow_pickle=True)
    Xaug, yaug = aug["X"].astype(np.float32), aug["y"].astype(np.int64)
    aug_classes = sorted(set(yaug.tolist()))
    print(f"real train {Xtr.shape}, aug {Xaug.shape} over {len(aug_classes)} classes", flush=True)

    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    def norm(X): return (X - mu) / sd
    Xtr_n, Xva_n, Xte_n, Xaug_n = norm(Xtr), norm(Xva), norm(Xte), norm(Xaug)

    def topk_hit(L, y, k=5):
        idx = np.argpartition(-L, k - 1, axis=1)[:, :k]
        return np.array([yi in r for yi, r in zip(y, idx)])

    def run(use_aug):
        if use_aug:
            X = np.concatenate([Xtr_n, Xaug_n]); Y = np.concatenate([ytr, yaug])
        else:
            X, Y = Xtr_n, ytr
        torch.manual_seed(42)
        head = nn.Linear(X.shape[1], C).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.from_numpy(X).to(dev); Yt = torch.from_numpy(Y).to(dev); n = len(Yt)
        best = -1; best_te = None
        for ep in range(60):
            head.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]
                F.cross_entropy(head(Xt[idx]), Yt[idx]).backward(); opt.step(); opt.zero_grad()
            head.eval()
            with torch.no_grad():
                vl = head(torch.from_numpy(Xva_n).to(dev)).cpu().numpy()
            present = sorted(set(yva.tolist()))
            hit = topk_hit(vl, yva); macro = np.mean([hit[yva == c].mean() for c in present])
            if macro > best:
                best = macro
                with torch.no_grad():
                    best_te = head(torch.from_numpy(Xte_n).to(dev)).cpu().numpy()
        # test metrics
        hit = topk_hit(best_te, yte)
        present = sorted(set(yte.tolist()))
        micro = hit.mean(); macro = np.mean([hit[yte == c].mean() for c in present])
        per = {labels[c]: float(hit[yte == c].mean()) for c in present}
        return micro, macro, per

    mi0, ma0, p0 = run(False)
    mi1, ma1, p1 = run(True)
    print("\n================ 결과 (test top-5) ================")
    print(f"{'':22s}{'micro':>8}{'MACRO':>8}")
    print(f"{'baseline (real only)':22s}{mi0:>8.4f}{ma0:>8.4f}")
    print(f"{'+ audio augmentation':22s}{mi1:>8.4f}{ma1:>8.4f}")
    print(f"{'Δ':22s}{mi1-mi0:>+8.4f}{ma1-ma0:>+8.4f}")
    print("\n증강한 (소수) 클래스별 top-5 변화:")
    aug_names = set(labels[c] for c in aug_classes)
    rows = [(n, p0.get(n, 0), p1.get(n, 0)) for n in p0 if n in aug_names]
    for n, a, b in sorted(rows, key=lambda x: -(x[2] - x[1])):
        print(f"   {n:12s} {a:.3f} -> {b:.3f}  ({b-a:+.3f})")
    out = {"baseline": {"micro": mi0, "macro": ma0, "per_class": p0},
           "with_aug": {"micro": mi1, "macro": ma1, "per_class": p1}}
    Path(f"{NAT_RUNS}/aug_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nsaved -> $NATIONALITY_RUNS/aug_result.json")


if __name__ == "__main__":
    main()

"""Probe each XLS-R layer + a learned weighted-sum; find best for MACRO top-5.

Uses per-layer pooled features from extract_layer_features.py. Reports, per layer,
test micro/macro top-5; then trains a learned softmax weighting over layers + linear
head. Compares against the top-layer (frozen baseline = macro 0.50 on full data).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_RUNS


import json
from pathlib import Path

import numpy as np


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=f"{DATA_ROOT}/taeho_layers")
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    fd = Path(args.features_dir)
    id2label = json.loads(Path(f"{NAT_RUNS}/features_fine_811/id2label.json").read_text())
    C = len(id2label)
    dev = "cuda"

    def load(s):
        d = np.load(fd / f"{s}.npz", allow_pickle=True)
        X = np.nan_to_num(d["X"].astype(np.float32))   # fp16 overflow -> inf in some layers
        X = np.clip(X, -50.0, 50.0)                     # tame outliers (real values |x| mostly < 10)
        return X, d["y"].astype(np.int64), d["layers"]
    Xtr, ytr, layers = load("train"); Xva, yva, _ = load("valid"); Xte, yte, _ = load("test")
    L = list(layers.tolist())
    print(f"train {Xtr.shape} valid {Xva.shape} test {Xte.shape} layers={L}", flush=True)

    def topk_hit(P, y, k=5):
        idx = np.argpartition(-P, k - 1, axis=1)[:, :k]
        return np.array([yi in r for yi, r in zip(y, idx)])
    def mami(P, y):
        hit = topk_hit(P, y); pres = sorted(set(y.tolist()))
        return float(hit.mean()), float(np.mean([hit[y == c].mean() for c in pres]))

    def train_head(Xtr2d, Xva2d, Xte2d, epochs=60):
        mu = Xtr2d.mean(0, keepdims=True); sd = Xtr2d.std(0, keepdims=True) + 1e-6
        a, b, c = (Xtr2d - mu) / sd, (Xva2d - mu) / sd, (Xte2d - mu) / sd
        torch.manual_seed(42)
        head = nn.Linear(a.shape[1], C).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        At = torch.from_numpy(a).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)
        best = -1; best_te = None
        for ep in range(epochs):
            head.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]
                F.cross_entropy(head(At[idx]), Yt[idx]).backward(); opt.step(); opt.zero_grad()
            head.eval()
            with torch.no_grad():
                vl = head(torch.from_numpy(b).to(dev)).cpu().numpy()
            _, vma = mami(vl, yva)
            if vma > best:
                best = vma
                with torch.no_grad():
                    best_te = head(torch.from_numpy(c).to(dev)).cpu().numpy()
        return mami(best_te, yte)

    print("\n=== per-layer linear probe (test) ===")
    print(f"{'layer':>6}{'micro':>9}{'MACRO':>9}")
    results = {}
    for li, lyr in enumerate(L):
        mi, ma = train_head(Xtr[:, li, :], Xva[:, li, :], Xte[:, li, :])
        results[f"layer{lyr}"] = {"micro": mi, "macro": ma}
        print(f"{lyr:>6}{mi:>9.4f}{ma:>9.4f}", flush=True)

    # --- learned weighted sum over layers (per-layer standardized) ---
    print("\n=== learned weighted-sum of layers ===", flush=True)
    mus = Xtr.mean(0, keepdims=True); sds = Xtr.std(0, keepdims=True) + 1e-6
    Atr = (Xtr - mus) / sds; Ava = (Xva - mus) / sds; Ate = (Xte - mus) / sds  # (N,L,C)
    Atr_t = torch.from_numpy(Atr).to(dev); Yt = torch.from_numpy(ytr).to(dev)
    Ava_t = torch.from_numpy(Ava).to(dev); Ate_t = torch.from_numpy(Ate).to(dev)
    nL = len(L)
    w = torch.zeros(nL, device=dev, requires_grad=True)
    head = nn.Linear(Xtr.shape[2], C).to(dev)
    opt = torch.optim.AdamW([w] + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    n = Atr_t.shape[0]; best = -1; best_te = None; best_w = None
    for ep in range(60):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            ws = torch.softmax(w, 0)
            x = (Atr_t[idx] * ws[None, :, None]).sum(1)
            F.cross_entropy(head(x), Yt[idx]).backward(); opt.step(); opt.zero_grad()
        head.eval()
        with torch.no_grad():
            ws = torch.softmax(w, 0)
            vl = head((Ava_t * ws[None, :, None]).sum(1)).cpu().numpy()
        _, vma = mami(vl, yva)
        if vma > best:
            best = vma
            with torch.no_grad():
                best_te = head((Ate_t * ws[None, :, None]).sum(1)).cpu().numpy()
                best_w = ws.detach().cpu().numpy()
    wmi, wma = mami(best_te, yte)
    results["weighted_sum"] = {"micro": wmi, "macro": wma,
                               "weights": {int(l): float(x) for l, x in zip(L, best_w)}}
    print(f"weighted-sum  micro={wmi:.4f}  MACRO={wma:.4f}")
    print("layer weights:", {int(l): round(float(x), 3) for l, x in zip(L, best_w)})

    best_layer = max((k for k in results if k.startswith("layer")), key=lambda k: results[k]["macro"])
    print(f"\n=== SUMMARY ===")
    print(f"top-layer(47) baseline (full data): macro 0.498 / micro 0.931")
    print(f"best single layer: {best_layer} macro {results[best_layer]['macro']:.4f}")
    print(f"weighted-sum: macro {wma:.4f}")
    Path(f"{NAT_RUNS}/layer_probe_result.json").write_text(json.dumps(results, indent=2))
    print("saved -> $NATIONALITY_RUNS/layer_probe_result.json")


if __name__ == "__main__":
    main()

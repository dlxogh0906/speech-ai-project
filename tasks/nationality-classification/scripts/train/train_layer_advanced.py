"""Push macro top-5 on cached layer features (31/35/39) — no new extraction.

Variants (all on $SPEECH_AI_DATA/taeho_layers_full):
  single31            linear head on layer 31 (control, = current best 0.578)
  concat              linear head on concat[31,35,39]
  concat_mlp          MLP head on concat
  concat_adv          concat + speaker-ADVERSARIAL (gradient reversal removes speaker
                      identity from the shared rep -> forces accent; research top lever)
Reports test micro/macro top-5 (selected by valid macro top-5).
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
from shared.paths import DATA_ROOT, NAT_RUNS


import json
from pathlib import Path

import numpy as np


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.autograd import Function

    fd = Path(f"{DATA_ROOT}/taeho_layers_full")
    id2label = json.loads(Path(f"{NAT_RUNS}/features_fine_811/id2label.json").read_text())
    C = len(id2label)
    dev = "cuda"

    def load(s):
        d = np.load(fd / f"{s}.npz", allow_pickle=True)
        X = np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)  # (N,L,1920) L=[31,35,39]
        return X, d["y"].astype(np.int64), np.array([str(x) for x in d["speaker"]])
    Xtr, ytr, spk = load("train"); Xva, yva, _ = load("valid"); Xte, yte, _ = load("test")
    L = Xtr.shape[1]
    # speaker ids (train only) for adversarial
    uspk = {s: i for i, s in enumerate(sorted(set(spk)))}
    sid = np.array([uspk[s] for s in spk], np.int64); n_spk = len(uspk)
    print(f"train {Xtr.shape} layers={L} speakers={n_spk}", flush=True)

    def sel(X, mode):
        if mode == "single31":
            return X[:, 0, :]                       # layer 31 is index 0
        return X.reshape(X.shape[0], -1)            # concat all layers

    def standardize(a, b, c):
        mu = a.mean(0, keepdims=True); sd = a.std(0, keepdims=True) + 1e-6
        return (a - mu) / sd, (b - mu) / sd, (c - mu) / sd

    def topk(P, y, k=5):
        idx = np.argpartition(-P, k - 1, axis=1)[:, :k]
        return np.array([yi in r for yi, r in zip(y, idx)])
    def mami(P, y):
        h = topk(P, y); pres = sorted(set(y.tolist()))
        return float(h.mean()), float(np.mean([h[y == c].mean() for c in pres]))

    class GRL(Function):
        @staticmethod
        def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
        @staticmethod
        def backward(ctx, g): return -ctx.lam * g, None

    def run(mode):
        concat = mode != "single31"
        Xa, Xb, Xc = (sel(Xtr, mode), sel(Xva, mode), sel(Xte, mode))
        Xa, Xb, Xc = standardize(Xa, Xb, Xc)
        D = Xa.shape[1]
        adv = mode == "concat_adv"
        mlp = mode == "concat_mlp"
        torch.manual_seed(42)
        if mlp:
            shared = nn.Sequential(nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.1)).to(dev); hd = 512
        elif adv:
            shared = nn.Sequential(nn.Linear(D, 512), nn.ReLU()).to(dev); hd = 512
        else:
            shared = nn.Identity().to(dev); hd = D
        cls = nn.Linear(hd, C).to(dev)
        spkh = nn.Linear(hd, n_spk).to(dev) if adv else None
        params = list(shared.parameters()) + list(cls.parameters()) + (list(spkh.parameters()) if adv else [])
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        At = torch.from_numpy(Xa).to(dev); Yt = torch.from_numpy(ytr).to(dev)
        St = torch.from_numpy(sid).to(dev); n = len(Yt)
        best = -1; best_te = None
        for ep in range(60):
            shared.train(); cls.train()
            if adv: spkh.train()
            lam = 0.3 * min(1.0, ep / 15)  # ramp adversarial strength
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]
                h = shared(At[idx])
                loss = F.cross_entropy(cls(h), Yt[idx])
                if adv:
                    loss = loss + F.cross_entropy(spkh(GRL.apply(h, lam)), St[idx])
                loss.backward(); opt.step(); opt.zero_grad()
            shared.eval(); cls.eval()
            with torch.no_grad():
                vl = cls(shared(torch.from_numpy(Xb).to(dev))).cpu().numpy()
            _, vma = mami(vl, yva)
            if vma > best:
                best = vma
                with torch.no_grad():
                    best_te = cls(shared(torch.from_numpy(Xc).to(dev))).cpu().numpy()
        return mami(best_te, yte)

    print(f"\n{'variant':14s}{'micro':>9}{'MACRO':>9}")
    res = {}
    for mode in ["single31", "concat", "concat_mlp", "concat_adv"]:
        mi, ma = run(mode); res[mode] = {"micro": mi, "macro": ma}
        print(f"{mode:14s}{mi:>9.4f}{ma:>9.4f}", flush=True)
    print(f"\nbaseline(L47): macro 0.498 / micro 0.937 ; layer31: macro 0.578")
    Path(f"{NAT_RUNS}/layer_advanced_result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

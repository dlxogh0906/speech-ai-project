"""Punchline viz: augmented points pile onto the SOURCE speaker, never reaching the
unseen TEST speaker -> augmentation adds density, not speaker coverage.

Per minority class, t-SNE over {real train (by speaker), augmented (by source speaker),
test (by speaker)}.
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
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FD = Path(f"{NAT_RUNS}/features_fine_811")
OUT = Path(f"{NAT_RUNS}/viz"); OUT.mkdir(parents=True, exist_ok=True)
id2label = json.loads((FD / "id2label.json").read_text())
labels = [id2label[str(i)] for i in range(len(id2label))]
name2id = {n: i for i, n in enumerate(labels)}
EN = {"광동어": "Cantonese", "우즈베크어": "Uzbek", "카자흐어": "Kazakh", "터키어": "Turkish",
      "말레이어": "Malay", "네팔어": "Nepali", "크메르어": "Khmer"}


def load(s):
    d = np.load(FD / f"{s}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64), np.array([str(x) for x in d["speaker"]])


def main():
    Xtr, ytr, str_ = load("train"); Xte, yte, ste = load("test")
    aug = np.load(f"{NAT_RUNS}/features_aug/train_aug.npz", allow_pickle=True)
    Xa, ya, spa = aug["X"].astype(np.float32), aug["y"].astype(np.int64), np.array([str(x) for x in aug["speaker"]])

    targets = [n for n in ["우즈베크어", "카자흐어", "터키어"] if name2id.get(n) in set(ya.tolist())]
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 6.5))
    if len(targets) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for ax, name in zip(axes, targets):
        c = name2id[name]
        rtr = np.where(ytr == c)[0]
        ia = np.where(ya == c)[0]
        if len(ia) > 600:
            ia = rng.choice(ia, 600, replace=False)
        ite = np.where(yte == c)[0]
        X = np.concatenate([Xtr[rtr], Xa[ia], Xte[ite]])
        grp = (["train"] * len(rtr)) + (["aug"] * len(ia)) + (["test"] * len(ite))
        spk = list(str_[rtr]) + list(spa[ia]) + list(ste[ite])
        grp = np.array(grp); spk = np.array(spk)
        Z = TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 6)),
                 init="pca", random_state=0).fit_transform(X)
        tr_spk = sorted(set(str_[rtr])); col = {s: plt.cm.tab10(i % 10) for i, s in enumerate(tr_spk)}
        # real train (o), augmented (small +, source-speaker color), test (big X black-edge)
        m = grp == "train"
        ax.scatter(Z[m, 0], Z[m, 1], c=[col.get(s, "gray") for s in spk[m]], marker="o", s=55,
                   alpha=0.9, edgecolors="k", linewidths=0.4, label="real train (color=speaker)")
        m = grp == "aug"
        ax.scatter(Z[m, 0], Z[m, 1], c=[col.get(s, "gray") for s in spk[m]], marker="+", s=30,
                   alpha=0.45, label="augmented (same speakers)")
        m = grp == "test"
        ax.scatter(Z[m, 0], Z[m, 1], c="red", marker="X", s=160, edgecolors="k", linewidths=0.9,
                   label="TEST (unseen speaker)")
        ax.set_title(f"{EN.get(name,name)}: train {len(set(str_[rtr]))} spk, test {len(set(ste[ite]))} spk")
        ax.legend(fontsize=8, loc="best"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Augmentation (+) piles onto the SAME training speakers; the unseen TEST "
                 "speaker (red X) sits apart -> aug can't cover it.", fontsize=13)
    plt.tight_layout(); plt.savefig(OUT / "C_augmentation.png", dpi=130); plt.close()
    print(f"saved -> {OUT}/C_augmentation.png")


if __name__ == "__main__":
    main()

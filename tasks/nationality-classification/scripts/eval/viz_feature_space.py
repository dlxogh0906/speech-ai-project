"""Visualize WHY minority classes fail: XLS-R feature space clusters by speaker.

Two figures (saved as PNG):
  A) t-SNE of features for a few major + minority classes, train (o) vs test (x).
     Majors form clean separable regions; minorities are tiny/overlapping and their
     TEST points (unseen speakers) scatter into the wrong regions.
  B) For minority classes, color points by SPEAKER -> each class is a few speaker-blobs,
     and test = a NEW speaker that lands away from the train blob (=> misclassified).
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
label2id = {v: k for k, v in {int(k): v for k, v in id2label.items()}.items()}
labels = [id2label[str(i)] for i in range(len(id2label))]

# matplotlib has no Korean font here -> romanize the labels we plot
EN = {"중국어": "Chinese", "일본어": "Japanese", "베트남어": "Vietnamese", "태국어": "Thai",
      "광동어": "Cantonese", "우즈베크어": "Uzbek", "크메르어": "Khmer", "카자흐어": "Kazakh",
      "힌디어": "Hindi", "몽골어": "Mongolian", "러시아어": "Russian"}
def en(name): return EN.get(name, name)


def load(split):
    d = np.load(FD / f"{split}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64), np.array([str(s) for s in d["speaker"]])


def main():
    Xtr, ytr, str_ = load("train")
    Xte, yte, ste = load("test")
    name2id = {n: i for i, n in enumerate(labels)}

    majors = ["중국어", "일본어", "베트남어"]
    minors = ["광동어", "우즈베크어", "크메르어", "카자흐어", "힌디어"]
    sel = [name2id[n] for n in majors + minors if n in name2id]

    rng = np.random.default_rng(0)
    Xs, ys, sp, src = [], [], [], []
    for c in sel:
        # train: cap per class for tsne
        itr = np.where(ytr == c)[0]
        if len(itr) > 250:
            itr = rng.choice(itr, 250, replace=False)
        ite = np.where(yte == c)[0]
        if len(ite) > 120:
            ite = rng.choice(ite, 120, replace=False)
        for i in itr:
            Xs.append(Xtr[i]); ys.append(c); sp.append("tr:" + str_[i]); src.append("train")
        for i in ite:
            Xs.append(Xte[i]); ys.append(c); sp.append("te:" + ste[i]); src.append("test")
    Xs = np.array(Xs); ys = np.array(ys); src = np.array(src); sp = np.array(sp)
    print(f"tsne on {len(Xs)} points ...", flush=True)
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Xs)

    # --- Figure A: colored by class, train(o) vs test(x) ---
    plt.figure(figsize=(12, 9))
    cmap = plt.cm.tab10
    for j, c in enumerate(sel):
        col = cmap(j % 10)
        is_major = labels[c] in majors
        for s, mk, sz, al in [("train", "o", 22, 0.4), ("test", "x", 44, 0.95)]:
            m = (ys == c) & (src == s)
            lab = f"{en(labels[c])} {'[MAJOR]' if is_major else '[minor]'}" if s == "train" else None
            plt.scatter(Z[m, 0], Z[m, 1], c=[col], marker=mk, s=sz, alpha=al,
                        edgecolors="none", label=lab)
    plt.title("XLS-R feature space (t-SNE)  |  o = train,  x = test (unseen speakers)\n"
              "MAJOR classes form tight separable clusters; minor classes scatter / overlap")
    plt.legend(fontsize=9, loc="best", ncol=2)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout(); plt.savefig(OUT / "A_by_class.png", dpi=130); plt.close()

    # --- Figure B: minority classes colored by speaker ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, name in zip(axes, ["광동어", "우즈베크어", "카자흐어"]):
        if name not in name2id:
            continue
        c = name2id[name]
        m = ys == c
        spk_here = sp[m]
        uniq = sorted(set(s.split(":", 1)[1] for s in spk_here))
        spk_col = {u: plt.cm.tab20(k % 20) for k, u in enumerate(uniq)}
        for s, mk, sz in [("train", "o", 40), ("test", "X", 130)]:
            mm = m & (src == s)
            cols = [spk_col[x.split(":", 1)[1]] for x in sp[mm]]
            ax.scatter(Z[mm, 0], Z[mm, 1], c=cols, marker=mk, s=sz,
                       edgecolors="k" if s == "test" else "none", linewidths=0.7, alpha=0.85)
        ntr = len(set(x.split(":",1)[1] for x in sp[m & (src=="train")]))
        nte = len(set(x.split(":",1)[1] for x in sp[m & (src=="test")]))
        ax.set_title(f"{en(name)}: color=speaker  o=train({ntr} spk)  X=test({nte} spk, unseen)")
        ax.set_xlabel("t-SNE 1")
    fig.suptitle("Minor classes = a few speaker-blobs. Test speakers (X) land AWAY from "
                 "the train blobs (o) -> misclassified. Duplication/aug can't fill that gap.", fontsize=13)
    plt.tight_layout(); plt.savefig(OUT / "B_by_speaker.png", dpi=130); plt.close()
    print(f"saved -> {OUT}/A_by_class.png , {OUT}/B_by_speaker.png")


if __name__ == "__main__":
    main()

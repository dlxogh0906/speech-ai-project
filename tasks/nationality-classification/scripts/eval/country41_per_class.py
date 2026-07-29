"""Per-class top-1/top-5 for the 41-class (40 country + Korean) model."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA

import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

FEAT = Path(f"{DATA_ROOT}/taeho_layers_full"); KOREAN = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)
    Xs, sps = [], []
    for sp in ["train", "valid", "test"]:
        d = np.load(FEAT / f"{sp}.npz", allow_pickle=True)
        Xs.append(np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, 0, :]); sps.extend([str(s) for s in d["speaker"]])
    X = np.concatenate(Xs); sps = np.array(sps); lab = np.array([c40(s) for s in sps], dtype=object)
    keep = lab != None  # noqa
    X, sps, lab = X[keep], sps[keep], lab[keep]
    kd = np.load(KOREAN, allow_pickle=True)
    kX = np.clip(np.nan_to_num(kd["X"].astype(np.float32)), -50, 50)[:, 0, :]
    kspk = np.array(["KOR:" + str(s) for s in kd["speaker"]]); klab = np.array(["한국어"] * len(kspk), dtype=object)
    X = np.concatenate([X, kX]); sps = np.concatenate([sps, kspk]); lab = np.concatenate([lab, klab])
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42); split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    dev = "cuda"
    Xtr, ytr = X[idx["train"]], y[idx["train"]]; Xva, yva = X[idx["valid"]], y[idx["valid"]]; Xte, yte = X[idx["test"]], y[idx["test"]]
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(42); head = nn.Linear(1920, len(classes)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)
    def logits(Xn):
        head.eval()
        with torch.no_grad(): return head(torch.from_numpy(Xn).to(dev)).cpu().numpy()
    def macro5(L, yy):
        i5 = np.argpartition(-L, 4, axis=1)[:, :5]; h = np.array([a in r for a, r in zip(yy, i5)])
        return np.mean([h[yy == c].mean() for c in sorted(set(yy.tolist()))])
    best = -1; bw = None
    for ep in range(60):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i + 4096]; F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
        m = macro5(logits(Xva), yva)
        if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(bw); tl = logits(Xte); pred = tl.argmax(1)
    t5 = np.argpartition(-tl, 4, axis=1)[:, :5]
    # 화자수(test)
    tespk = defaultdict(set)
    for c, s in zip(yte, sps[idx["test"]]): tespk[int(c)].add(s)
    # 전체 발화수(클래스별)
    tot = Counter(lab.tolist())
    present = sorted(set(yte.tolist()), key=lambda c: -int((yte == c).sum()))
    print(f"{'국가/클래스':<14}{'test수':>7}{'test화자':>8}{'top1':>8}{'top5':>8}")
    print("-" * 47)
    for c in present:
        m = yte == c; nn_ = int(m.sum())
        a1 = float((pred[m] == c).mean()); a5 = float(np.mean([c in t5[i] for i in np.where(m)[0]]))
        mark = " ★KR" if classes[c] == "한국어" else ""
        print(f"{classes[c]:<14}{nn_:>7,}{len(tespk[c]):>8}{a1:>8.3f}{a5:>8.3f}{mark}")
    print("-" * 47)
    hit1 = (pred == yte); i5 = t5; hit5 = np.array([yi in r for yi, r in zip(yte, i5)])
    print(f"{'micro 평균':<14}{len(yte):>7,}{'':>8}{hit1.mean():>8.3f}{hit5.mean():>8.3f}")
    print(f"{'MACRO 평균':<14}{'':>7}{'':>8}{np.mean([hit1[yte==c].mean() for c in present]):>8.3f}{np.mean([hit5[yte==c].mean() for c in present]):>8.3f}")
    print(f"\n(test 등장 {len(present)}/{len(classes)} 클래스)")


if __name__ == "__main__":
    main()

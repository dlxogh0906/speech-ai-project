"""Train a merged-taxonomy linear probe on cached layer-31 features (euro=0 data),
using the SAME data assembly + speaker split as merge_country41_euro.py, and compare
to the 41-class baseline. Encoder is NOT re-run.

13 region groups (data-driven from the country41 confusion).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS

import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

ASIA = Path(f"{DATA_ROOT}/taeho_layers_full")
KOR = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
OUT = Path(f"{NAT_OUTPUTS}/0701_taxonomy13"); OUT.mkdir(parents=True, exist_ok=True)

GROUP = {
    "한국어": "한국(내국인)",
    "중국": "중국·홍콩", "홍콩": "중국·홍콩",
    "일본": "일본", "베트남": "베트남", "태국": "태국", "몽골": "몽골", "미얀마": "미얀마",
    "러시아": "러시아·중앙아시아", "카자흐스탄": "러시아·중앙아시아", "우즈베키스탄": "러시아·중앙아시아",
    "키르기스스탄": "러시아·중앙아시아", "타지키스탄": "러시아·중앙아시아",
    "투르크메니스탄": "러시아·중앙아시아", "우크라이나": "러시아·중앙아시아",
    "인도네시아": "동남아", "말레이시아": "동남아", "필리핀": "동남아", "싱가포르": "동남아", "캄보디아": "동남아",
    "네팔": "남아시아", "인도": "남아시아", "방글라데시": "남아시아", "파키스탄": "남아시아",
    "미국": "서구권", "영국": "서구권", "독일": "서구권", "프랑스": "서구권", "이탈리아": "서구권",
    "스페인": "서구권", "루마니아": "서구권", "불가리아": "서구권", "튀르키예": "서구권", "이란": "서구권",
    "르완다": "아프리카", "에티오피아": "아프리카", "잠비아": "아프리카",
    "브라질": "중남미", "칠레": "중남미", "콜롬비아": "중남미", "페루": "중남미",
}


def main():
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}

    def c40(s):
        mt = spk_mt.get(s)
        return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    X, sps, lab = [], [], []
    for sp in ["train", "valid", "test"]:
        d = np.load(ASIA / f"{sp}.npz", allow_pickle=True)
        Xa = np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, 0, :]
        for i, s in enumerate(d["speaker"]):
            c = c40(str(s))
            if c and c in GROUP:
                X.append(Xa[i]); sps.append(str(s)); lab.append(GROUP[c])
        del d, Xa
    kd = np.load(KOR, allow_pickle=True)
    kX = np.clip(np.nan_to_num(kd["X"].astype(np.float32)), -50, 50)[:, 0, :]
    for i, s in enumerate(kd["speaker"]):
        X.append(kX[i]); sps.append("KOR:" + str(s)); lab.append("한국(내국인)")
    del kd, kX

    X = np.stack(X); sps = np.array(sps); lab = np.array(lab, dtype=object)
    classes = sorted(set(lab.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    print(f"{len(y):,} utts | {len(classes)} groups | test {len(idx['test']):,}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = idx["train"], idx["valid"], idx["test"]
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xtr = ((X[tr]-mu)/sd); Xva = ((X[va]-mu)/sd); Xte = ((X[te]-mu)/sd)
    ytr, yva, yte = y[tr], y[va], y[te]

    torch.manual_seed(42); head = nn.Linear(X.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)

    def logits(Xn):
        head.eval()
        with torch.no_grad():
            return head(torch.from_numpy(Xn).to(dev)).cpu().numpy()

    def macro5(L, yy):
        i5 = np.argpartition(-L, 4, axis=1)[:, :5]
        h = np.array([a in r for a, r in zip(yy, i5)])
        return np.mean([h[yy == c].mean() for c in sorted(set(yy.tolist()))])

    best = -1; bw = None
    for ep in range(60):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i+4096]
            F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
        m = macro5(logits(Xva), yva)
        if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(bw)

    tl = logits(Xte); pred = tl.argmax(1); t5 = np.argpartition(-tl, 4, axis=1)[:, :5]
    hit1 = (pred == yte); hit5 = np.array([yi in r for yi, r in zip(yte, t5)])
    present = sorted(set(yte.tolist()))
    o = {"micro1": float(hit1.mean()), "micro5": float(hit5.mean()),
         "macro1": float(np.mean([hit1[yte == c].mean() for c in present])),
         "macro5": float(np.mean([hit5[yte == c].mean() for c in present])),
         "nclass_present": len(present), "nclass": len(classes)}
    print(f"\n=== 13-group taxonomy (test) ===")
    print(f"micro top1={o['micro1']:.4f} top5={o['micro5']:.4f} | macro top1={o['macro1']:.4f} top5={o['macro5']:.4f} | {len(present)}/{len(classes)}")
    print(f"(baseline country41: micro1=0.8326 top5=0.9631 macro1=0.3560 macro5=0.5775)\n")
    print(f"{'그룹':<16}{'n':>8}{'top1':>8}{'top5':>8}")
    for c in sorted(present, key=lambda c: -int((yte == c).sum())):
        m = yte == c; nn_ = int(m.sum())
        print(f"{classes[c]:<16}{nn_:>8,}{hit1[m].mean()*100:>7.1f}%{hit5[m].mean()*100:>7.1f}%")

    # save head (demo-compatible format) + confusion
    torch.save({"head": {"weight": head.state_dict()["weight"].cpu(),
                          "bias": head.state_dict()["bias"].cpu()},
                "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                "labels": classes, "c2i": c2i, "layer": 31, "kind": "taxonomy13_12s"},
               OUT / "head.pt")
    N = len(classes); C = np.zeros((N, N), int)
    for t, p in zip(yte, pred): C[t, p] += 1
    with open(OUT / "confusion_counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + classes)
        for i, c in enumerate(classes): w.writerow([c] + C[i].tolist())
    json.dump({**o, "groups": GROUP}, open(OUT / "metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n-> saved head.pt / confusion_counts.csv / metrics.json in {OUT}")


if __name__ == "__main__":
    main()

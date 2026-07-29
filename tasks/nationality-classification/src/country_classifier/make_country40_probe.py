"""40-country relabel + new speaker-disjoint split + linear probe (on cached layer-31).

Relabels each utterance from motherTongue(61) to a 40-COUNTRY scheme:
  - 영어 -> 미국/영국 (by residence country US/GB; other English countries dropped)
  - 스페인어 -> 페루/콜롬비아/스페인/칠레 (by country PE/CO/ES/CL; others dropped)
  - every other motherTongue -> its single home country
  - rare tongues not in the 40-set are dropped
Reuses cached layer-31 features (whole corpus), re-splits speaker-disjoint stratified on
the 40-country labels, trains a linear head, and reports top-1/top-5 + the English/Spanish
within-language confusion (does the model separate US vs GB, Peru vs Chile?).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA


import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from country_classifier.make_speaker_split import assign_speakers

# split-by-country languages: motherTongue -> {residence_country: 국가명}
SPLIT = {
    "영어": {"US": "미국", "GB": "영국"},
    "스페인어": {"PE": "페루", "CO": "콜롬비아", "ES": "스페인", "CL": "칠레"},
}
# single-country languages: motherTongue -> 국가명 (post-merged6 names as in our data)
LANG2C = {
    "베트남어": "베트남", "일본어": "일본", "중국어": "중국", "태국어": "태국", "몽골어": "몽골",
    "러시아어": "러시아", "미얀마어": "미얀마", "인도네시아어": "인도네시아", "광동어": "홍콩",
    "우즈베크어": "우즈베키스탄", "말레이어": "말레이시아", "터키어": "튀르키예", "네팔어": "네팔",
    "타갈로그어": "필리핀", "크메르어": "캄보디아", "독일어": "독일", "키르기스어": "키르기스스탄",
    "카자흐어": "카자흐스탄", "암하라어": "에티오피아", "루마니아어": "루마니아", "키냐르완다어": "르완다",
    "힌디어": "인도", "타지크어": "타지키스탄", "우크라이나어": "우크라이나", "투르크멘어": "투르크메니스탄",
    "페르시아어": "이란", "이탈리아어": "이탈리아", "불가리아어": "불가리아", "잠비아어": "잠비아",
    "프랑스어": "프랑스", "방글라데시어": "방글라데시", "포르투갈어": "브라질", "싱가포르어": "싱가포르",
    "우르두어": "파키스탄",
}
FEAT = Path(f"{DATA_ROOT}/taeho_layers_full")


def main():
    import torch, torch.nn as nn, torch.nn.functional as F

    # speaker -> (motherTongue, majority residence country)
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}
    spk_country = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_country[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_country.items()}

    def country40(spk):
        mt = spk_mt.get(spk)
        if mt in SPLIT:
            return SPLIT[mt].get(spk_c.get(spk))      # None if country not in allowed set -> drop
        return LANG2C.get(mt)                          # None if rare tongue -> drop

    # load cached layer-31 features for the WHOLE corpus (combine old splits)
    Xs, sps = [], []
    for split in ["train", "valid", "test"]:
        d = np.load(FEAT / f"{split}.npz", allow_pickle=True)
        X = np.clip(np.nan_to_num(d["X"].astype(np.float32)), -50, 50)[:, 0, :]  # layer 31
        Xs.append(X); sps.extend([str(s) for s in d["speaker"]])
    X = np.concatenate(Xs, 0); sps = np.array(sps)
    print(f"combined features: {X.shape}", flush=True)

    # assign 40-country labels, drop unmapped
    lab = np.array([country40(s) for s in sps], dtype=object)
    keep = lab != None  # noqa: E711
    X, sps, lab = X[keep], sps[keep], lab[keep]
    classes = sorted(set(lab.tolist()))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab], np.int64)
    print(f"40-country: {len(classes)} classes, {len(y):,} utts kept "
          f"(dropped {int((~keep).sum()):,})", flush=True)

    # speaker-disjoint stratified split on the 40-country labels
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    idx = {s: np.where(split == s)[0] for s in ["train", "valid", "test"]}
    for s in ["train", "valid", "test"]:
        print(f"  {s}: {len(idx[s]):,} utts, {len(set(sps[idx[s]]))} spk, "
              f"{len(set(y[idx[s]].tolist()))} classes", flush=True)

    dev = "cuda"
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["valid"]], y[idx["valid"]]
    Xte, yte = X[idx["test"]], y[idx["test"]]
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    C = len(classes)
    torch.manual_seed(42)
    head = nn.Linear(1920, C).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev); n = len(Yt)

    def logits(Xn):
        head.eval()
        with torch.no_grad():
            return head(torch.from_numpy(Xn).to(dev)).cpu().numpy()

    def mami(L, yy):
        idx5 = np.argpartition(-L, 4, axis=1)[:, :5]
        hit5 = np.array([a in r for a, r in zip(yy, idx5)])
        top1 = (L.argmax(1) == yy)
        pres = sorted(set(yy.tolist()))
        return (top1.mean(), hit5.mean(),
                np.mean([top1[yy == c].mean() for c in pres]),
                np.mean([hit5[yy == c].mean() for c in pres]))

    best = -1; bw = None
    for ep in range(60):
        head.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            j = perm[i:i + 4096]
            F.cross_entropy(head(At[j]), Yt[j]).backward(); opt.step(); opt.zero_grad()
        _, _, _, vma = mami(logits(Xva), yva)
        if vma > best:
            best = vma; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(bw)
    tl = logits(Xte)
    t1, t5, m1, m5 = mami(tl, yte)
    print(f"\n=== 40-country linear probe (test) ===")
    print(f"  top1(micro)={t1:.4f}  top5(micro)={t5:.4f}  |  top1(macro)={m1:.4f}  top5(macro)={m5:.4f}")
    print(f"  (참고 61-모국어: micro top5 0.953 / macro 0.578)")

    # within-language confusion: English(미국/영국), Spanish(페루/콜롬비아/스페인/칠레)
    pred = tl.argmax(1)
    for grp in [["미국", "영국"], ["페루", "콜롬비아", "스페인", "칠레"]]:
        gids = [c2i[c] for c in grp if c in c2i]
        print(f"\n  [{'/'.join(grp)}] 내부 혼동 (행=정답, 열=예측%):")
        for tc in gids:
            m = yte == tc
            if m.sum() == 0:
                print(f"    {classes[tc]:6s}: test 없음"); continue
            row = {classes[pc]: f"{(pred[m]==pc).mean()*100:.0f}%" for pc in gids}
            other = 100 - sum((pred[m] == pc).mean() * 100 for pc in gids)
            print(f"    {classes[tc]:6s}(n={int(m.sum())}): " + ", ".join(f"{k} {v}" for k, v in row.items()) + f", 기타 {other:.0f}%")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""같은 시험지에 대한 모델 여러 개의 logits.npz 를 나란히 놓고, 화자 단위 부트스트랩으로 차이의 95% 구간을 낸다.

입력 logits.npz 는 finetune_full_encoder.py --eval_only 형식(ok · L · labels). 행 순서는 --test CSV 와 같아야 한다.
구간 = 화자를 복원추출(--n 회)해 다시 잰 차이의 2.5~97.5 백분위(같은 표본으로 두 모델을 잰다). 모든 모델이 읽은 행만 쓴다.

구간 종류
  행 평균      구간 안 발화 전체의 정확도
  국가 평균    국가마다 정확도를 내고 평균 — 작은 국가도 같은 무게
  학습 화자 수  --train 을 주면 웹마이크 문장을 국가의 학습 화자 수로 나눠 국가 평균을 낸다 (꼬리 ≤15 · 중간 16~100 · 머리 >100)
  Korea 오판   외국인 발화를 Korea 로 찍은 비율 — 낮을수록 좋다

    $PY scripts/eval/compare_logits_bootstrap.py --test <csv> [--train <train.csv>] \
        --models zeroth=<dir>/logits.npz full300m=<dir>/logits.npz full2b=<dir>/logits.npz --out <dir>/compare.md
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, csv
import numpy as np

KOREA = "Korea"
TIERS = (("꼬리 (학습 화자 ≤15)", 0, 15), ("중간 (16~100)", 16, 100), ("머리 (>100)", 101, 10 ** 9))
web = lambda r: r["domain"].startswith("web")
sent = lambda r: web(r) and r["task"] != "RW"

# (이름, 행 고르기, 종류) — 종류: row = 행 평균 · macro = 국가 평균 · korea = Korea 오판율
SEGMENTS = [
    ("웹마이크 전체", web, "row"),
    ("웹마이크 문장·자유발화", sent, "row"),
    ("웹마이크 단어", lambda r: web(r) and r["task"] == "RW", "row"),
    ("131 외국인", lambda r: r["domain"] == "p505", "row"),
    ("한국어 스튜디오(Kspon)", lambda r: r["domain"] == "kspon", "row"),
    ("한국어 증강", lambda r: r["domain"] == "kor_aug_test", "row"),
    ("Zeroth 원어민 낭독", lambda r: r["domain"] == "zeroth_test", "row"),
    ("웹마이크 문장 · 국가 평균", sent, "macro"),
    ("131 외국인 · 국가 평균", lambda r: r["domain"] == "p505", "macro"),
    ("외국인 → Korea 오판 (낮을수록 좋음)", lambda r: r["label_en"] != KOREA, "korea"),
]


def load(npz, rows):
    """→ (읽힌 행, top-1 맞음, top-5 맞음, top-1 이 Korea)"""
    d = np.load(npz, allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    if len(d["ok"]) != len(rows):
        raise SystemExit(f"{npz}: 행 {len(d['ok']):,} ≠ 시험지 {len(rows):,} — 다른 시험지로 잰 logits 다")
    c2i = {c: i for i, c in enumerate(labels)}
    y = np.array([c2i.get(r["label_en"], -1) for r in rows])
    top5 = np.argsort(-d["L"], 1)[:, :5]
    return (d["ok"].astype(bool) & (y >= 0), top5[:, 0] == y, (top5 == y[:, None]).any(1),
            top5[:, 0] == c2i.get(KOREA, -1))


def units(rows, idx):
    """화자 단위 묶음 → (묶음별 행 번호들, 묶음별 국가 번호)."""
    g = collections.OrderedDict()
    for i in idx:
        g.setdefault((rows[i]["speaker"], rows[i]["label_en"]), []).append(i)
    cls = {c: k for k, c in enumerate(sorted({c for _, c in g}))}
    return [np.array(v) for v in g.values()], np.array([cls[c] for _, c in g])


def value(groups, ucls, x, kind, w=None):
    """묶음 가중치 w(복원추출 횟수)로 잰 값 — row: 행 평균 · macro: 국가 평균."""
    s = np.array([x[g].sum() for g in groups], np.float64)
    n = np.array([len(g) for g in groups], np.float64)
    w = np.ones(len(groups)) if w is None else w
    if kind != "macro":
        return (w * s).sum() / (w * n).sum()
    cs = np.bincount(ucls, weights=w * s)
    cn = np.bincount(ucls, weights=w * n)
    m = cn > 0
    return (cs[m] / cn[m]).mean()


def boot(groups, ucls, a, b, kind, n, rng):
    """화자 복원추출 → (a − b) 분포의 (2.5, 97.5) 백분위."""
    sa = np.array([a[g].sum() for g in groups], np.float64)
    sb = np.array([b[g].sum() for g in groups], np.float64)
    cnt = np.array([len(g) for g in groups], np.float64)
    out = np.empty(n)
    for k in range(n):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)).astype(np.float64)
        if kind != "macro":
            out[k] = ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum()
        else:
            ca, cb, cn = (np.bincount(ucls, weights=w * v) for v in (sa, sb, cnt))
            m = cn > 0
            out[k] = (ca[m] / cn[m]).mean() - (cb[m] / cn[m]).mean()
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="이름=logits.npz. 첫째가 기준")
    ap.add_argument("--train", default="", help="학습 목록 — 주면 학습 화자 수 구간을 더한다")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.test, encoding="utf-8")))
    models = [m.split("=", 1) for m in a.models]
    H = {name: load(path, rows) for name, path in models}
    names = [n for n, _ in models]
    base = names[0]
    ok = np.ones(len(rows), bool)
    for n in names:
        ok &= H[n][0]

    segs = list(SEGMENTS)
    if a.train:
        spk = collections.defaultdict(set)
        for r in csv.DictReader(open(a.train, encoding="utf-8")):
            spk[r["label_en"]].add(r["speaker"])
        for title, lo, hi in TIERS:
            segs.append((f"웹마이크 문장 · {title} · 국가 평균",
                         lambda r, lo=lo, hi=hi: sent(r) and lo <= len(spk.get(r["label_en"], ())) <= hi, "macro"))

    rng = np.random.default_rng(a.seed)
    lines = [f"시험지 `{a.test}` · 화자 부트스트랩 {a.n}회 · 기준 = {base}", "",
             "| 구간 | 발화 · 화자 · 국가 | " + " | ".join(f"{n} top-1 / top-5" for n in names) + " |",
             "|---|--:|" + "---|" * len(names)]
    diffs = []
    for seg, f, kind in segs:
        idx = [i for i in np.flatnonzero(ok) if f(rows[i])]
        if not idx:
            continue
        groups, ucls = units(rows, idx)
        size = f"{len(idx):,} · {len(groups):,} · {ucls.max() + 1}"
        metrics = ((3, "비율"),) if kind == "korea" else ((1, "top-1"), (2, "top-5"))
        cells = [" / ".join(f"{value(groups, ucls, H[n][k], kind):.3f}" for k, _ in metrics) for n in names]
        lines.append(f"| {seg} | {size} | " + " | ".join(cells) + " |")
        for n in names[1:]:
            for k, mname in metrics:
                lo, hi = boot(groups, ucls, H[n][k], H[base][k], kind, a.n, rng)
                d = value(groups, ucls, H[n][k], kind) - value(groups, ucls, H[base][k], kind)
                diffs.append(f"| {seg} | {n} − {base} {mname} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "| 구간 | 차이 | 값 [95% 구간] |", "|---|---|---|"] + diffs
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()

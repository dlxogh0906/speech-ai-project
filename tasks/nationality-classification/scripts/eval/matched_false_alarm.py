# -*- coding: utf-8 -*-
"""같은 오경보율에서 집단(꼬리 · 중간) 국가의 top-5 를 견준다 — 집단 성적이 오른 게 그 국가를 더 자주 찍어서인지 본다 (2026-10-20).

왜: 행 균등 · 오버샘플링 · 사후 보정은 작은 국가의 점수를 통째로 올린다. 그러면 그 국가 발화의 top-5 적중도 오르지만
    다른 국가 발화의 top-5 에도 그 국가가 더 자주 낀다(오경보). 두 모델을 같은 오경보율에 놓고 적중을 견줘야
    "더 잘 가려낸다"와 "더 자주 찍는다"가 갈린다.

방법 (웹마이크 문장 · 자유발화만, 단어 제외 — compare_logits_bootstrap.py 의 계층 구간과 같은 행)
  - 집단 = --train 의 학습 화자 수가 (lo, hi] 인 국가 (--group 0,15 = 꼬리 · 15,100 = 중간)
  - 오경보 = 집단 밖 국가 발화의 top-5 에 집단 국가가 하나라도 낀 비율
  - 적중 = 집단 국가 발화의 top-5 에 정답이 있는 비율, 국가마다 내서 평균(국가 평균)
  - 모델마다 집단 국가 로짓에 같은 b 를 더해 오경보를 --rates 에 맞춘다(오경보는 b 에 대해 단조라 이분 탐색).
  - --ref 를 주면 그 모델 대비 차이의 95% 구간을 낸다: 집단 발화의 화자를 복원추출(--n 회). b 는 전체 표본에서 한 번 맞추고 고정한다
    (b 를 정하는 불확실성은 빠진 조건부 구간이다).

    $PY scripts/eval/matched_false_alarm.py --test <test.csv> --train <train.csv> --group 0,15 --rates 0.1,0.2 \
        --ref 대조군 --models 대조군=<dir>/logits.npz 새모델=<dir>/logits.npz
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


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class Model:
    """한 모델의 웹 문장 로짓 + 집단 표시. at(b) → (오경보, 집단 발화별 적중, 비집단 top-1)."""

    def __init__(self, path, te, spk, lo, hi):
        z = np.load(path, allow_pickle=True)
        labels = [str(c) for c in z["labels"]]
        idx = {c: i for i, c in enumerate(labels)}
        y = np.array([idx.get(r["label_en"], -1) for r in te])
        sent = np.array([r["domain"].startswith("web") and r["task"] != "RW" for r in te]) & (y >= 0) & z["ok"].astype(bool)
        self.g = np.array([lo < len(spk[c]) <= hi for c in labels])
        self.labels = labels
        L = z["L"].astype(np.float64)
        ing = self.g[np.maximum(y, 0)]
        self.Ln, self.yn = L[sent & ~ing], y[sent & ~ing]
        self.Lg, self.yg = L[sent & ing], y[sent & ing]
        self.rows_g = np.where(sent & ing)[0]

    def at(self, b):
        off = b * self.g[None, :]
        tn = np.argpartition(-(self.Ln + off), 5, 1)[:, :5]
        tg = np.argpartition(-(self.Lg + off), 5, 1)[:, :5]
        fa = self.g[tn].any(1).mean()
        hit = (tg == self.yg[:, None]).any(1).astype(float)
        top1 = ((self.Ln + off).argmax(1) == self.yn).mean()
        return fa, hit, top1

    def solve(self, rate, lo=-30.0, hi=30.0, it=40):
        """오경보가 rate 에 가장 가까운 b (오경보는 b 에 대해 단조 증가)."""
        for _ in range(it):
            mid = (lo + hi) / 2
            if self.at(mid)[0] < rate:
                lo = mid
            else:
                hi = mid
        return min((lo, hi), key=lambda b: abs(self.at(b)[0] - rate))


def country_avg(hit, cls, sel=None):
    if sel is not None:
        hit, cls = hit[sel], cls[sel]
    return float(np.mean([hit[cls == k].mean() for k in np.unique(cls)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True, help="학습 목록 — 국가별 학습 화자 수로 집단을 정한다")
    ap.add_argument("--group", default="0,15", help="학습 화자 수 구간 (lo, hi] — 0,15 꼬리 · 15,100 중간")
    ap.add_argument("--rates", default="0.10,0.15,0.20,0.25", help="맞출 오경보율 (쉼표)")
    ap.add_argument("--ref", default="", help="이 이름의 모델 대비 차이 · 95%% 구간을 낸다")
    ap.add_argument("--n", type=int, default=1000, help="부트스트랩 횟수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True, help="이름=logits.npz ...")
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.group.split(","))
    rates = [float(x) for x in a.rates.split(",")]
    te = load_rows(a.test)
    spk = collections.defaultdict(set)
    for r in load_rows(a.train):
        spk[r["label_en"]].add(r["speaker"])
    M = collections.OrderedDict((n, Model(p, te, spk, lo, hi)) for n, p in (m.split("=", 1) for m in a.models))
    first = next(iter(M.values()))
    for n, m in M.items():
        if m.labels != first.labels or not np.array_equal(m.rows_g, first.rows_g):
            raise SystemExit(f"{n}: 라벨 또는 집단 행이 첫 모델과 다르다 — 같은 시험지 · 같은 라벨의 로짓만 견줄 수 있다")
    cls = first.yg
    gspk = np.array([te[i]["speaker"] for i in first.rows_g])
    spks = sorted(set(gspk))
    print(f"집단: 학습 화자 ({lo}, {hi}] · {len(set(cls))}개국 · test {len(spks)}명 · {len(cls):,}발화 (웹 문장) · "
          f"비집단 {len(first.yn):,}발화")
    print()
    print("| 모델 | 원래 오경보 | 원래 집단 top-5 (국가 평균) | "
          + " | ".join(f"오경보 {r:.0%} → 집단 top-5 · 비집단 top-1" for r in rates) + " |")
    print("|---|--:|--:|" + "--:|" * len(rates))
    hits = {}
    for n, m in M.items():
        fa0, h0, _ = m.at(0.0)
        cells = []
        for r in rates:
            b = m.solve(r)
            fa, h, t1 = m.at(b)
            hits[(n, r)] = (h, fa)
            cells.append(f"{country_avg(h, cls):.3f} · {t1:.3f}")
        print(f"| {n} | {fa0:.1%} | {country_avg(h0, cls):.3f} | " + " | ".join(cells) + " |")
    if not a.ref:
        return
    if a.ref not in M:
        raise SystemExit(f"--ref {a.ref!r} 가 --models 에 없다")
    pos = {s: np.where(gspk == s)[0] for s in spks}
    rng = np.random.default_rng(a.seed)
    boots = [np.concatenate([pos[s] for s in rng.choice(spks, len(spks))]) for _ in range(a.n)]
    print()
    print(f"차이 = 모델 − {a.ref} (같은 오경보율) · 집단 화자 부트스트랩 {a.n}회 · b 고정")
    print()
    print("| 모델 | " + " | ".join(f"오경보 {r:.0%}" for r in rates) + " |")
    print("|---|" + "---|" * len(rates))
    for n in M:
        if n == a.ref:
            continue
        cells = []
        for r in rates:
            h, hr = hits[(n, r)][0], hits[(a.ref, r)][0]
            d = np.array([country_avg(h, cls, s) - country_avg(hr, cls, s) for s in boots])
            cells.append(f"{country_avg(h, cls) - country_avg(hr, cls):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        print(f"| {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

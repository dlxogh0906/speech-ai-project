# -*- coding: utf-8 -*-
"""앙상블 여지 — 이미 있는 test 로짓을 섞어 기준 모델과 짝지어 비교한다 (2026-10-26).

모델마다 log-softmax 를 구해 같은 가중치로 평균한다(가중치를 test 로 고르지 않는다). 잣대는 웹마이크 문장 · 자유발화.
  - 발화 1개 top-1 · top-5, 국가 평균 top-1 · top-5, 머리 · 중간 · 꼬리 국가 평균 top-5
  - 같은 화자 3문장 합산 top-1 (모든 모델이 같은 문장 묶음을 쓴다)
  - 웹 단어 top-1, 한국인 원어민 top-1, 외국인 → 한국 비율
  - 기준과의 차이: 화자 복원추출(기본 1000회) 95% 구간, 같은 발화 · 같은 화자끼리 짝지어 잰다

    $PY scripts/eval/ensemble_logits_report.py --train <train.csv> --test <test.csv> \
        --models 300m=<a/logits.npz> 2B=<b/logits.npz> --combos "300m+2B" --base 300m --out <표.md>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, csv, json
import numpy as np


def log_softmax(L):
    z = L - L.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def load(p, n, labels=None):
    d = np.load(p, allow_pickle=True)
    if len(d["ok"]) != n:
        raise SystemExit(f"{p}: 행 {len(d['ok']):,} ≠ 시험지 {n:,}")
    lab = [str(x) for x in d["labels"]]
    if labels is not None and lab != labels:
        raise SystemExit(f"{p}: 라벨 순서가 다르다")
    return lab, log_softmax(d["L"].astype(np.float64)), d["ok"].astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="이름=logits.npz")
    ap.add_argument("--combos", nargs="+", required=True, help='"이름+이름" — 같은 가중치 평균')
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    with open(a.train, encoding="utf-8") as f:
        tr_spk = collections.defaultdict(set)
        for r in csv.DictReader(f):
            tr_spk[r["label_en"]].add(r["speaker"])
    with open(a.test, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    labels, LS, okall = None, {}, np.ones(n, bool)
    for spec in a.models:
        name, p = spec.split("=", 1)
        labels, LS[name], ok = load(p, n, labels)
        okall &= ok
    ci = {c: i for i, c in enumerate(labels)}
    y = np.array([ci[r["label_en"]] for r in rows])
    spk = np.array([r["speaker"] for r in rows]); dom = np.array([r["domain"] for r in rows]); task = np.array([r["task"] for r in rows])
    web = np.array([d.startswith("web") for d in dom]) & okall
    sent = web & (task != "RW"); word = web & (task == "RW")
    kk = ci["Korea"]; kor = (y == kk) & okall; foreign = (y != kk) & okall

    def tier(c):
        m = len(tr_spk[c])
        return "꼬리" if m <= 15 else ("중간" if m <= 100 else "머리")

    idx = np.where(sent)[0]
    by_spk = collections.OrderedDict()
    for i in idx:
        by_spk.setdefault(spk[i], []).append(i)
    spk_list = list(by_spk)
    groups = [np.array(by_spk[s]) for s in spk_list]
    sets3 = [[g] if len(g) == 3 else [rng.choice(g, 3, replace=False) for _ in range(a.draws)] for g in groups if len(g) >= 3]
    g3_spk = [s for s, g in zip(spk_list, groups) if len(g) >= 3]
    countries = sorted({labels[y[i]] for i in idx})

    combos = {}
    for spec in [m.split("=", 1)[0] for m in a.models] + a.combos:
        parts = spec.split("+")
        combos[spec] = sum(LS[p] for p in parts) / len(parts)

    def per_utt(S):
        rk = (S > S[np.arange(n), y][:, None]).sum(1)
        return rk == 0, rk < 5

    def spk3(S):
        out = []
        for sets in sets3:
            t = 0.0
            for s in sets:
                m = S[s].mean(0); t += float((m > m[y[s[0]]]).sum() == 0)
            out.append(t / len(sets))
        return np.array(out)

    res = {}
    for name, S in combos.items():
        t1, t5 = per_utt(S)
        c_t1 = {c: t1[sent & (y == ci[c])].mean() for c in countries}
        c_t5 = {c: t5[sent & (y == ci[c])].mean() for c in countries}
        res[name] = {"t1": t1, "t5": t5, "s3": spk3(S),
                     "top1": t1[sent].mean(), "top5": t5[sent].mean(),
                     "cm1": np.mean(list(c_t1.values())), "cm5": np.mean(list(c_t5.values())),
                     "tier5": {t: np.mean([c_t5[c] for c in countries if tier(c) == t]) for t in ("머리", "중간", "꼬리")},
                     "word1": t1[word].mean(), "kor": (S[kor].argmax(1) == kk).mean(), "f2k": (S[foreign].argmax(1) == kk).mean()}
        res[name]["spk3"] = res[name]["s3"].mean()

    # 짝지은 차이 — 화자 복원추출
    sg = [np.array(by_spk[s]) for s in spk_list]
    g3_pos = {s: j for j, s in enumerate(g3_spk)}
    base = res[a.base]
    boot_idx = [rng.integers(0, len(sg), len(sg)) for _ in range(a.n_boot)]

    def diff_ci(xa, xb):
        da = np.array([xa[g].sum() - xb[g].sum() for g in sg], np.float64); cnt = np.array([len(g) for g in sg], np.float64)
        bs = [da[k].sum() / cnt[k].sum() for k in boot_idx]
        return da.sum() / cnt.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def diff_ci3(xa, xb):
        d = xa - xb
        pos = [g3_pos.get(s) for s in spk_list]
        bs = []
        for k in boot_idx:
            v = [d[pos[j]] for j in k if pos[j] is not None]
            bs.append(np.mean(v))
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    fmt = lambda v: f"{v:.3f}"
    sd = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    L = [f"# 앙상블 — 같은 가중치 log-softmax 평균 vs {a.base}", "",
         f"웹마이크 문장 · 자유발화 {int(sent.sum()):,}발화 · {len(sg):,}명 · {len(countries)}개국. 구간은 화자 복원추출 {a.n_boot}회 95%.", "",
         "| 모델 | 모델 수 | 발화 top-1 | 발화 top-5 | 3문장 top-1 | 국가 평균 top-1 | 국가 평균 top-5 | 머리 · 중간 · 꼬리 top-5 | 단어 top-1 | 한국인 → 한국 | 외국인 → 한국 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, r in res.items():
        L.append(f"| {name} | {len(name.split('+'))} | {fmt(r['top1'])} | {fmt(r['top5'])} | {fmt(r['spk3'])} | {fmt(r['cm1'])} | {fmt(r['cm5'])} | "
                 f"{fmt(r['tier5']['머리'])} · {fmt(r['tier5']['중간'])} · {fmt(r['tier5']['꼬리'])} | {fmt(r['word1'])} | {fmt(r['kor'])} | {r['f2k']:.4f} |")
    L += ["", f"## {a.base} 대비 차이", "", "| 모델 | 발화 top-1 [95%] | 발화 top-5 [95%] | 3문장 top-1 [95%] |", "|---|--:|--:|--:|"]
    for name, r in res.items():
        if name == a.base:
            continue
        L.append(f"| {name} | {sd(diff_ci(r['t1'], base['t1']))} | {sd(diff_ci(r['t5'], base['t5']))} | {sd(diff_ci3(r['s3'], base['s3']))} |")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

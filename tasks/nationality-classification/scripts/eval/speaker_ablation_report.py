# -*- coding: utf-8 -*-
"""화자 수 실험 표 — 학습 화자를 줄인 나라마다 대조군(전부 학습)과 test 를 비교한다 (2026-10-22).

make_speaker_ablation_manifest.py 로 만든 목록으로 학습한 모델(--runs)을 대조군(--ref)과 같은 시험지로 잰다.
  - 잣대: 웹마이크 문장·자유발화(단어 RW 제외)에서 그 나라 발화의 top-1 · top-5
  - 같은 오경보율 top-5: 그 나라 로짓에 상수를 더해, 다른 나라 웹 문장 발화의 top-5 에 그 나라가 끼는 비율(오경보)을
    대조군의 오경보에 맞춘 뒤 잰 top-5. 줄인 나라를 덜 찍게 된 몫을 빼고 가려내는 힘만 본다(상수는 고정한 조건부 구간)
  - 구간: 그 나라 test 화자를 복원추출(기본 1000회)한 95% 백분위. 두 모델의 차이는 같은 발화끼리 짝지어 잰다
  - 같은 나라를 두 학습에서 다른 조건으로 줄였으면 두 조건의 차이도 낸다(같은 test 화자 · 작은 조건 화자 ⊂ 큰 조건)
  - 안 바꾼 나라들의 국가 평균 top-5 로 "학습 목록이 20% 줄어든 효과"를 따로 본다

    $PY scripts/eval/speaker_ablation_report.py --test <test_koraug_zeroth.csv> --ref 대조군=<logits.npz> \
        --runs A=<logits.npz>=<abl_meta.json> B=<logits.npz>=<abl_meta.json> --out <표.md>
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

KO = {"Germany": "독일", "France": "프랑스", "Italy": "이탈리아", "Romania": "루마니아", "Japan": "일본",
      "Vietnam": "베트남", "Mongolia": "몽골", "Turkey": "튀르키예"}
web = lambda r: r["domain"].startswith("web")
sent = lambda r: web(r) and r["task"] != "RW"


def load(npz, rows):
    d = np.load(npz, allow_pickle=True)
    if len(d["ok"]) != len(rows):
        raise SystemExit(f"{npz}: 행 {len(d['ok']):,} ≠ 시험지 {len(rows):,}")
    return [str(x) for x in d["labels"]], d["L"].astype(np.float64), d["ok"].astype(bool)


def margin(L, c):
    """나라 c 의 로짓 − 다른 나라 중 5번째로 큰 로짓. > 0 이면 c 가 top-5 안이다."""
    o = L.copy(); o[:, c] = -np.inf
    return L[:, c] - np.partition(o, -5, axis=1)[:, -5]


def groups_of(rows, idx):
    g = collections.OrderedDict()
    for i in idx:
        g.setdefault(rows[i]["speaker"], []).append(i)
    return [np.array(v) for v in g.values()]


def rate(groups, x, w=None):
    s = np.array([x[g].sum() for g in groups], np.float64); n = np.array([len(g) for g in groups], np.float64)
    w = np.ones(len(groups)) if w is None else w
    return (w * s).sum() / (w * n).sum()


def boot_diff(groups, a, b, n, rng):
    """화자 복원추출 → mean(a) − mean(b) 의 (2.5, 97.5) 백분위."""
    sa = np.array([a[g].sum() for g in groups], np.float64); sb = np.array([b[g].sum() for g in groups], np.float64)
    cnt = np.array([len(g) for g in groups], np.float64)
    out = np.empty(n)
    for k in range(n):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)).astype(np.float64)
        out[k] = ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum()
    return np.percentile(out, [2.5, 97.5])


def boot_mean_diff(parts, n, rng):
    """[(groups, a, b), ...] 나라 여럿 — 나라마다 따로 화자를 복원추출해 차이의 평균 → 95% 백분위."""
    out = np.zeros(n)
    for groups, a, b in parts:
        sa = np.array([a[g].sum() for g in groups], np.float64); sb = np.array([b[g].sum() for g in groups], np.float64)
        cnt = np.array([len(g) for g in groups], np.float64)
        for k in range(n):
            w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)).astype(np.float64)
            out[k] += ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum() / len(parts)
    return np.percentile(out, [2.5, 97.5])


def boot_macro_diff(rows, idx, a, b, n, rng):
    """(화자, 나라) 묶음 복원추출 → 국가 평균 차이의 95% 백분위."""
    g = collections.OrderedDict()
    for i in idx:
        g.setdefault((rows[i]["speaker"], rows[i]["label_en"]), []).append(i)
    groups = [np.array(v) for v in g.values()]
    cls = {c: k for k, c in enumerate(sorted({c for _, c in g}))}
    uc = np.array([cls[c] for _, c in g])
    sa = np.array([a[x].sum() for x in groups], np.float64); sb = np.array([b[x].sum() for x in groups], np.float64)
    cnt = np.array([len(x) for x in groups], np.float64)

    def macro(w, s):
        cs, cn = np.bincount(uc, weights=w * s), np.bincount(uc, weights=w * cnt)
        m = cn > 0
        return (cs[m] / cn[m]).mean()
    one = np.ones(len(groups))
    out = np.empty(n)
    for k in range(n):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)).astype(np.float64)
        out[k] = macro(w, sa) - macro(w, sb)
    return macro(one, sa), macro(one, sb), np.percentile(out, [2.5, 97.5])


def cond_label(cond):
    if "x" in cond:
        n, k = cond.split("x")
        return f"{n}명 × {k}문장"
    return f"{cond}명 × 녹음 전부"


def fmt_d(d, ci):
    return f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]".replace("-", "−")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref", required=True, help="이름=logits.npz (대조군 — 전부 학습)")
    ap.add_argument("--runs", nargs="+", required=True, help="꼬리표=logits.npz=abl_meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.test, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ref_name, ref_path = a.ref.split("=", 1)
    labels, Lref, okr = load(ref_path, rows)
    c2i = {c: i for i, c in enumerate(labels)}
    runs = []
    for spec in a.runs:
        tag, lg, meta = spec.split("=", 2)
        lab, L, ok = load(lg, rows)
        if lab != labels:
            raise SystemExit(f"{lg}: 클래스 순서가 대조군과 다르다")
        runs.append((tag, L, ok, json.load(open(meta, encoding="utf-8"))))
    ok_all = okr.copy()
    for _, _, ok, _ in runs:
        ok_all &= ok
    y = np.array([c2i.get(r["label_en"], -1) for r in rows])
    S = np.array([sent(r) and ok_all[i] and y[i] >= 0 for i, r in enumerate(rows)])
    rng = np.random.default_rng(a.seed)

    def hits(L):
        t5 = np.argsort(-L, 1)[:, :5]
        return (t5[:, 0] == y).astype(np.float64), (t5 == y[:, None]).any(1).astype(np.float64)
    H = {ref_name: hits(Lref)}
    for tag, L, _, _ in runs:
        H[tag] = hits(L)
    Lof = {ref_name: Lref}
    Lof.update({tag: L for tag, L, _, _ in runs})

    changed = collections.OrderedDict()                      # 나라 → [(꼬리표, 조건, 메타)]
    for tag, _, _, meta in runs:
        for c, v in meta["per_class"].items():
            changed.setdefault(c, []).append((tag, v["cond"], v))

    res = {}                                                 # (나라, 꼬리표) → 값들
    lines = [f"# 화자 수 실험 — 학습 화자를 줄인 나라 vs 대조군({ref_name})", "",
             "웹마이크 문장·자유발화 기준. 구간은 그 나라 test 화자 복원추출 95%. "
             "같은 오경보율 = 그 나라를 다른 나라 발화의 top-5 에 끼우는 비율을 대조군과 같게 맞춘 뒤의 top-5.", ""]
    lines += ["## 1. 나라별", "",
              "| 나라 | 모델 | 학습 화자 · 행 | test 화자 · 발화 | top-1 | top-5 | top-5 − 대조군 [95%] | 오경보 (원래) "
              "| 같은 오경보율 top-5 | − 대조군 [95%] |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, items in changed.items():
        ci = c2i[c]
        idx_c = np.where(S & (y == ci))[0]; idx_n = np.where(S & (y != ci))[0]
        groups = groups_of(rows, idx_c)
        m_ref = margin(Lref, ci)
        fa_ref = float((m_ref[idx_n] > 0).mean())
        in_ref = (m_ref > 0).astype(np.float64)
        v0 = items[0][2]
        lines.append(f"| {KO.get(c, c)} | 대조군 (전부) | {v0['spk_src']:,} · {v0['rows_src']:,} | {len(groups)} · {len(idx_c):,} "
                     f"| {rate(groups, H[ref_name][0]):.3f} | {rate(groups, H[ref_name][1]):.3f} | — | {fa_ref:.1%} "
                     f"| {rate(groups, in_ref):.3f} | — |")
        for tag, cond, v in sorted(items, key=lambda t: (int(t[1].split('x')[0]), 'x' not in t[1])):
            L = Lof[tag]
            m = margin(L, ci)
            fa = float((m[idx_n] > 0).mean())
            thr = np.quantile(m[idx_n], 1 - fa_ref)
            matched = (m > thr).astype(np.float64)
            t1, t5 = rate(groups, H[tag][0]), rate(groups, H[tag][1])
            d5 = t5 - rate(groups, H[ref_name][1]); ci5 = boot_diff(groups, H[tag][1], H[ref_name][1], a.n, rng)
            dm = rate(groups, matched) - rate(groups, in_ref); cim = boot_diff(groups, matched, in_ref, a.n, rng)
            res[(c, tag)] = dict(cond=cond, groups=groups, t5=H[tag][1], matched=matched, ref5=H[ref_name][1], ref_m=in_ref)
            lines.append(f"| | {tag} · {cond_label(cond)} | {v['spk_dst']} · {v['rows_dst']:,} | | {t1:.3f} | {t5:.3f} "
                         f"| {fmt_d(d5, ci5)} | {fa:.1%} | {rate(groups, matched):.3f} | {fmt_d(dm, cim)} |")
    lines.append("")

    lines += ["## 2. 같은 나라 안에서 두 조건 — 큰 조건 − 작은 조건", "",
              "| 나라 | 비교 | top-5 차이 [95%] | 같은 오경보율 top-5 차이 [95%] |", "|---|---|--:|--:|"]
    for c, items in changed.items():
        if len(items) < 2:
            continue
        key = lambda t: (int(t[1].split('x')[0]), 'x' not in t[1])
        (ts, cs, _), (tb, cb, _) = sorted(items, key=key)[:2]
        rs, rb = res[(c, ts)], res[(c, tb)]
        g = rs["groups"]
        d5 = rate(g, rb["t5"]) - rate(g, rs["t5"]); ci5 = boot_diff(g, rb["t5"], rs["t5"], a.n, rng)
        dm = rate(g, rb["matched"]) - rate(g, rs["matched"]); cim = boot_diff(g, rb["matched"], rs["matched"], a.n, rng)
        lines.append(f"| {KO.get(c, c)} | {cond_label(cb)} − {cond_label(cs)} | {fmt_d(d5, ci5)} | {fmt_d(dm, cim)} |")
    lines.append("")

    by_cond = collections.OrderedDict()
    for (c, tag), r in sorted(res.items(), key=lambda kv: (int(kv[1]["cond"].split('x')[0]), 'x' not in kv[1]["cond"])):
        by_cond.setdefault(r["cond"], []).append((c, r))
    lines += ["## 3. 조건별 — 나라 평균", "",
              "| 조건 | 나라 | top-5 | − 대조군 [95%] | 같은 오경보율 top-5 | − 대조군 [95%] | 대조군 top-5 |",
              "|---|---|--:|--:|--:|--:|--:|"]
    for cond, lst in by_cond.items():
        t5 = np.mean([rate(r["groups"], r["t5"]) for _, r in lst])
        r5 = np.mean([rate(r["groups"], r["ref5"]) for _, r in lst])
        tm = np.mean([rate(r["groups"], r["matched"]) for _, r in lst])
        rm = np.mean([rate(r["groups"], r["ref_m"]) for _, r in lst])
        ci5 = boot_mean_diff([(r["groups"], r["t5"], r["ref5"]) for _, r in lst], a.n, rng)
        cim = boot_mean_diff([(r["groups"], r["matched"], r["ref_m"]) for _, r in lst], a.n, rng)
        lines.append(f"| {cond_label(cond)} | {' · '.join(KO.get(c, c) for c, _ in lst)} | {t5:.3f} | {fmt_d(t5 - r5, ci5)} "
                     f"| {tm:.3f} | {fmt_d(tm - rm, cim)} | {r5:.3f} |")
    lines.append("")

    lines += ["## 4. 안 바꾼 나라 · 전체 — 학습 목록이 줄어든 효과", "",
              "| 모델 | 안 바꾼 나라 국가 평균 top-5 (웹 문장) | − 대조군 [95%] | 웹 전체 top-5 | Zeroth 원어민 top-1 |",
              "|---|--:|--:|--:|--:|"]
    ch = {c2i[c] for c in changed}
    idx_u = np.where(S & ~np.isin(y, list(ch)))[0]
    W = np.array([web(r) and ok_all[i] and y[i] >= 0 for i, r in enumerate(rows)])
    Z = np.array([r["domain"] == "zeroth_test" and ok_all[i] and y[i] >= 0 for i, r in enumerate(rows)])
    for name in [ref_name] + [t for t, _, _, _ in runs]:
        mu, mr, ci = boot_macro_diff(rows, idx_u, H[name][1], H[ref_name][1], a.n, rng)
        d = "—" if name == ref_name else fmt_d(mu - mr, ci)
        lines.append(f"| {name} | {mu:.3f} | {d} | {H[name][1][W].mean():.3f} | {H[name][0][Z].mean():.3f} |")
    lines.append("")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

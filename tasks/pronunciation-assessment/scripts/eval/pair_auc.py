# -*- coding: utf-8 -*-
"""세기 쌍 **구별력**(AUC) 비교 — 모델마다 `eval_pair_rescue.py extract` 결과(같은 발화)를 읽어, 쌍(c→a)마다
"사람이 a 로 들은 자리"와 "c 로 들은 자리"를 LLR(표준 c 를 a 로 바꾼 CTC 로그우도 차)로 얼마나 잘 가르나를 잰다.

재현율·헛지적은 판정 문턱에 따라 서로 바뀐다(1차 합성 학습의 개선은 문턱만 옮긴 것과 같았다 — 2026-10-11 리뷰).
AUC 는 문턱과 무관하므로 "듣는 힘"이 늘었는지는 이걸로 본다.

점수 = 그리디 받아쓰기가 이미 a 로 적었으면 맨 위, 표준 c 로 적었으면 LLR, 그 밖(제3의 소리·탈락)은 맨 아래
  — "받아쓰기 + LLR 문턱"으로 a 를 지적하는 판정의 ROC. (추출은 c 로 적은 자리에서만 LLR 을 잰다.)
  '|표준으로 적은 자리만' = 모든 모델이 c 로 적은 같은 자리끼리 LLR 만으로 잰 AUC(구조 판정 몫은 빠진다 — 참고용).
범위
  onset_lenis  표준이 평음(ㄱㄷㅂㅈ)인 첫소리 자리(발화 첫 자리 · 모음 사이 · 공명음 뒤) — 힌트·합성 학습이 겨냥한 범위
  all          모든 쌍 · 모든 자리(받침 포함)
가중 평균 = 쌍마다 양성(오류) 자리 수로 가중. 구간 = 문항(파일 이름의 문항 코드) 단위 재표집 1000번, 모델 차이는 짝지어(같은 표본).

    $PY .../pair_auc.py base=$MDD_OUTPUTS/analysis/pair_rescue_2026-10-10 v1=$MDD_OUTPUTS/analysis/pair_rescue_ft3000 \\
                        v2=$MDD_OUTPUTS/analysis/pair_rescue_v2_3000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, glob, json, os, random, re

ONSET = ("initial", "intervocalic", "post_sonorant")
LENIS = "ㄱㄷㅂㅈ"


def wauc(inv, nb, is_pos, w):
    """가중 AUC(동점 절반) — inv: 값의 순위 칸(np.unique 역색인), nb: 칸 수, w: 자리 가중(재표집 배수)."""
    import numpy as np
    pw = np.bincount(inv[is_pos], weights=w[is_pos], minlength=nb)
    nw = np.bincount(inv[~is_pos], weights=w[~is_pos], minlength=nb)
    P, N = pw.sum(), nw.sum()
    if P <= 0 or N <= 0:
        return None, 0.0
    below = np.cumsum(nw) - nw
    return float((pw * (below + 0.5 * nw)).sum() / (P * N)), float(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="이름=extract 폴더")
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-pos", dest="min_pos", type=int, default=30, help="양성 자리가 이보다 적은 쌍은 평균에서 뺀다")
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from mdd.pair_rescue import FAMILY_OF, position

    can = [l.split() for l in open(os.path.join(a.manifest, a.split + ".canon"), encoding="utf-8").read().splitlines()]
    item = []
    with open(os.path.join(a.manifest, a.split + ".tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            m = re.search(r"-(R[WSP]\d{3})-", line)
            item.append(m.group(1) if m else "?")
    names, data = [], {}
    for spec in a.models:
        name, d = spec.split("=", 1)
        U = {}
        for p in sorted(glob.glob(os.path.join(d, "%s.*-of-*.jsonl" % a.split))):
            for l in open(p, encoding="utf-8"):
                u = json.loads(l); U[u["u"]] = u
        meta = os.path.join(d, "extract_meta.json")
        names.append(name)
        data[name] = (U, json.load(open(meta, encoding="utf-8")).get("ckpt") if os.path.exists(meta) else None)
    common = set.intersection(*[set(U) for U, _ in data.values()])
    print("[발화] 모델 %s · 공통 %d" % (" / ".join("%s %d" % (n, len(data[n][0])) for n in names), len(common)), flush=True)

    # 자리 목록: (문항, 범위표시, 쌍, 양성?, 모델별 LLR)
    import numpy as np
    it_idx = {}
    rows = collections.defaultdict(lambda: {"g": [], "y": [], "v": [[] for _ in names]})
    for uid in sorted(common):
        cg = can[uid]
        per = [dict(((j, c), (t, pm, llr)) for j, c, t, pm, llr in data[n][0][uid]["sites"]) for n in names]
        gi = it_idx.setdefault(item[uid], len(it_idx))
        for (j, c), (t, _pm0, _llr0) in per[0].items():
            if t is None or c not in FAMILY_OF:
                continue
            onset_lenis = c in LENIS and position(cg, j) in ONSET
            for alt in FAMILY_OF[c]:
                if alt == c or (t != c and t != alt):
                    continue
                vals, cond = [], []
                for pm_ in per:
                    _t, pm, llr = pm_.get((j, c), (None, None, {}))
                    v = llr.get(alt)
                    ok = v is not None and v == v
                    # 합친 점수: 그리디가 이미 a 로 적었으면 맨 위, 표준 c 로 적었으면 LLR, 그 밖(제3의 소리·탈락)은 맨 아래
                    vals.append(1e9 if pm == alt else (v if (pm == c and ok) else -1e9))
                    cond.append(v if (pm == c and ok) else None)
                for scope in (("all", "onset_lenis") if onset_lenis else ("all",)):
                    r = rows[(scope, c + alt)]
                    r["g"].append(gi); r["y"].append(t == alt)
                    for k in range(len(names)):
                        r["v"][k].append(vals[k])
                    r2 = rows[(scope + "|표준으로 적은 자리만", c + alt)]
                    if all(x is not None for x in cond):          # 모든 모델이 c 로 적은 자리만(같은 자리끼리)
                        r2["g"].append(gi); r2["y"].append(t == alt)
                        for k in range(len(names)):
                            r2["v"][k].append(cond[k])
    prep = {}
    for key, r in rows.items():
        y = np.array(r["y"], bool)
        if y.sum() < a.min_pos or (~y).sum() == 0:
            continue
        g = np.array(r["g"])
        cols = []
        for k in range(len(names)):
            u, inv = np.unique(np.array(r["v"][k]), return_inverse=True)
            cols.append((inv, len(u)))
        prep[key] = (g, y, cols)
    n_items = len(it_idx)

    def table(m):
        """m: 문항별 배수 → {범위: {모델: 가중 AUC}}, {(범위, 쌍): {모델: AUC}}."""
        out, per_pair = {}, {}
        for scope in ("onset_lenis", "all", "onset_lenis|표준으로 적은 자리만", "all|표준으로 적은 자리만"):
            res = {}
            for k, n in enumerate(names):
                num = den = 0.0
                for (sc, pr), (g, y, cols) in prep.items():
                    if sc != scope:
                        continue
                    val, P = wauc(cols[k][0], cols[k][1], y, m[g])
                    if val is None:
                        continue
                    num += val * P; den += P
                    per_pair.setdefault((sc, pr), {})[n] = val
                res[n] = num / den if den else None
            out[scope] = res
        return out, per_pair

    main_t, per_pair = table(np.ones(n_items))
    rng = np.random.RandomState(a.seed)
    boots = collections.defaultdict(list)
    for _ in range(a.n_boot):
        m = np.bincount(rng.randint(0, n_items, n_items), minlength=n_items).astype(float)
        t, _pp = table(m)
        for scope, res in t.items():
            for n in names[1:]:
                if res[n] is not None and res[names[0]] is not None:
                    boots[(scope, n)].append(res[n] - res[names[0]])
    ci = {}
    for (scope, n), v in boots.items():
        v.sort()
        ci["%s|%s-%s" % (scope, n, names[0])] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    detail = {}
    for (sc, pr), d in per_pair.items():
        g, y, cols = prep[(sc, pr)]
        detail.setdefault(sc, {})[pr] = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "auc": d}
    for sc in detail:
        detail[sc] = dict(sorted(detail[sc].items(), key=lambda x: -x[1]["n_pos"]))
    rep = {"models": dict((n, data[n][1]) for n in names), "n_utts": len(common), "weighted_auc": main_t,
           "ci95_diff_vs_first": ci, "pairs": detail}
    print(json.dumps({"weighted_auc": main_t, "ci95_diff_vs_first": ci}, ensure_ascii=False, indent=1), flush=True)
    for scope in ("onset_lenis",):
        print("\n| 쌍 (%s) | 양성 | %s |" % (scope, " | ".join(names)))
        print("|---|---:|" + "---:|" * len(names))
        for pr, d in detail[scope].items():
            print("| %s→%s | %d | %s |" % (pr[0], pr[1], d["n_pos"], " | ".join("%.4f" % d["auc"][n] if d["auc"][n] is not None else "—" for n in names)))
    if a.out:
        json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

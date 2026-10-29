# -*- coding: utf-8 -*-
"""세기 쌍 되살리기(`mdd.pair_rescue`) 효과 — 서비스 모델·서비스 경로 그대로, 받아쓰기가 표준으로 덮은 세기 오류를
쌍별 문턱으로 얼마나 되살리고, 맞게 읽은 자리를 얼마나 더 틀렸다고 하는지 잰다.

extract (GPU)  발화마다: 무음 다듬기(mdd.vad.trim) → w2v-BERT CTC 로그확률 → 그리디 받아쓰기 → 표준 축 투영
               → 표준 자모 그대로 적힌 세기 계열 자리마다 같은 계열 다른 자모의 LLR
               고르는 발화 = 세기 쌍 오류가 있는 발화 전부 + 무작위 --n-random 개(헛지적은 무작위 쪽에서만 잰다)
fit (CPU)      쌍마다 "맞게 읽은 c 자리 중 a 로 바꾸는 비율 ≤ 예산" 이 되는 문턱을 고른다.
               발화를 둘로 나눠 한쪽에서 고르고 다른 쪽에서 잰다(교차). 최종 문턱은 전부로 다시 고른다.
정답 = TAG(.ltr, 사람이 들은 발음) · 기준 = .canon. 화자 정보(meta)는 읽지 않는다.

    $PY .../eval_pair_rescue.py extract --shard 0/3
    $PY .../eval_pair_rescue.py fit
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, json, os, random, time, wave

SR = 16000
LENIS = set("ㄱㄷㅂㅈㅅ")


def proj(canon, seq):
    import edlib
    c, s = "".join(canon), "".join(seq)
    if not s:
        return [None] * len(c)
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    return [None if t == "-" else t for q, t in zip(a["query_aligned"], a["target_aligned"]) if q != "-"]


def read_split(mdir, split):
    with open(os.path.join(mdir, split + ".tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t") for l in f]
    rd = lambda ext: [l.split() for l in open(os.path.join(mdir, "%s.%s" % (split, ext)), encoding="utf-8").read().splitlines()]
    return root, rows, rd("ltr"), rd("canon")


def extract(a):
    import numpy as np
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    from mdd.pair_rescue import FAMILY_OF, candidates, pair_llrs
    from mdd.vad import trim

    root, rows, ltr, can = read_split(a.manifest, a.split)
    ok_idx, err_utts = [], set()
    for i in range(len(can)):
        if not can[i] or any(len(x) != 1 for x in can[i] + ltr[i]):
            continue
        ok_idx.append(i)
        tm = proj(can[i], ltr[i])
        if any(x in FAMILY_OF and tm[j] is not None and tm[j] != x and FAMILY_OF.get(tm[j]) == FAMILY_OF[x]
               for j, x in enumerate(can[i])):
            err_utts.add(i)
    rand = set(random.Random(a.seed).sample(ok_idx, min(a.n_random, len(ok_idx))))
    pick = sorted(err_utts | rand)
    si, sn = (int(x) for x in a.shard.split("/"))
    pick = pick[si::sn]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "%s.%d-of-%d.jsonl" % (a.split, si, sn))
    done = set()
    if os.path.exists(path):
        done = set(json.loads(l)["u"] for l in open(path, encoding="utf-8") if l.strip())
    todo = [i for i in pick if i not in done]
    if a.limit:
        todo = todo[: a.limit]
    print("[%s] 발화 %d · 세기 오류 발화 %d · 무작위 %d · 이 조각 %d (이미 %d)" % (
        a.split, len(ok_idx), len(err_utts), len(rand), len(pick), len(done)), flush=True)

    dec = HFDecoder(a.ckpt)
    meta = {"ckpt": os.path.abspath(a.ckpt), "manifest": a.manifest, "split": a.split, "n_random": a.n_random, "seed": a.seed}
    mp = os.path.join(a.out, "extract_meta.json")
    if os.path.exists(mp) and json.load(open(mp, encoding="utf-8"))["ckpt"] != meta["ckpt"]:
        raise SystemExit("이 폴더는 다른 모델로 추출했다: %s — --out 을 바꿀 것" % mp)
    json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo:
        for n, i in enumerate(todo):
            with wave.open(os.path.join(root, rows[i][0]), "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            x, _ = trim(x, SR)
            canon, tag = can[i], ltr[i]
            y = [dec.d.index(t) for t in canon]
            if dec.d.unk() in y:
                continue
            lp, Ls = forward_lprobs(dec, [x])
            l = lp[0, :Ls[0]]
            g = l.argmax(-1).tolist()
            hyp, prev = [], None
            for v in g:
                if v != prev and v != dec.blank:
                    t = dec.d[v]
                    if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163:
                        hyp.append(t)
                prev = v
            pm, tm = proj(canon, hyp), proj(canon, tag)
            cands = candidates(canon, keep=lambda j: pm[j] == canon[j])
            llrs = pair_llrs(l, y, cands, dec.d.index, blank=dec.blank) if cands else []
            per = collections.defaultdict(dict)
            for (j, c, alt), v in zip(cands, llrs or []):
                per[j][alt] = round(v, 3) if v != float("-inf") else None
            sites = [[j, c, tm[j], pm[j], per.get(j, {})] for j, c in enumerate(canon) if c in FAMILY_OF]
            ok = [j for j, c in enumerate(canon) if tm[j] == c]
            er = [j for j, c in enumerate(canon) if tm[j] != c]
            cnt = [len(ok), sum(pm[j] != canon[j] for j in ok), len(er), sum(pm[j] != canon[j] for j in er),
                   sum(pm[j] == tm[j] for j in er)]
            fo.write(json.dumps({"u": i, "rand": i in rand, "sites": sites, "cnt": cnt,
                                 "align_fail": llrs is None}, ensure_ascii=False) + "\n")
            if n % 1000 == 0:
                fo.flush()
                print("  %d/%d · %.0fs" % (n + 1, len(todo), time.time() - t0), flush=True)
    print("[완료] %s · %.0fs" % (path, time.time() - t0), flush=True)


def fit(a):
    from mdd.pair_rescue import PAIRS, rescue

    mp = os.path.join(a.out, "extract_meta.json")
    model = json.load(open(mp, encoding="utf-8"))["ckpt"] if os.path.exists(mp) else None
    if model is None:
        print("⚠️ extract_meta.json 이 없다 — 이 문턱이 어느 모델 것인지 기록할 수 없다. --model-name 으로 적을 것", flush=True)
        model = a.model_name
    U = []
    for p in sorted(glob.glob(os.path.join(a.out, "%s.*-of-*.jsonl" % a.split))):
        U += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    seen, uniq = set(), []
    for u in U:
        if u["u"] not in seen:
            seen.add(u["u"]); uniq.append(u)
    U = uniq
    budgets = [float(b) for b in a.budgets.split(",")]
    print("[데이터] 발화 %d (무작위 %d) · 정렬 실패 %d" % (len(U), sum(u["rand"] for u in U), sum(u["align_fail"] for u in U)))

    def choose(units, budget, pairs):
        vals, n_ok = collections.defaultdict(list), collections.Counter()
        for u in units:
            if not u["rand"]:
                continue
            for j, c, t, p, llr in u["sites"]:
                if t != c:
                    continue
                n_ok[c] += 1
                if p == c:
                    for alt, v in llr.items():
                        if v is not None:
                            vals[c + alt].append(v)
        tau = {}
        for ca in pairs:
            allowed = int(budget * n_ok[ca[0]])
            v = sorted(vals[ca], reverse=True)
            if allowed <= 0 or not v:
                continue
            tau[ca] = v[allowed] if len(v) > allowed else v[-1] - 1e-6
        return tau

    def evaluate(units, tau):
        """→ 쌍별 [오류자리, 전 들린그대로, 후 들린그대로, 전 덮음, 후 덮음] · 쌍별 [맞게읽은 c 자리, 전 a로, 후 a로] · 전체 카운트."""
        E = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
        FA = collections.defaultdict(lambda: [0, 0, 0])
        tot = collections.Counter()
        for u in units:
            sites = u["sites"]
            pm = dict((j, p) for j, c, t, p, llr in sites)
            cands = [(j, c, alt) for j, c, t, p, llr in sites for alt in llr]
            llrs = [(llr[alt] if llr[alt] is not None else float("-inf")) for j, c, t, p, llr in sites for alt in llr]
            # rescue 는 표준 축 전체 투영을 받는다 — 세기 자리만 있으면 충분하다(다른 자리는 안 바뀐다)
            idx = dict((j, k) for k, (j, c, t, p, llr) in enumerate(sites))
            proj_pred = [None] * (max(idx) + 1 if idx else 0)
            for j, c, t, p, llr in sites:
                proj_pred[j] = p
            new, hits = rescue(proj_pred, cands, llrs, tau)
            if u["rand"]:
                n_ok, ok_flag, n_err, err_det, err_pres = u["cnt"]
                tot["ok"] += n_ok; tot["ok_flag"] += ok_flag; tot["err"] += n_err
                tot["err_det"] += err_det; tot["err_pres"] += err_pres
            for j, c, t, p, llr in sites:
                q = new[j]
                if t != c and t is not None and t in "ㄱㄲㅋㄷㄸㅌㅂㅃㅍㅈㅉㅊㅅㅆ" and (c + t) in PAIRS:
                    e = E[c + t]
                    e[0] += 1; e[1] += p == t; e[2] += q == t; e[3] += p == c; e[4] += q == c
                if u["rand"]:
                    if t == c:
                        for ca in [x for x in PAIRS if x[0] == c]:
                            f = FA[ca]
                            f[0] += 1; f[1] += p == ca[1]; f[2] += q == ca[1]
                        tot["ok_flip"] += (p == c and q != c)
                    elif t is not None:
                        tot["err_rescued"] += (p == c and q != c)
                        tot["err_rescued_right"] += (p == c and q == t)
        return E, FA, tot

    rng = random.Random(a.seed)
    fold = dict((u["u"], rng.random() < 0.5) for u in U)
    lines = ["# 세기 쌍 되살리기 — 모델 %s · %s (본 문항 · 새 화자)" % (model, a.split), "",
             "발화 %d개(세기 오류 발화 전부 + 무작위 %d). 헛지적·전체 수치는 무작위 발화에서만 쟀다. 문턱은 발화를 둘로 나눠 교차로 고르고 쟀다." % (
                 len(U), sum(u["rand"] for u in U)), ""]
    rep = {"split": a.split, "n_utts": len(U), "budgets": {}}
    scen = {"평음 자리만(ㄱㄷㅂㅈㅅ → 다른 세기)": [p for p in PAIRS if p[0] in LENIS], "모든 쌍": list(PAIRS)}
    for b in budgets:
        for sname, pairs in scen.items():
            agg_E = collections.defaultdict(lambda: [0, 0, 0, 0, 0]); agg_FA = collections.defaultdict(lambda: [0, 0, 0]); agg_T = collections.Counter()
            for f in (False, True):
                tune = [u for u in U if fold[u["u"]] != f]
                test = [u for u in U if fold[u["u"]] == f]
                E, FA, T = evaluate(test, choose(tune, b, pairs))
                for k, v in E.items():
                    agg_E[k] = [x + y for x, y in zip(agg_E[k], v)]
                for k, v in FA.items():
                    agg_FA[k] = [x + y for x, y in zip(agg_FA[k], v)]
                agg_T.update(T)
            T = agg_T
            ok_b, ok_a = T["ok_flag"] / T["ok"], (T["ok_flag"] + T["ok_flip"]) / T["ok"]
            rec_b, rec_a = T["err_det"] / T["err"], (T["err_det"] + T["err_rescued"]) / T["err"]
            pres_b, pres_a = T["err_pres"] / T["err"], (T["err_pres"] + T["err_rescued_right"]) / T["err"]
            rep["budgets"].setdefault(str(b), {})[sname] = {
                "overall": {"false_flag": [round(ok_b, 5), round(ok_a, 5)], "recall": [round(rec_b, 4), round(rec_a, 4)],
                            "preserve": [round(pres_b, 4), round(pres_a, 4)]},
                "pairs": dict((k, {"err": v[0], "pres": [round(v[1] / max(1, v[0]), 4), round(v[2] / max(1, v[0]), 4)],
                                   "revert": [round(v[3] / max(1, v[0]), 4), round(v[4] / max(1, v[0]), 4)],
                                   "fa": [round(agg_FA[k][1] / max(1, agg_FA[k][0]), 5), round(agg_FA[k][2] / max(1, agg_FA[k][0]), 5)]})
                              for k, v in agg_E.items() if k in pairs)}
            lines += ["## 예산 %.2f%% · %s" % (100 * b, sname), "",
                      "전체(무작위 발화): 맞게 읽은 자리 헛지적 %.2f%% → %.2f%% · 오류 재현율 %.1f%% → %.1f%% · 들린 그대로 %.1f%% → %.1f%%" % (
                          100 * ok_b, 100 * ok_a, 100 * rec_b, 100 * rec_a, 100 * pres_b, 100 * pres_a), "",
                      "| 오류 | 자리 | 들린 그대로 전 → 후 | 표준으로 덮음 전 → 후 | 맞게 읽었는데 그 소리로 적음 전 → 후 |",
                      "|---|---:|---:|---:|---:|"]
            for k in sorted((k for k in agg_E if k in pairs), key=lambda k: (k[0] not in LENIS, -agg_E[k][0])):
                v, fa = agg_E[k], agg_FA[k]
                lines.append("| %s→%s | %d | %.1f%% → %.1f%% | %.1f%% → %.1f%% | %.2f%% → %.2f%% |" % (
                    k[0], k[1], v[0], 100 * v[1] / max(1, v[0]), 100 * v[2] / max(1, v[0]), 100 * v[3] / max(1, v[0]),
                    100 * v[4] / max(1, v[0]), 100 * fa[1] / max(1, fa[0]), 100 * fa[2] / max(1, fa[0])))
            lines.append("")
    final = dict((str(b), choose(U, b, list(PAIRS))) for b in budgets)
    json.dump({"tau_by_budget": final, "note": "LLR = log P(표준에서 한 자리만 바꾼 열) − log P(표준). 문턱을 넘으면 그 소리로 본다",
               "model": model, "split": a.split}, open(os.path.join(a.out, "pair_tau.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "fit"])
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    ap.add_argument("--n-random", dest="n_random", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="점검용 — 조각 앞의 N개만")
    ap.add_argument("--model-name", dest="model_name", default=None,
                    help="fit: extract_meta.json 이 없는 예전 추출 폴더의 모델 경로를 직접 적는다")
    ap.add_argument("--budgets", default="0.0025,0.005,0.01,0.02", help="맞게 읽은 c 자리 중 a 로 바꿔도 되는 비율(쌍마다)")
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "analysis", "pair_rescue_2026-10-10"))
    a = ap.parse_args()
    (extract if a.mode == "extract" else fit)(a)


if __name__ == "__main__":
    main()

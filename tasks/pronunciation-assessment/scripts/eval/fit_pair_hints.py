# -*- coding: utf-8 -*-
"""세기 쌍 '힌트' 문턱 — 전문가 리뷰(2026-10-11) 권고대로 좁힌 판정을 `eval_pair_rescue.py extract` 결과로 다시 맞춘다.

좁힌 것
  - 표준이 **평음**(ㄱ·ㄷ·ㅂ·ㅈ)인 자리만. ㅅ→ㅆ 과 표준이 경음·격음인 쌍은 끈다(헛지적이 높거나 정밀도가 낮다).
  - **첫소리만**: 받침(coda)은 세기 대립이 없어 뺀다. 장애음 뒤(경음화 맥락)도 뺀다.
  - 문턱은 (쌍, 위치) 마다 따로: 발화 첫 자리 · 모음 사이 · 공명음(ㄴㄹㅁㅇ) 뒤 — 단서가 다르다.
  - 예산 = 그 (쌍, 위치)의 맞게 읽은 자리 중 바꿔 보이는 비율. 기본 0.25%.
  - **틀림이 아니라 힌트**로만 쓴다(화면 지적 errorType 은 바꾸지 않는다). 발화당 힌트는 근거가 가장 큰 1개만.
평가: 문항(파일 이름의 문항 코드) 단위로 둘로 나눠 한쪽에서 문턱을 고르고 다른 쪽에서 잰다(같은 문장이 양쪽에 들어가지 않게).
켜는 조건: 고르는 쪽에서 새 힌트 정밀도 ≥ --min-precision 이고 오류 자리 ≥ --min-err.
헛지적·정밀도는 무작위 발화에서만 잰다(추출이 오류 발화를 일부러 더 넣었기 때문).

    $PY .../fit_pair_hints.py --extract $MDD_OUTPUTS/analysis/pair_rescue_2026-10-10
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

CLASSES = ("initial", "intervocalic", "post_sonorant")
LENIS = "ㄱㄷㅂㅈ"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", default=os.path.join(str(MDD_OUTPUTS), "analysis", "pair_rescue_2026-10-10"))
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--budget", type=float, default=0.0025)
    ap.add_argument("--min-precision", dest="min_precision", type=float, default=0.35)
    ap.add_argument("--min-err", dest="min_err", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from mdd.pair_rescue import FAMILY_OF, position

    can = [l.split() for l in open(os.path.join(a.manifest, a.split + ".canon"), encoding="utf-8").read().splitlines()]
    item = []
    with open(os.path.join(a.manifest, a.split + ".tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            m = re.search(r"-(R[WSP]\d{3})-", line)
            item.append(m.group(1) if m else "?")
    meta = json.load(open(os.path.join(a.extract, "extract_meta.json"), encoding="utf-8"))
    U = {}
    for p in sorted(glob.glob(os.path.join(a.extract, "%s.*-of-*.jsonl" % a.split))):
        for l in open(p, encoding="utf-8"):
            u = json.loads(l); U[u["u"]] = u
    # 자리: (u, rand, item, key=(c+a, 위치), c, t, pm, llr)
    S = []
    for u in U.values():
        cg = can[u["u"]]
        for j, c, t, pm, llr in u["sites"]:
            if c not in LENIS:
                continue
            pos = position(cg, j)
            if pos not in CLASSES:
                continue
            S.append((u["u"], u["rand"], item[u["u"]], pos, c, t, pm, llr))
    items = sorted(set(item))
    rng = random.Random(a.seed)
    fold = dict((it, rng.random() < 0.5) for it in items)

    def tune(sites):
        """(쌍, 위치) → (문턱, 고르는 쪽 정밀도, 오류 자리 수)."""
        ok = collections.defaultdict(list); n_ok = collections.Counter(); err = collections.defaultdict(list)
        for u, rnd, it, pos, c, t, pm, llr in sites:
            for alt in FAMILY_OF[c]:
                if alt == c or (c + alt) == "ㅅㅆ":
                    continue
                k = (c + alt, pos); v = llr.get(alt)
                if t == c and rnd:
                    n_ok[k] += 1
                    if pm == c and v is not None:
                        ok[k].append(v)
                elif t == alt and pm == c and v is not None:
                    err[k].append(v)
        out = {}
        for k in set(n_ok) | set(err):
            allowed = int(a.budget * n_ok[k])
            v = sorted(ok[k], reverse=True)
            if allowed <= 0 or len(v) <= allowed:
                continue
            th = v[allowed]
            fp = sum(x > th for x in v)
            tp = sum(x > th for x in err[k])
            out[k] = (th, tp / max(1, tp + fp), len(err[k]))
        return out

    def evaluate(sites, enabled):
        """발화당 근거가 가장 큰 힌트 1개만 남긴다."""
        best = {}
        for u, rnd, it, pos, c, t, pm, llr in sites:
            if pm != c:
                continue
            for alt, v in llr.items():
                k = (c + alt, pos)
                if k not in enabled or v is None or not (v > enabled[k]):
                    continue
                m = v - enabled[k]
                if u not in best or m > best[u][0]:
                    best[u] = (m, c, t, alt, rnd, k)
        res = collections.Counter(); per = collections.defaultdict(collections.Counter)
        for u, (m, c, t, alt, rnd, k) in best.items():
            per[k]["hint"] += 1; per[k]["right"] += t == alt; per[k]["err"] += t != c
            if rnd:
                res["hint_rand"] += 1; res["right_rand"] += t == alt; res["err_rand"] += t != c
            res["hint_all"] += 1; res["right_all"] += t == alt
        res["utt_rand"] = sum(1 for x in U.values() if x["rand"] and item[x["u"]] in {it for it in items if fold[it] == evaluate.f})
        return res, per

    tot = collections.Counter(); per_tot = collections.defaultdict(collections.Counter); chosen = {}
    for f in (False, True):
        tr = [s for s in S if fold[s[2]] != f]
        te = [s for s in S if fold[s[2]] == f]
        t = tune(tr)
        enabled = dict((k, th) for k, (th, prec, ne) in t.items() if prec >= a.min_precision and ne >= a.min_err)
        chosen[str(f)] = sorted("%s→%s@%s" % (k[0][0], k[0][1], k[1]) for k in enabled)
        evaluate.f = f
        r, per = evaluate(te, enabled)
        tot.update(r)
        for k, v in per.items():
            per_tot[k].update(v)
    # 오류 자리 중 힌트로 드러난 비율(무작위 발화) 은 위에서 발화당 1개로 자른 뒤 센다
    n_rand_utt = sum(1 for u in U.values() if u["rand"])
    rep = {"model": meta.get("ckpt"), "budget": a.budget, "min_precision": a.min_precision,
           "enabled_by_fold": chosen,
           "hint_rate_per_utt_rand": tot["hint_rand"] / max(1, n_rand_utt),
           "hint_precision_rand(실제 오류)": tot["err_rand"] / max(1, tot["hint_rand"]),
           "hint_right_sound_rand": tot["right_rand"] / max(1, tot["hint_rand"]),
           "hints_rand": tot["hint_rand"], "hints_all": tot["hint_all"], "hints_all_right_sound": tot["right_all"],
           "by_pair_pos": dict(("%s→%s@%s" % (k[0][0], k[0][1], k[1]), dict(v)) for k, v in sorted(per_tot.items()))}
    out = os.path.join(a.extract, "hints_b%s.json" % a.budget)
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    # 최종 문턱(전부로 고름) — 서비스에 붙일 때 쓰는 파일. 모델 경로를 같이 적는다
    t = tune(S)
    final = dict(("%s|%s" % k, th) for k, (th, prec, ne) in t.items() if prec >= a.min_precision and ne >= a.min_err)
    json.dump({"model": meta.get("ckpt"), "budget": a.budget, "min_precision": a.min_precision, "positions": list(CLASSES),
               "tau": final, "use": "힌트 전용 — errorType 을 바꾸지 않는다. 발화당 1개"},
              open(os.path.join(a.extract, "hint_tau_b%s.json" % a.budget), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

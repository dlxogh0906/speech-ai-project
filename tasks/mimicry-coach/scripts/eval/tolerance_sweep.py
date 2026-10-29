# -*- coding: utf-8 -*-
"""허용치를 몇 배로 바꿔 가며 셋을 같이 본다 — 보정(계획서 8장) 전 출발점.

  - 주입 검출(type_hit): 정답을 아는 어긋남을 그 음절·그 종류로 짚은 비율(validate_synthetic 의 rows.jsonl)
  - 방해 변형 지적: 같은 사람·같은 억양인데 지적한 음절 비율 = 측정 잡음(같은 rows.jsonl)
  - 실제 쌍 지적: 다른 사람이 평범하게 읽은 같은 문장에서 지적한 음절 비율 = 엄격함(real_pairs 의 pairs.jsonl)
디코드는 다시 안 한다(저장된 음절 특징만 다시 비교). 종류별 값도 같이 낸다 — 종류마다 따로 고를 수 있게.

    $PY tasks/mimicry-coach/scripts/eval/tolerance_sweep.py --synth <validate 출력> --real <real_pairs 출력> \\
        --scales 1,1.25,1.5,2 --out <md 경로>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 validate_synthetic

import argparse, collections, json, os

import numpy as np

TYPES = ["pitch_level", "pitch_dir", "duration", "pause", "energy"]
INJ_TYPE = {"pitch_up": "pitch_level", "pitch_down": "pitch_level", "contour": "pitch_dir", "longer": "duration",
            "shorter": "duration", "pause_add": "pause", "weaker": "energy"}


def scaled(s):
    from mimicry.compare import TOL
    return {k: TOL[k] * s for k in TYPES}


def real_rates(pairs, tol):
    from mimicry.compare import compare
    from validate_synthetic import unpack
    syl, fl, by = 0, 0, collections.Counter()
    shown0 = 0
    for p in pairs:
        take = dict(syllables=unpack(p["take_syl"], p["text"]), bound_agree=p["agree"], line_jer=p.get("line_jer"),
                    line_heard=p.get("line_heard"))
        C = compare(dict(syllables=unpack(p["ref_syl"], p["text"])), take, tol=tol)
        syl += p["n"]
        fl += sum(1 for row in C["rows"] if any(v["flag"] for v in row["checks"].values()))
        for row in C["rows"]:
            for t, v in row["checks"].items():
                by[t] += int(v["flag"])
        shown0 += int(not C["selected"] and C["quality"]["ok"])
    return fl / max(1, syl), {t: by[t] / max(1, syl) for t in TYPES}, shown0 / max(1, len(pairs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--scales", default="1,1.25,1.5,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from validate_synthetic import score
    rows = [json.loads(l) for l in open(os.path.join(a.synth, "rows.jsonl"), encoding="utf-8")]
    pairs = [json.loads(l) for l in open(os.path.join(a.real, "pairs.jsonl"), encoding="utf-8")]
    scales = [float(x) for x in a.scales.split(",")]
    L = ["# 허용치 배수에 따른 교환 관계", "",
         "허용치 = 계획서 6.5 초안 × 배수(음높이 2반음 · 끝억양 3반음 · 길이 1.4배 · 쉼 0.25초 · 세기 6dB — 길이는 로그 배수라 ×2 면 1.4² = 1.96배). "
         "합성 원천 %d개 · 실제 쌍 %d개. 7음절 대사 기준 잘못된 지적 = 방해 변형 음절 비율 × 7." % (
             sum(1 for r in rows if r["variant"] == "ref"), len(pairs)), "",
         "| 배수 | 주입 검출(평균) | 주입이 1순위 | 방해 변형 지적(음절) | 7음절 대사당 잘못된 지적 | 실제 쌍 지적(음절) | 실제 쌍 지적 0 발화 |",
         "|--:|--:|--:|--:|--:|--:|--:|"]
    per_type = []
    for s in scales:
        tol = scaled(s)
        nu, inj, _ = score(rows, tol)
        n_syl = sum(st["syl"] for st in nu.values())
        n_fl = sum(st["flag_syl"] for st in nu.values())
        hit = np.mean([st["type_hit"] / st["n"] for st in inj.values()])
        top1 = np.mean([st["top1"] / st["n"] for st in inj.values()])
        rr, rby, r0 = real_rates(pairs, tol)
        L.append("| ×%.2g | %.1f%% | %.1f%% | %.1f%% | %.2f | %.1f%% | %.1f%% |" % (
            s, 100 * hit, 100 * top1, 100.0 * n_fl / n_syl, 7.0 * n_fl / n_syl, 100 * rr, 100 * r0))
        nby = {t: sum(st["flag_" + t] for st in nu.values()) / n_syl for t in TYPES}
        iby = {}
        for t in TYPES:
            ks = [st for (k, b), st in inj.items() if INJ_TYPE[k] == t]
            iby[t] = np.mean([st["type_hit"] / st["n"] for st in ks]) if ks else float("nan")
        per_type.append((s, iby, nby, rby))
    L += ["", "## 종류별 — 주입 검출 / 방해 변형 지적(음절) / 실제 쌍 지적(음절)", "",
          "| 배수 | " + " | ".join(TYPES) + " |", "|--:|" + "---|" * len(TYPES)]
    for s, iby, nby, rby in per_type:
        L.append("| ×%.2g | " % s + " | ".join("%.0f%% / %.1f%% / %.1f%%" % (100 * iby[t], 100 * nby[t], 100 * rby[t])
                                               for t in TYPES) + " |")
    md = "\n".join(L) + "\n"
    open(a.out, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""스크리닝 결과를 기준선(S0 = H0 의 같은 step) 대비 **변화량**으로 한 표에 모은다.

목적은 순위가 아니라 **방향**이다 — 처음 보는 문항의 헛지적·JER·F1 이 기준선보다 나아졌나.
valid(seen) · valid_unseen 만 쓴다(test 는 보지 않는다).

    $PY .../summarize_screen.py --base $MDD_OUTPUTS/experiments/holdout_v1/holdout_v1_H0/sel \\
        --runs $MDD_OUTPUTS/experiments/screen_v1/* --out $MDD_OUTPUTS/experiments/screen_v1/summary.md
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os

KEYS = [("normal_site_error", "헛지적"), ("jer", "JER"), ("f1", "MDD F1"), ("preservation", "보존"), ("reversion", "되돌림")]


def metrics(sel, step, split):
    p = os.path.join(sel, "step_%d_%s" % (step, split), "summary.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))["baseline"]
    return {"normal_site_error": b["sites"]["normal_site_error"], "jer": b["jer"]["jer"], "f1": b["mdd"]["f1"],
            "preservation": b["sites"]["preservation"], "reversion": b["sites"]["reversion"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="기준선 sel 폴더(H0)")
    ap.add_argument("--runs", nargs="+", required=True, help="스크리닝 결과 폴더들")
    ap.add_argument("--steps", nargs="+", type=int, default=[2000, 4000])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = ["# 스크리닝 요약 — 기준선(H0 같은 step) 대비 변화", "",
         "헛지적·JER 은 **낮을수록**, F1·보존은 **높을수록** 좋다. 값은 절대값(괄호 안은 기준선 대비 변화).", ""]
    for step in a.steps:
        for split, title in (("valid_unseen", "처음 보는 문항 (valid_unseen)"), ("valid", "본 문항 (valid)")):
            base = metrics(a.base, step, split)
            if base is None:
                continue
            L += ["## %s · step %d" % (title, step), "",
                  "| 실험 | 인자 | " + " | ".join(n for _, n in KEYS) + " |", "|---|---|" + "--:|" * len(KEYS)]
            L.append("| **S0 (H0)** | — | " + " | ".join("%.4f" % base[k] for k, _ in KEYS) + " |")
            for d in sorted(a.runs):
                if not os.path.isdir(d):
                    continue
                m = metrics(os.path.join(d, "sel"), step, split)
                if m is None:
                    continue
                args = open(os.path.join(d, "args.txt")).read().strip() if os.path.exists(os.path.join(d, "args.txt")) else ""
                L.append("| %s | `%s` | " % (os.path.basename(d), args) +
                         " | ".join("%.4f (%+.4f)" % (m[k], m[k] - base[k]) for k, _ in KEYS) + " |")
            L.append("")
    # ── 궤적 비교: 같은 step 이 아니라 H0 의 모든 step 과 비교한다 ──
    # 학습률을 낮추거나 층을 고정하면 학습이 느려져, 같은 step 에서는 "덜 외운" 상태라 unseen 이 좋아 보일 수 있다.
    # 진짜 개선이면 H0 의 어떤 step 보다도 (seen 헛지적, unseen 헛지적) 두 축이 동시에 나아야 한다.
    import glob, re
    h0 = {}
    for q in glob.glob(os.path.join(a.base, "step_*_valid", "summary.json")):
        st = int(re.search(r"step_(\d+)_valid", q).group(1))
        u = metrics(a.base, st, "valid_unseen")
        if u is not None:
            h0[st] = (metrics(a.base, st, "valid")["normal_site_error"], u["normal_site_error"], u["f1"])
    if h0:
        L += ["## H0 궤적 대비 판정 (valid · 헛지적 두 축)", "",
              "H0 의 어느 step 이 **seen 헛지적과 unseen 헛지적 모두** 같거나 낮으면 그 점은 '지배됨' — 학습을 덜 한 H0 로 "
              "이미 얻을 수 있는 상태라 새 설정의 이득이 아니다.", "",
              "| 실험 | step | seen 헛지적 | unseen 헛지적 | unseen F1 | 판정 |", "|---|--:|--:|--:|--:|---|"]
        for st in sorted(h0):
            L.append("| S0 (H0) | %d | %.4f | %.4f | %.4f | 기준 궤적 |" % ((st,) + h0[st]))
        for d in sorted(a.runs):
            if not os.path.isdir(d):
                continue
            for st in a.steps:
                v, u = metrics(os.path.join(d, "sel"), st, "valid"), metrics(os.path.join(d, "sel"), st, "valid_unseen")
                if v is None or u is None:
                    continue
                dom = [k for k, (sv, su, _) in h0.items() if sv <= v["normal_site_error"] and su <= u["normal_site_error"]]
                verdict = ("지배됨 (H0 step %s 가 두 축 모두 낫다)" % ",".join(str(k) for k in sorted(dom))) if dom \
                          else "**지배되지 않음 — 새 영역**"
                L.append("| %s | %d | %.4f | %.4f | %.4f | %s |" % (os.path.basename(d), st, v["normal_site_error"],
                                                                     u["normal_site_error"], u["f1"], verdict))
        L.append("")
    txt = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

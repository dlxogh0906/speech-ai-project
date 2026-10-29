# -*- coding: utf-8 -*-
"""문항 hold-out 실험의 **체크포인트 선택** — UER 최저가 아니라 MDD 기준으로, valid 만 보고 고른다.

규칙(종합 점수를 만들지 않는다 — 제약 하나 + 목표 하나):
  제약  valid_unseen 정상자리오류 ≤ 기준 + tol
        · 기준 = --reference 가 있으면 그 실험(H0)의 선택 step 값, 없으면 이 실험의 최솟값
  목표  제약을 만족하는 step 중 valid_unseen MDD F1 최대 (동률이면 이른 step)
판단에 필요한 나머지(seen 성능·보존·되돌림·JER)는 표로 함께 남긴다.

입력: <sel_dir>/step_XXXXX_{valid,valid_unseen}/summary.json (mdd_error_analysis.py 산출물)

    $PY .../select_holdout_ckpt.py --sel-dir $R/sel --out $R/selection.json [--reference $H0/selection.json]
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, glob, json, os, re


def load(sel_dir):
    rows = {}
    for p in glob.glob(os.path.join(sel_dir, "step_*_*", "summary.json")):
        m = re.search(r"step_(\d+)_(valid_unseen|valid)$", os.path.basename(os.path.dirname(p)))
        if not m:
            continue
        b = json.load(open(p))["baseline"]
        rows.setdefault(int(m.group(1)), {})[m.group(2)] = {
            "jer": b["jer"]["jer"], "precision": b["mdd"]["precision"], "recall": b["mdd"]["recall"],
            "f1": b["mdd"]["f1"], "preservation": b["sites"]["preservation"], "reversion": b["sites"]["reversion"],
            "normal_site_error": b["sites"]["normal_site_error"]}
    return dict((k, v) for k, v in rows.items() if "valid" in v and "valid_unseen" in v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sel-dir", dest="sel_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="기준 실험(H0)의 selection.json")
    ap.add_argument("--tol", type=float, default=0.002, help="정상자리오류 허용 폭(절대값, 0.002 = 0.2%p)")
    a = ap.parse_args()
    rows = load(a.sel_dir)
    if not rows:
        raise SystemExit("채점된 step 이 없다: %s" % a.sel_dir)
    if a.reference:
        ref = json.load(open(a.reference))
        base = ref["chosen"]["valid_unseen"]["normal_site_error"]
        base_src = "reference %s step %d" % (a.reference, ref["chosen"]["step"])
    else:
        base = min(v["valid_unseen"]["normal_site_error"] for v in rows.values())
        base_src = "이 실험의 최솟값"
    ok = [s for s in sorted(rows) if rows[s]["valid_unseen"]["normal_site_error"] <= base + a.tol]
    if not ok:                               # 제약을 아무도 못 지키면 정상자리오류 최소 step
        pick = min(rows, key=lambda s: rows[s]["valid_unseen"]["normal_site_error"])
        note = "제약을 만족하는 step 없음 → 정상자리오류 최소 step"
    else:
        pick = max(ok, key=lambda s: (rows[s]["valid_unseen"]["f1"], -s))
        note = "제약 만족 %d개 중 valid_unseen F1 최대" % len(ok)
    out = {"rule": "valid_unseen 정상자리오류 ≤ %.4f(%s) + %.4f 에서 valid_unseen MDD F1 최대" % (base, base_src, a.tol),
           "note": note, "chosen": dict({"step": pick}, **rows[pick]),
           "table": [dict({"step": s}, **rows[s]) for s in sorted(rows)]}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("%6s | %-44s | %-44s" % ("step", "valid_unseen  nse / F1 / 보존 / 되돌림 / JER", "valid_seen  nse / F1 / 보존 / 되돌림"))
    for s in sorted(rows):
        u, v = rows[s]["valid_unseen"], rows[s]["valid"]
        print("%6d%s| %.4f / %.4f / %.4f / %.4f / %.4f | %.4f / %.4f / %.4f / %.4f" % (
            s, "*" if s == pick else " ", u["normal_site_error"], u["f1"], u["preservation"], u["reversion"], u["jer"],
            v["normal_site_error"], v["f1"], v["preservation"], v["reversion"]))
    print("[pick] step %d — %s" % (pick, out["rule"]))


if __name__ == "__main__":
    main()

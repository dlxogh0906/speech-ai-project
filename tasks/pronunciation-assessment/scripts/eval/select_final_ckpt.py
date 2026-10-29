# -*- coding: utf-8 -*-
"""Stage 2 체크포인트 중 FINAL 선택 — valid_unseen 채점 결과(eval_actual_ctc.py metrics.json)만으로 고른다.

규칙: JER 최소 → 동률(소수 4자리 반올림이 같음)이면 되돌림(reversion) 낮은 것 → 보존(preservation) 높은 것
      → 그래도 같으면 이른 스텝.

    $PY .../select_final_ckpt.py --metrics <dir>/ckpt_*_valid_unseen.metrics.json --out <dir>/final_selection.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, re


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="valid_unseen", help="모든 metrics 가 이 split 이어야 한다")
    ap.add_argument("--priority", default="jer,reversion,-preservation",
                    help="정렬 키 (앞이 우선). 지표: normal jer reversion preservation. '-' 는 클수록 좋음")
    a = ap.parse_args()
    rows = []
    for p in a.metrics:
        m = json.load(open(p, encoding="utf-8"))
        up = re.search(r"checkpoint_\d+_(\d+)\.pt$", m["ckpt"]) or re.search(r"step_(\d+)/", m["ckpt"])   # fairseq · HF 폴더
        rows.append({"ckpt": m["ckpt"], "updates": int(up.group(1)) if up else -1, "metrics": p,
                     "split": m["split"], "jer": m["jer"]["jer"], "sub": m["jer"]["sub"], "del": m["jer"]["del"],
                     "ins": m["jer"]["ins"], "reversion": m["sites"]["reversion"], "preservation": m["sites"]["preservation"],
                     "normal_site_error": m["sites"]["normal_site_error"], "mdd_f1": m["mdd"]["f1"]})
    assert len({r["split"] for r in rows}) == 1 and rows[0]["split"] == a.split, f"{a.split} 채점 결과만 넣을 것"
    field = {"normal": "normal_site_error", "jer": "jer", "reversion": "reversion", "preservation": "preservation"}
    keys = [(k.startswith("-"), field[k.lstrip("-")]) for k in a.priority.split(",")]
    rows.sort(key=lambda r: tuple((-1 if neg else 1) * round(r[f], 4) for neg, f in keys) + (r["updates"],))
    sel = {"rule": f"{a.priority} (소수 4자리에서 동률이면 다음 키) → 이른 스텝", "split": a.split,
           "final": rows[0], "ranking": rows}
    json.dump(sel, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for i, r in enumerate(rows):
        print(f"{'★' if i == 0 else ' '} {r['updates']:>5}  JER {r['jer']:.4f}  되돌림 {r['reversion']}  보존 {r['preservation']}  "
              f"정상자리오류 {r['normal_site_error']}  {os.path.basename(r['ckpt'])}")
    print(rows[0]["ckpt"])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""데모에서 모은 사람 판단(`deploy/mimicry-demo/app.py --labels`) → 지적 종류·크기별로 "귀로 들어도 맞다"는 비율.

허용치 보정(계획서 8장)의 재료다. 지적마다 잰 값은 허용치의 몇 배(ratio)인지 같이 남으므로, 배수 구간별로 맞다는 비율을 보면
"이 종류는 허용치를 몇 배로 잡아야 사람 귀와 맞는가"가 보인다. 통과("원본과 거의 같아요") 판정이 맞았는지도 센다.
판단 파일엔 녹음이 없다(대사·지적·잰 값·판단만).

    $PY tasks/mimicry-coach/scripts/eval/summarize_labels.py --labels <labels.jsonl> --out <md>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, json

BINS = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, float("inf"))]
TYPE_KO = {"pitch_level": "음높이", "pitch_dir": "끝 억양", "duration": "길이", "pause": "쉼", "energy": "세기",
           "tempo": "빠르기", "voicing": "목소리", "pass": "통과 판정"}


def summarize(rows):
    by = collections.defaultdict(collections.Counter)
    bins = collections.defaultdict(collections.Counter)
    for r in rows:
        it, v = r.get("item") or {}, r.get("verdict")
        t = it.get("type")
        if v not in ("yes", "no", "unsure") or not t:
            continue
        by[t][v] += 1
        ratio = it.get("ratio")
        if t != "pass" and isinstance(ratio, (int, float)):
            for lo, hi in BINS:
                if lo <= ratio < hi:
                    bins[(t, lo)][v] += 1
                    break
    prec = lambda c: "—" if not (c["yes"] + c["no"]) else "%.0f%%" % (100.0 * c["yes"] / (c["yes"] + c["no"]))
    L = ["# 사람 판단 — 지적이 귀로 들어도 맞나", "", "판단 %d개. 맞음 비율 = 맞아요 ÷ (맞아요 + 아니에요). 모르겠어요는 빼고 센다." % sum(
        sum(c.values()) for c in by.values()), "",
         "| 종류 | 판단 | 맞아요 | 아니에요 | 모르겠어요 | 맞음 비율 | " + " | ".join(
             "%.1f~%s배" % (lo, "" if hi == float("inf") else "%.1f" % hi) for lo, hi in BINS) + " |",
         "|---|--:|--:|--:|--:|--:|" + "--:|" * len(BINS)]
    for t in sorted(by, key=lambda k: -sum(by[k].values())):
        c = by[t]
        cells = [("%s (%d)" % (prec(bins[(t, lo)]), sum(bins[(t, lo)].values()))) if t != "pass" else "—" for lo, _ in BINS]
        L.append("| %s | %d | %d | %d | %d | **%s** | %s |" % (TYPE_KO.get(t, t), sum(c.values()), c["yes"], c["no"], c["unsure"],
                                                            prec(c), " | ".join(cells)))
    L += ["", "배수 구간 = 그 지적의 차이가 허용치(compare.TOL)의 몇 배였나. 낮은 구간의 맞음 비율이 낮으면 그 종류의 허용치를 올릴 근거,",
          "높은 구간도 낮으면 측정 자체를 의심할 근거다. 판단이 종류·구간마다 30개는 넘어야 믿을 만하다."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.labels, encoding="utf-8") if l.strip()]
    md = summarize(rows)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()

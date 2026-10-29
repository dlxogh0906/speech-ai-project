# -*- coding: utf-8 -*-
"""크기별 검출률 — "이만큼 어긋나면 몇 % 를 그 음절·그 종류로 짚나" (초안 허용치 기준).

validate_synthetic 의 주입 변형을 크기만 바꿔 넣는다(같은 목소리 / 목소리 바꿈 바탕). 넣은 크기와 함께
실제로 잰 변화량도 남긴다 — PSOLA 평활·가장자리 처리 탓에 넣은 만큼 다 안 들어가는 경우를 가르려고.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/dose_response.py --n 24 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 validate_synthetic

import argparse, collections, json, math, os, random, time

import numpy as np

DOSES = {"pitch_up": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
         "contour": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
         "longer": [1.2, 1.3, 1.4, 1.6, 1.8, 2.2],
         "shorter": [0.85, 0.75, 0.7, 0.6, 0.5],
         "pause_add": [0.1, 0.2, 0.3, 0.4, 0.6],
         "weaker": [3.0, 4.5, 6.0, 8.0, 10.0, 12.0]}
UNIT = {"pitch_up": "반음", "contour": "반음(목표)", "longer": "배", "shorter": "배", "pause_add": "초", "weaker": "dB"}


def measured(kind, chk):
    """그 음절에서 실제로 잰 변화(원본 대비)."""
    if chk is None:
        return None
    if kind == "pitch_up":
        return chk["diff"]
    if kind == "contour":
        return chk["take"] - chk["ref"]
    if kind in ("longer", "shorter"):
        return chk["diff"]
    if kind == "pause_add":
        return chk["take"] - chk["ref"]
    return -chk["diff"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=9.0)
    ap.add_argument("--rescore", action="store_true", help="dose.jsonl 만 다시 요약")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "dose.jsonl")
    if not a.rescore:
        run(a, path)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    summarize(rows, os.path.join(a.out, "summary.md"))


def run(a, path):
    from mimicry import synth
    from mimicry.align import FRAME, OFFSET
    from mimicry.analyze import crosscheck
    from mimicry.audio import load_audio
    from mimicry.compare import compare
    from mimicry.pipeline import Coach
    from validate_synthetic import pick_target, zeroth_sources

    coach = Coach()
    rng = random.Random(a.seed)
    fo = open(path, "w", encoding="utf-8")
    t0 = time.time()
    for sid, (p, text) in enumerate(zeroth_sources("test_data_01", a.n, a.seed, 8, 32)):
        x = load_audio(p)
        if len(x) / 16000.0 > a.max_sec:
            continue
        A, FA = coach.analyze(x, text)
        pr = A["pitch_range"]
        wide = (max(50.0, pr[0] * 0.8), min(1100.0, pr[1] * 1.25))
        high = A["baseline_hz"] >= 165.0
        y0 = synth.psola(x, prange=wide)
        for kind, doses in DOSES.items():
            k = pick_target(A, kind, rng)
            if k is None:
                continue
            for base in ("same", "gender"):
                for m in doses:
                    src = y0 if kind in ("pause_add", "weaker") else x
                    y, _, truth = synth.inject(src, kind, A["syllables"][k], wide, ref_delta=A["syllables"][k]["delta"], mag=m)
                    if base == "gender":
                        y = synth.change_gender(y, wide, high)
                    B, FB = coach.analyze(y, text)
                    crosscheck(x, A, y, B, feats=(FA, FB), hop=FRAME, off=OFFSET)
                    C = compare(A, B)
                    row = C["rows"][k]
                    chk = row["checks"].get(truth["type"])
                    tdir = truth["dir"]
                    good = bool(chk and chk["flag"] and C["quality"]["ok"] and
                                ((chk["take_dir"] == tdir) if truth["type"] == "pitch_dir" else (chk["dir"] == tdir)))
                    others = sum(1 for r in C["rows"] if r["i"] != k and any(v["flag"] for v in r["checks"].values()))
                    fo.write(json.dumps(dict(src=sid, kind=kind, base=base, mag=m, k=k, hit=good, uncertain=row["uncertain"],
                                             measured=measured(kind, chk), ratio=None if chk is None else chk["ratio"],
                                             others=others, n=len(A["syllables"]), retake=not C["quality"]["ok"])) + "\n")
        fo.flush()
        print("[%d] %.0fs" % (sid + 1, time.time() - t0), flush=True)
    fo.close()


def summarize(rows, out):
    from mimicry.compare import TOL
    tol_txt = {"pitch_up": "허용치 %.1f반음" % TOL["pitch_level"], "contour": "허용치 %.1f반음(방향이 다를 때)" % TOL["pitch_dir"],
               "longer": "허용치 %.2f배" % math.exp(TOL["duration"]), "shorter": "허용치 %.2f배" % math.exp(-TOL["duration"]),
               "pause_add": "허용치 %.2f초" % TOL["pause"], "weaker": "허용치 %.0fdB" % TOL["energy"]}
    L = ["# 크기별 검출률 (초안 허용치)", "",
         "주입한 음절을 그 종류·그 방향으로 짚은 비율. 괄호 = 그 음절을 실제로 잰 변화의 중앙값(주입한 만큼 다 안 들어갈 수 있다) · "
         "확신 없음 = 경계 불일치로 그 음절을 안 잼.", ""]
    for kind in DOSES:
        R = [r for r in rows if r["kind"] == kind]
        if not R:
            continue
        L += ["## %s — %s" % (kind, tol_txt[kind]), "", "| 크기(%s) | 같은 목소리 | 목소리 바꿈 | 확신 없음 |" % UNIT[kind], "|--:|--:|--:|--:|"]
        for m in DOSES[kind]:
            cells = []
            for base in ("same", "gender"):
                rr = [r for r in R if r["base"] == base and r["mag"] == m]
                ms = [r["measured"] for r in rr if r["measured"] is not None]
                cells.append("%.0f%% (%s)" % (100.0 * np.mean([r["hit"] for r in rr]), "%.2f" % np.median(ms) if ms else "—")
                             if rr else "—")
            ru = [r for r in R if r["mag"] == m]
            L.append("| %g | %s | %s | %.0f%% |" % (m, cells[0], cells[1], 100.0 * np.mean([r["uncertain"] for r in ru])))
        L.append("")
    md = "\n".join(L)
    open(out, "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()

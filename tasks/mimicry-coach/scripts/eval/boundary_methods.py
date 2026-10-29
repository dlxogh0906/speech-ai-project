# -*- coding: utf-8 -*-
"""음절 경계를 녹음에 옮기는 방법 비교 — 녹음 CTC 강제 정렬 vs MFCC-DTW(계획서 6.4) vs 정렬 모델 은닉층 DTW.

  - 합성(정답 있음): Zeroth 발화를 비튼 변형(목소리·빠르기·잡음·조합 + 음절 하나 늘이기/줄이기/쉼/음높이/억양)에서
    원본 경계를 시간 사상으로 옮긴 값이 정답. |방법 − 정답| 의 중앙값·p90·50 ms 넘는 비율.
  - 실제 다른 화자 쌍(정답 없음): 같은 문장 다른 화자. 방법끼리 50 ms 안에서 맞는 비율(오류가 서로 독립이면
    다른 방법들과 덜 맞는 쪽이 더 틀린 쪽이다).
특징 층은 6·12·18·24 와 CTC 사후확률. analyze.crosscheck 가 쓰는 층(pipeline.DTW_LAYER)을 고른 근거다.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/boundary_methods.py --n-synth 20 --n-real 40 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 validate_synthetic · real_pairs

import argparse, json, os, random, time

import numpy as np

LAYERS = [6, 12, 18, 24]
KINDS = ["gender", "combo", "tempo1.25", "noise20", "longer", "shorter", "pause_add", "pitch_up", "contour"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-synth", dest="n_synth", type=int, default=20)
    ap.add_argument("--n-real", dest="n_real", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    from mimicry import dtw, synth
    from mimicry.align import FRAME, OFFSET, Aligner
    from mimicry.analyze import analyze, bounds
    from mimicry.audio import load_audio
    from real_pairs import zeroth_pairs
    from validate_synthetic import pick_target, zeroth_sources

    al = Aligner()

    def run(w, text):
        lp, hid = al.forward(w, LAYERS)
        F = dict(hid)
        F["post"] = lp.numpy()
        return analyze(w, text, al, lp=lp, keep_tracks=False), F

    def methods(xr, Ar, Fr, xt, At, Ft):
        rb = bounds(Ar)
        out = {"ctc": bounds(At), "mfcc": dtw.project(xr, xt, rb, Ar["speech"], At["speech"])}
        for L in LAYERS + ["post"]:
            out["L%s" % L] = dtw.project_feats(Fr[L], Ft[L], rb, FRAME, OFFSET, Ar["speech"], At["speech"], "cos")
        return out

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    res = {"synth": [], "real": []}
    t0 = time.time()
    for si, (path, text) in enumerate(zeroth_sources("test_data_01", a.n_synth, a.seed, 8, 32)):
        x = load_audio(path)
        A, F = run(x, text)
        pr = A["pitch_range"]
        wide = (max(50.0, pr[0] * 0.8), min(1100.0, pr[1] * 1.25))
        high = A["baseline_hz"] >= 165.0
        rb = np.array(bounds(A))
        for kind in KINDS:
            k = None
            if kind in ("gender", "combo", "tempo1.25", "noise20"):
                y, tm = synth.nuisance(x, kind, wide, np.random.default_rng(rng.randrange(1 << 30)), high)
            else:
                k = pick_target(A, kind, rng)
                if k is None:
                    continue
                src = synth.psola(x, prange=wide) if kind == "pause_add" else x
                y, tm, _ = synth.inject(src, kind, A["syllables"][k], wide, ref_delta=A["syllables"][k]["delta"])
                y = synth.change_gender(y, wide, high)
            truth = synth.map_time(tm, rb + 1e-3) - 1e-3                # 음절 시작은 오른쪽 극한(쉼을 넣으면 쉼 끝)
            B, G = run(y, text)
            res["synth"].append(dict(kind=kind, k=k, truth=[float(v) for v in truth], m=methods(x, A, F, y, B, G)))
        print("synth %d %.0fs" % (si + 1, time.time() - t0), flush=True)
    for text, pa, pb in zeroth_pairs(a.n_real, a.seed, 8, 32):
        xa, xb = load_audio(pa), load_audio(pb)
        try:
            A, F = run(xa, text)
            B, G = run(xb, text)
        except Exception as e:
            print("skip", e)
            continue
        res["real"].append(dict(m=methods(xa, A, F, xb, B, G), dst=float(12 * np.log2(B["baseline_hz"] / A["baseline_hz"]))))
    json.dump(res, open(os.path.join(a.out, "bounds.json"), "w"))

    M = ["ctc", "mfcc"] + ["L%s" % L for L in LAYERS + ["post"]]
    name = {"ctc": "강제 정렬", "mfcc": "MFCC-DTW", "Lpost": "사후확률 DTW"}
    nm = lambda m: name.get(m, "%s층 DTW" % m[1:])
    L = ["# 경계 옮기는 방법 비교", "", "합성 %d건(원천 %d) · 실제 다른 화자 %d쌍." % (len(res["synth"]), a.n_synth, len(res["real"])), "",
         "## 합성 — |방법 − 정답| ms: 중앙값 / p90 / 50ms 넘는 비율", "",
         "| 변형 | " + " | ".join(nm(m) for m in M) + " |", "|---|" + "--:|" * len(M)]
    for kind in KINDS:
        rows = [r for r in res["synth"] if r["kind"] == kind]
        if not rows:
            continue
        cells = []
        for m in M:
            e = np.concatenate([np.abs(np.array(r["m"][m]) - np.array(r["truth"])) * 1000 for r in rows])
            cells.append("%.0f / %.0f / %.1f%%" % (np.median(e), np.percentile(e, 90), 100 * np.mean(e > 50)))
        L.append("| %s | %s |" % (kind, " | ".join(cells)))
    far = [r for r in res["real"] if abs(r["dst"]) >= 5]
    agree = lambda rows, x, y: 100 * np.mean(np.concatenate([np.abs(np.array(r["m"][x]) - np.array(r["m"][y])) <= 0.05 for r in rows]))
    L += ["", "## 실제 다른 화자 쌍 — 방법끼리 50ms 안 일치율 (전체 / 음높이 차 ≥5반음 %d쌍)" % len(far), "",
          "| | " + " | ".join(nm(m) for m in M) + " |", "|---|" + "--:|" * len(M)]
    for x in M:
        L.append("| %s | %s |" % (nm(x), " | ".join("—" if x == y else "%.1f / %.1f" % (agree(res["real"], x, y), agree(far, x, y))
                                                 for y in M)))
    md = "\n".join(L) + "\n"
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()

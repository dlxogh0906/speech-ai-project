# -*- coding: utf-8 -*-
"""실제 다른 화자 쌍 — 같은 문장을 다른 사람이 읽은 Zeroth 낭독(공개 원어민 데이터).

정답이 없으니 "어긋남을 맞게 짚었나"는 못 잰다. 대신 잰다:
  - 경계 일치율(녹음 강제 정렬 vs 정렬 모델 특징 DTW vs MFCC-DTW) — 방법끼리 50 ms 안에서 맞는 비율
  - 재녹음 판정 비율, 확신 없어 안 잰 음절 비율
  - 음절당 지적 비율(종류별)·발화당 보여줄 지적 수 — 두 사람이 평범하게 읽어도 얼마나 "다르다"고 하는지
목소리 높이 차(기준 음높이 ≥ 5 반음)로도 나눠 본다. 성별 메타데이터(AUDIO_INFO)는 읽지 않는다 — 소리에서 잰 값이다.
`--reports N` 이면 앞 N 쌍의 HTML 보고서를 만든다(사람 귀로 지적이 맞는지 들어 보는 용도).
쌍마다 음절 특징(원본·녹음)을 pairs.jsonl 에 남긴다 — 허용치를 바꿔 다시 셀 땐 tolerance_sweep.py.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/real_pairs.py --n 60 --reports 8 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 validate_synthetic(pack)

import argparse, collections, glob, json, os, random, re, time

import numpy as np

TYPES = ["pitch_level", "pitch_dir", "duration", "pause", "energy"]


def zeroth_pairs(n, seed, min_syl, max_syl):
    from shared.paths import DATA_ROOT
    root = os.path.join(os.fspath(DATA_ROOT), "zeroth", "wav16k")
    by_text = collections.defaultdict(list)
    for tf in sorted(glob.glob(os.path.join(root, "*", "003", "*", "*.trans.txt"))):
        d = os.path.dirname(tf)
        for line in open(tf, encoding="utf-8"):
            stem, _, text = line.strip().partition(" ")
            if text and not re.search(r"[0-9A-Za-z]", text) and min_syl <= len(re.sub(r"[^가-힣]", "", text)) <= max_syl:
                by_text[text].append((stem.split("_")[0], os.path.join(d, stem + ".wav")))
    rng = random.Random(seed)
    texts = sorted(t for t, v in by_text.items() if len({s for s, _ in v}) >= 2)
    rng.shuffle(texts)
    out = []
    for t in texts[:n]:
        v = sorted(by_text[t])
        a = rng.choice(v)
        b = rng.choice([x for x in v if x[0] != a[0]])
        out.append((t, a[1], b[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-syl", dest="min_syl", type=int, default=8)
    ap.add_argument("--max-syl", dest="max_syl", type=int, default=32)
    ap.add_argument("--reports", type=int, default=0, help="앞 N 쌍의 HTML 보고서")
    a = ap.parse_args()

    from mimicry import dtw
    from validate_synthetic import pack
    from mimicry.analyze import bounds
    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach
    from mimicry.report import html

    os.makedirs(a.out, exist_ok=True)
    if a.reports:
        os.makedirs(os.path.join(a.out, "reports"), exist_ok=True)
    coach = Coach()
    rows = []
    t0 = time.time()
    for pi, (text, pa, pb) in enumerate(zeroth_pairs(a.n, a.seed, a.min_syl, a.max_syl)):
        xa, xb = load_audio(pa), load_audio(pb)
        try:
            ref = coach.analyze(xa, text)
            A, B, C = coach.compare(xa, xb, text, ref=ref)
        except Exception as e:
            print("[skip] pair %d: %s" % (pi, e), flush=True)
            continue
        mf = dtw.project(xa, xb, bounds(A), A["speech"], B["speech"])
        ctc = bounds(B)
        feat = [s["dtw"][0] for s in B["syllables"]] + [B["syllables"][-1]["dtw"][1]]
        flagged = collections.Counter()
        for row in C["rows"]:
            for t, v in row["checks"].items():
                flagged[t] += int(v["flag"])
        rows.append(dict(pair=pi, n=len(A["syllables"]), dst=float(12 * np.log2(B["baseline_hz"] / A["baseline_hz"])),
                         agree=B["bound_agree"], line_jer=B.get("line_jer"), line_heard=B.get("line_heard"),
                         retake=not C["quality"]["ok"], uncertain=C["quality"]["uncertain"],
                         flag_syl=sum(1 for row in C["rows"] if any(v["flag"] for v in row["checks"].values())),
                         flagged=dict(flagged), shown=len(C["selected"]), tempo=C["tempo"],
                         shown_types=[s["type"] for s in C["selected"]], text=re.sub(r"[^가-힣]", "", text),
                         ref_syl=pack(A["syllables"]), take_syl=pack(B["syllables"]),
                         ag_ctc_mfcc=float(np.mean(np.abs(np.array(ctc) - np.array(mf)) <= 0.05)),
                         ag_ctc_feat=float(np.mean(np.abs(np.array(ctc) - np.array(feat)) <= 0.05)),
                         ag_feat_mfcc=float(np.mean(np.abs(np.array(feat) - np.array(mf)) <= 0.05))))
        if pi < a.reports:
            with open(os.path.join(a.out, "reports", "pair_%02d.html" % pi), "w", encoding="utf-8") as f:
                f.write(html(A, B, C, xa, xb, "Zeroth 실제 화자 쌍 %d (원본 = 화자 A, 녹음 = 화자 B)" % pi))
        print("[%d] %d음절 일치 %.0f%% 지적 %d %.0fs" % (pi, rows[-1]["n"], 100 * B["bound_agree"], rows[-1]["shown"],
                                                time.time() - t0), flush=True)
    with open(os.path.join(a.out, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def block(rs, name):
        if not rs:
            return ["| %s | 0 | — |" % name]
        syl = sum(r["n"] for r in rs)
        f = lambda k: 100.0 * sum(r["flagged"].get(k, 0) for r in rs) / syl
        return ["| %s | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.2f | %.1f%% | %s |" % (
            name, len(rs), 100 * np.mean([r["ag_ctc_feat"] for r in rs]), 100 * np.mean([r["ag_ctc_mfcc"] for r in rs]),
            100 * np.mean([r["ag_feat_mfcc"] for r in rs]), 100.0 * np.mean([r["retake"] for r in rs]),
            100.0 * sum(r["uncertain"] for r in rs) / syl, 100.0 * sum(r["flag_syl"] for r in rs) / syl,
            np.mean([r["shown"] for r in rs]), 100.0 * np.mean([r["shown"] == 0 and not r["retake"] for r in rs]),
            " · ".join("%s %.1f%%" % (k, f(k)) for k in TYPES))]
    near = [r for r in rows if abs(r["dst"]) < 5]
    far = [r for r in rows if abs(r["dst"]) >= 5]
    shown = collections.Counter(t for r in rows for t in r["shown_types"])
    L = ["# 실제 다른 화자 쌍 (Zeroth 낭독)", "",
         "같은 문장을 다른 사람이 읽은 %d쌍. 정답이 없다 — 지적이 맞는지는 귀로 들어 봐야 한다(reports/)." % len(rows), "",
         "| 묶음 | 쌍 | 정렬↔특징DTW | 정렬↔MFCC-DTW | 특징DTW↔MFCC | 재녹음 | 확신 없음 | 지적된 음절 | 보여줄 지적/발화 | 지적 0 | 종류별(음절당) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    L += block(rows, "전체") + block(near, "음높이 차 <5반음") + block(far, "음높이 차 ≥5반음")
    L += ["", "보여준 지적(발화당 최대 3)의 종류: " + " · ".join("%s %d" % kv for kv in shown.most_common())]
    md = "\n".join(L) + "\n"
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()

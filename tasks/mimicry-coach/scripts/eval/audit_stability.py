# -*- coding: utf-8 -*-
"""서비스 점검 ⑥ — 첫 지적이 사소한 변화에 흔들리나: 같은 녹음 앞뒤에 무음을 붙이거나 음량·코덱만 바꿔도 1순위가 그대로인가.

원본 = 원작 대사(목소리 분리본), 녹음 = 같은 대사 TTS 3개. 변화: 앞 무음 0.15초·0.3초 · 뒤 무음 0.3초 · 음량 −6 dB · opus 32k.
데모와 같은 경로(Coach.compare — 대사 자리 찾기·자르기·경계 교차 확인 포함)로 분석해, 보여 줄 첫 지적(종류·음절 구간)이
변화 전 녹음과 같은 비율을 잰다. 지적이 없으면(통과) "없음"끼리 같다고 본다.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/audit_stability.py --clips /data/speech-ai/out/mimic/original_clips_v1 \\
        --samples /data/speech-ai/out/mimic/sample_clips_v1 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, json, os

import numpy as np

SR = 16000


def perturbations():
    from mimicry import synth
    z = lambda s: np.zeros(int(s * SR), np.float32)
    return {"앞 무음 0.15초": lambda y: np.concatenate([z(0.15), y]),
            "앞 무음 0.3초": lambda y: np.concatenate([z(0.3), y]),
            "뒤 무음 0.3초": lambda y: np.concatenate([y, z(0.3)]),
            "음량 −6 dB": lambda y: (y * 0.5).astype(np.float32),
            "opus 32k": lambda y: synth.codec(y, "opus", 32)}


def top1(C):
    """→ (종류, 첫 음절, 끝 음절, 문장) — 보여 줄 지적이 없으면 ("없음" 또는 재녹음 사유, …)."""
    if not C["quality"]["ok"]:
        return ("재녹음:" + str(C["quality"].get("reason")), None, None, "")
    s = C["selected"]
    if not s:
        return ("없음", None, None, "")
    x = s[0]
    return (x["type"], x.get("i"), x.get("i_end"), x["say"]["what"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="원작 발췌 clips.json 폴더")
    ap.add_argument("--samples", required=True, help="TTS 견본 clips.json 폴더(같은 id 의 대사)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach

    os.makedirs(a.out, exist_ok=True)
    coach = Coach()
    P = perturbations()
    orig = json.load(open(os.path.join(a.clips, "clips.json"), encoding="utf-8"))
    samp = {c["id"]: c for c in json.load(open(os.path.join(a.samples, "clips.json"), encoding="utf-8"))["clips"]}
    rows = []
    for c in orig["clips"]:
        if not (c.get("qc") or {}).get("ok", True) or c["id"] not in samp:
            continue
        xr = load_audio(os.path.join(a.clips, c["files"]["ref"]))
        pre = coach.analyze(xr, c["line"])
        for k, f in sorted(samp[c["id"]]["files"].items()):
            if k not in ("ref", "take_a", "take_b"):
                continue
            y = load_audio(os.path.join(a.samples, f))
            t0 = top1(coach.compare(xr, y, c["line"], ref=pre)[2])
            row = dict(id=c["id"], take=k, orig=t0, pert={})
            for name, fn in P.items():
                row["pert"][name] = top1(coach.compare(xr, fn(y), c["line"], ref=pre)[2])
            rows.append(row)
            print("[%s/%s] %s · 바뀜 %d" % (c["id"], k, t0[0], sum(1 for v in row["pert"].values() if v[:3] != t0[:3])),
                  flush=True)
    json.dump(rows, open(os.path.join(a.out, "stability.json"), "w"), ensure_ascii=False, indent=1)
    md = summarize(rows, list(P))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


def summarize(rows, names):
    same = collections.Counter(int(r["pert"][n][:3] == r["orig"][:3]) for r in rows for n in names)
    L = ["# 서비스 점검 ⑥ 첫 지적이 사소한 변화에 흔들리나", "",
         "원본 = 원작 대사 %d개, 녹음 = TTS %d개. 같은 녹음에 변화를 주고 데모 경로(Coach.compare)로 다시 분석 — 첫 지적(종류·음절 구간)이"
         " 그대로인 비율. 지적 없음·재녹음 사유도 그대로여야 같다고 본다." % (len({r["id"] for r in rows}), len(rows)), "",
         "| 변화 | 1순위 그대로 |", "|---|--:|"]
    for n in names:
        k = sum(1 for r in rows if r["pert"][n][:3] == r["orig"][:3])
        L.append("| %s | %d/%d |" % (n, k, len(rows)))
    tot = same[1] + same[0]
    L += ["| **전체** | **%d/%d = %.0f%%** |" % (same[1], tot, 100.0 * same[1] / max(1, tot)), "", "## 바뀐 것", ""]
    for r in rows:
        for n in names:
            v = r["pert"][n]
            if v[:3] != r["orig"][:3]:
                L.append("- %s/%s · %s: %s → %s" % (r["id"], r["take"], n, r["orig"][3] or r["orig"][0], v[3] or v[0]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""원본 대사 하나와 내 녹음 하나 → 어긋난 곳(최대 3개) · 고치는 법 · 결과 보고서.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/score/compare_take.py \\
        --ref 원본.wav --take 내녹음.m4a --text "니가 가라 하와이" --out 결과폴더

출력: 터미널 요약 · <out>/report.html(음성 내장 — 브라우저로 바로 열린다) · <out>/result.json(음절별 수치 전부).
녹음 형식은 wav/flac 이 아니면 ffmpeg 로 읽는다(폰 녹음 m4a·브라우저 webm 그대로 넣어도 된다).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, time


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k != "tracks"}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="원본 대사 음성")
    ap.add_argument("--take", required=True, help="내 녹음")
    ap.add_argument("--text", required=True, help="대사(한글). 문장부호는 무시, 숫자·영문은 한글로")
    ap.add_argument("--out", help="보고서·JSON 을 쓸 폴더(없으면 터미널 요약만)")
    ap.add_argument("--top", type=int, default=3, help="보여줄 지적 수(계획서: 대사당 최대 3)")
    ap.add_argument("--title", default=None, help="보고서 제목(예: 영화 이름)")
    ap.add_argument("--no-crosscheck", dest="check", action="store_false",
                    help="경계 교차 확인(정렬 vs DTW) 끄기 — 확신 없는 음절도 다 잰다(실험용)")
    a = ap.parse_args()

    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach
    from mimicry.report import html, terminal

    t0 = time.time()
    coach = Coach()
    t1 = time.time()
    xr, xt = load_audio(a.ref), load_audio(a.take)
    A, B, C = coach.compare(xr, xt, a.text, top=a.top, check=a.check)
    t2 = time.time()
    print(terminal(A, B, C))
    print("  (모델 적재 %.1f초 · 분석 %.2f초 / 원본 %.1f초 + 녹음 %.1f초, %s)" % (
        t1 - t0, t2 - t1, len(xr) / 16000.0, len(xt) / 16000.0, coach.al.device))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "report.html"), "w", encoding="utf-8") as f:
            f.write(html(A, B, C, xr, xt, a.title))
        with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(dict(ref=A, take=B, compare=C)), f, ensure_ascii=False, indent=1)
        print("  보고서: %s" % os.path.join(a.out, "report.html"))


if __name__ == "__main__":
    main()

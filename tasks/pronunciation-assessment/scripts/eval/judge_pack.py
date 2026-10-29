# -*- coding: utf-8 -*-
"""측정 묶음의 발화를 **단어 판정(`mdd.word_judge`)** 으로 돌려 처리 시간과 판정을 남긴다 — 장치 간 비교용(results/21).

묶음은 `bench_pronunciation_latency.py --export-pack` 이 만든다(index.json + clips/). 발화마다 문항(단어·문장)을
읽어야 할 말로 넘기고, 반복 측정한 처리 시간 중앙값과 판정(등급 · 편집거리 · 들린 자모열)을 jsonl 로 쓴다.
`--ref` 로 다른 장치의 결과를 주면 판정 일치율을 함께 낸다(예: GPU 결과 ↔ NPU 결과).

    # GPU 서버 — 기준 결과
    $PY .../judge_pack.py --pack <묶음> --hf-ckpt $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 \\
        --device cuda --out <묶음>/ref_cuda.jsonl
    # NPU 서버
    python .../judge_pack.py --pack <묶음> --hf-ckpt <model 폴더> --device npu --rbln <.rbln> --rbln-device <장치> \\
        --ref <묶음>/ref_cuda.jsonl --out npu.jsonl
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, wave

import numpy as np


def read_wav16(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="--export-pack 으로 만든 폴더")
    ap.add_argument("--hf-ckpt", dest="hf_ckpt", required=True)
    ap.add_argument("--device", required=True, choices=["cuda", "cpu", "npu"])
    ap.add_argument("--rbln", default=None)
    ap.add_argument("--rbln-device", dest="rbln_device", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5, help="발화마다 반복 횟수(처리 시간은 중앙값)")
    ap.add_argument("--ref", default=None, help="비교할 다른 장치의 결과 jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from mdd.word_judge import WordJudge
    J = WordJudge(a.hf_ckpt, devices=(a.device,), rbln=a.rbln, rbln_device=a.rbln_device)
    if a.device not in J.decs:
        raise SystemExit("장치를 올리지 못했다: %s" % a.device)
    J.warmup()
    print("[load] 장치 %s · torch 스레드 %d (OMP_NUM_THREADS=%s)"
          % (J.decs[a.device].device, torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS")), flush=True)
    idx = json.load(open(os.path.join(a.pack, "index.json"), encoding="utf-8"))
    rows = []
    for r in idx:
        x = read_wav16(os.path.join(a.pack, r["file"]))
        outs = [J.judge(x, r["text"], a.device) for _ in range(a.repeat)]
        o = outs[-1]
        rows.append({"file": r["file"], "task": r["task"], "sec": round(len(x) / 16000, 2), "text": r["text"],
                     "result": o["result"], "edit": o.get("edit"), "pred": "".join(o.get("pred", [])),
                     "forward_ms": float(np.median([q["ms"].get("forward", 0) for q in outs])),
                     "total_ms": float(np.median([q["ms"]["total"] for q in outs]))})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for task, name in (("RW", "단어"), (None, "긴 발화")):
        R = [r for r in rows if (r["task"] == "RW") == (task == "RW")]
        if R:
            print("[%s %d개 · 평균 %.1f초] 모델 %.1f ms · 판정 전체 %.1f ms (중앙값)"
                  % (name, len(R), np.mean([r["sec"] for r in R]), np.median([r["forward_ms"] for r in R]),
                     np.median([r["total_ms"] for r in R])), flush=True)
    if a.ref:
        ref = dict((d["file"], d) for d in map(json.loads, open(a.ref, encoding="utf-8")))
        both = [r for r in rows if r["file"] in ref]
        same_grade = sum(r["result"] == ref[r["file"]]["result"] for r in both)
        same_pred = sum(r["pred"] == ref[r["file"]]["pred"] for r in both)
        print("[비교 %s] 등급 일치 %d/%d · 들린 자모열 일치 %d/%d"
              % (os.path.basename(a.ref), same_grade, len(both), same_pred, len(both)), flush=True)
        for r in both:
            if r["result"] != ref[r["file"]]["result"]:
                print("  등급 다름 %s %s: %s(%s) ↔ 기준 %s(%s)" % (r["file"], r["text"][:12], r["result"], r["pred"],
                                                            ref[r["file"]]["result"], ref[r["file"]]["pred"]))
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()

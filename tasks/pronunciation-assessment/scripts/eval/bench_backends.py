# -*- coding: utf-8 -*-
"""발음 진단 인코더 — 실행 백엔드별 지연·메모리 측정 (경량화 비교용).

`bench_latency.py` 는 서빙 경로 전체(인코더 + 강제정렬 + 진단)를 잰다. 경량화로 바뀌는 건
**인코더 forward** 뿐이라(정렬·진단은 lprobs 만 받는다), 여기서는 그 구간만 같은 조건으로 잰다.
입력 → lprobs(numpy) 까지를 한 번으로 친다 — 파형 정규화와 장치 간 복사를 포함한다.

백엔드
    torch  cpu  fp32   서빙 경로 그대로 (`gop_feats_v2.forward_lprobs`, MDD_FORCE_CPU=1)
    torch  cuda fp16   서빙 경로 그대로
    ort    cpu         --onnx 로 준 그래프 (fp32 / int8)
    ort    cuda        --onnx 로 준 그래프 (CUDAExecutionProvider)

⚠️ 단독 실행으로 잴 것 — 동시에 돌리면 값이 부풀려진다(docs/ENVIRONMENT.md 함정 9).
   작은 CPU 인스턴스를 흉내 내려면 스레드 수만 줄이지 말고 코어도 묶는다:
       taskset -c 0-3 $PY .../bench_backends.py --backend ort --device cpu --threads 4 --onnx ...

    $PY tasks/pronunciation-assessment/scripts/eval/bench_backends.py --backend torch --device cpu --threads 16 --out a.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()

import argparse, json, os, statistics, time
import numpy as np

SR = 16000


def proc_mem_mb():
    """(현재 RSS, 최대 RSS) MiB — /proc 에서 읽는다(psutil 불필요)."""
    cur = peak = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
    return round(cur), round(peak)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["torch", "ort"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=16, help="CPU intra-op 스레드")
    ap.add_argument("--ckpt", default=os.environ.get("MDD_SERVE_CKPT"), help="torch 백엔드 체크포인트")
    ap.add_argument("--onnx", default=None, help="ort 백엔드 그래프")
    ap.add_argument("--durations", default="3,6,12")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="", help="결과 표에 쓸 이름")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mem0 = proc_mem_mb()
    t0 = time.perf_counter()
    if a.backend == "torch":
        if not a.ckpt:
            ap.error("--ckpt 가 없고 .env 에 MDD_SERVE_CKPT 도 없다")
        if a.device == "cpu":
            os.environ["MDD_FORCE_CPU"] = "1"
        import torch
        torch.set_num_threads(a.threads)
        from mdd import mdd_eval
        from mdd.gop_feats_v2 import forward_lprobs
        mdd_eval.CKPT = a.ckpt
        dec = mdd_eval.Decoder()
        if dec.device != a.device:
            raise SystemExit(f"요청 장치 {a.device} 인데 모델이 {dec.device} 에 올라갔다")

        def run(w):
            lp, Ls = forward_lprobs(dec, [w])            # .float().cpu() 가 GPU 동기화도 겸한다
            return lp[0, :Ls[0]].numpy()

        precision = "fp16" if a.device == "cuda" else "fp32"
        model_file = a.ckpt
    else:
        if not a.onnx:
            ap.error("ort 백엔드는 --onnx 가 필요하다")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if a.device == "cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(a.onnx, so, providers=prov)
        if a.device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
            raise SystemExit(f"CUDA provider 가 안 잡혔다: {sess.get_providers()}")
        in_type = sess.get_inputs()[0].type                   # tensor(float) / tensor(float16)
        in_dt = np.float16 if "float16" in in_type else np.float32

        def run(w):
            x = (w - w.mean()) / np.sqrt(w.var() + 1e-5)      # F.layer_norm(t, t.shape) 와 같은 식
            return sess.run(None, {"source": x[None, :].astype(in_dt)})[0][0].astype(np.float32)

        precision = "fp16" if in_dt == np.float16 else "fp32/int8(그래프 따름)"
        model_file = a.onnx
    load_sec = time.perf_counter() - t0
    mem_loaded = proc_mem_mb()

    gpu_mem = None
    if a.device == "cuda" and a.backend == "torch":
        import torch
        torch.cuda.reset_peak_memory_stats()

    rng = np.random.default_rng(0)
    rows = []
    for dur in [float(x) for x in a.durations.split(",")]:
        w = (rng.standard_normal(int(SR * dur)) * 0.05).astype(np.float32)
        for _ in range(a.warmup):
            run(w)
        ts = []
        for _ in range(a.reps):
            s = time.perf_counter()
            lp = run(w)
            ts.append(time.perf_counter() - s)
        med = statistics.median(ts) * 1000
        rows.append({"sec": dur, "frames": int(lp.shape[0]), "median_ms": round(med, 1),
                     "min_ms": round(min(ts) * 1000, 1), "max_ms": round(max(ts) * 1000, 1),
                     "rtf_x": round(dur * 1000 / med, 1)})
        print(f"  {dur:>4.0f}초  {med:>8.1f} ms  (min {min(ts)*1000:.1f} · max {max(ts)*1000:.1f})  "
              f"실시간 {dur*1000/med:.1f}배", flush=True)

    if a.device == "cuda" and a.backend == "torch":
        import torch
        gpu_mem = round(torch.cuda.max_memory_allocated() / 2**20)

    res = {"tag": a.tag, "backend": a.backend, "device": a.device, "precision": precision,
           "threads": a.threads if a.device == "cpu" else None,
           "cpu_affinity": sorted(os.sched_getaffinity(0)),
           "model_file": model_file, "model_bytes": os.path.getsize(model_file),
           "load_sec": round(load_sec, 1),
           "rss_mb": {"before_load": mem0[0], "after_load": mem_loaded[0], "peak": proc_mem_mb()[1]},
           "torch_gpu_peak_alloc_mb": gpu_mem,
           "reps": a.reps, "rows": rows,
           "_note": "인코더 forward(정규화·복사 포함)만. 입력은 가우스 잡음 — 연산량은 내용과 무관하다."}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"적재 {load_sec:.1f}s · RSS {mem_loaded[0]} MiB (peak {res['rss_mb']['peak']}) → {a.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""국적 분류 최대 역량 — 2B GPU 배치 · ONNX CPU 배치/워커.

`bench_latency.py` 는 1건 지연을 쟀다. 이건 "장비를 최대로 쓰면 몇 건까지 가는가"다.
발음 쪽 `bench_batch.py` + `bench_cpu_workers.py` 의 국적판.

세 모델의 확장 방식이 다르다:
  2B (torch, GPU)     배치를 키운다 — GPU 가 1건으로는 안 채워진다
  300m/KD (ONNX, CPU) 배치는 CPU 에서 거의 안 먹힌다 → 워커 프로세스를 늘린다

2B GPU 배치는 8/7 GPU 종료 전에 확보해야 한다.

    $PY tasks/nationality-classification/scripts/eval/bench_capacity.py --mode gpu2b
    $PY tasks/nationality-classification/scripts/eval/bench_capacity.py --mode onnx_batch
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
from shared.paths import DATA_ROOT, NAT_OUTPUTS, FAIRSEQ_ROOT

import os, json, time, argparse, statistics
import numpy as np
import torch

SR = 16000
ONNX_300M = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/encoder_l17.onnx"
ONNX_KD = f"{DATA_ROOT}/kd_encoder/run2/deploy/encoder_kd_l17.onnx"
CKPT_2B = f"{DATA_ROOT}/models/xlsr2_2B_1000k_inference.pt"


def med(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def gpu_2b(A, w):
    from country_classifier.model import build_model
    enc = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=CKPT_2B,
                      fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).encoder.eval()
    from torch.nn.utils import remove_weight_norm
    for m in enc.modules():
        try: remove_weight_norm(m)
        except (ValueError, AttributeError): pass
    enc = enc.cuda().half()

    rows = []
    print(f"{'배치':>5}{'배치전체_ms':>13}{'발화당_ms':>11}{'초당건수':>10}{'GPU_GB':>9}")
    for B in [int(x) for x in A.batches.split(",")]:
        src = torch.from_numpy(np.stack([w] * B)).cuda().half()
        pm = torch.zeros(B, w.size, dtype=torch.bool, device="cuda")

        def run():
            with torch.inference_mode():
                res = enc.extract_features(source=src, padding_mask=pm, mask=False, layer=31)
            f = res["layer_results"][31][0].transpose(0, 1)
            return f.mean(1)
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
            t = med(lambda: (run(), torch.cuda.synchronize()), A.reps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{B:>5}  메모리 부족 — 여기가 상한"); torch.cuda.empty_cache(); break
            raise
        g = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"배치": B, "배치전체_ms": round(t, 1), "발화당_ms": round(t / B, 1),
                     "초당건수": round(B / (t / 1000), 1), "GPU_GB": round(g, 2)})
        print(f"{B:>5}{t:>13.1f}{t/B:>11.1f}{B/(t/1000):>10.1f}{g:>9.2f}")
        del src, pm
    return rows


def onnx_batch(A, w):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = A.threads
    rows = []
    for name, path in (("300m", ONNX_300M), ("KD", ONNX_KD)):
        if not os.path.exists(path):
            continue
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        print(f"\n[{name} ONNX · {A.threads}스레드]")
        print(f"{'배치':>5}{'배치전체_ms':>13}{'발화당_ms':>11}{'초당건수':>10}")
        for B in [int(x) for x in A.batches.split(",")]:
            src = np.stack([w] * B)
            try:
                t = med(lambda: s.run(["pooled"], {"source": src}), max(2, A.reps // 2), warm=1)
            except Exception as e:
                print(f"{B:>5}  실패 — {str(e)[:70]}")
                break
            rows.append({"모델": name, "배치": B, "배치전체_ms": round(t, 1),
                         "발화당_ms": round(t / B, 1), "초당건수": round(B / (t / 1000), 2)})
            print(f"{B:>5}{t:>13.1f}{t/B:>11.1f}{B/(t/1000):>10.2f}")
        del s
    return rows


def main():
    ap = argparse.ArgumentParser(description="국적 분류 최대 역량")
    ap.add_argument("--mode", default="gpu2b", choices=["gpu2b", "onnx_batch"])
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    torch.set_num_threads(A.threads)
    w = (np.random.default_rng(0).standard_normal(int(SR * A.dur)) * 0.05).astype(np.float32)
    print(f"발화 {A.dur:.0f}초 · mode={A.mode}\n")

    rows = gpu_2b(A, w) if A.mode == "gpu2b" else onnx_batch(A, w)

    if rows:
        best = max(rows, key=lambda r: r["초당건수"])
        print(f"\n최대 {best['초당건수']} 건/초 (배치 {best['배치']}), 사용자 대기 {best['배치전체_ms']:.0f} ms")
    out = A.out or f"{NAT_OUTPUTS}/bench_capacity_{A.mode}.json"
    try:
        json.dump({"_설명": "인코더 forward 만. 헤드는 <0.1ms.", "mode": A.mode,
                   "발화초": A.dur, "측정": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"→ {out}")
    except OSError as e:
        print(f"[저장 실패] {e} — 위 수치를 직접 옮길 것")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""국적 분류 추론 속도 — 2B / 300m ONNX / KD ONNX, GPU·CPU.

발음 쪽 `bench_latency.py` 의 국적판. 서비스 탑재 판단용이라 인코더 forward +
평균풀링 + 선형 헤드까지, 실제 서빙이 하는 일을 그대로 잰다.

세 모델의 실행 방식이 다르다:
  2B        torch (fairseq 체크포인트). GPU fp16 / CPU fp32
  300m ONNX onnxruntime CPU 전용 (설치된 provider 가 CPU·Azure 뿐)
  KD  ONNX  동일

ONNX 는 GPU provider 가 없어 CPU 만 잰다. 2B 는 GPU 종료(8/7) 전에 확보해야 한다.

    $PY tasks/nationality-classification/scripts/eval/bench_latency.py --threads 16
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
from shared.paths import DATA_ROOT, MODEL_ROOT, NAT_OUTPUTS, FAIRSEQ_ROOT

import os, json, time, argparse, statistics
import numpy as np
import torch

SR = 16000
ONNX_300M = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/encoder_l17.onnx"
ONNX_KD = f"{DATA_ROOT}/kd_encoder/run2/deploy/encoder_kd_l17.onnx"
CKPT_2B = f"{DATA_ROOT}/models/xlsr2_2B_1000k_inference.pt"


def bench(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser(description="국적 분류 추론 속도")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--durations", default="3,6,12")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip_2b_cpu", action="store_true", help="2B CPU 는 매우 느리다")
    ap.add_argument("--out", default=f"{NAT_OUTPUTS}/bench_latency.json")
    A = ap.parse_args()

    torch.set_num_threads(A.threads)
    durs = [float(x) for x in A.durations.split(",")]
    rng = np.random.default_rng(0)
    waves = {d: (rng.standard_normal(int(SR * d)) * 0.05).astype(np.float32) for d in durs}
    rows = []

    def add(model, device, d, ms):
        rows.append({"모델": model, "장치": device, "발화초": d,
                     "지연_ms": round(ms, 1), "실시간배수": round(d * 1000 / ms, 1)})
        print(f"  {model:12} {device:6} {d:>4.0f}초  {ms:>9.1f} ms  (실시간 {d*1000/ms:>6.1f}배)", flush=True)

    # ── ONNX 2종 (CPU 전용) ──
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = A.threads
    for name, path in (("300m ONNX", ONNX_300M), ("KD ONNX", ONNX_KD)):
        if not os.path.exists(path):
            print(f"  {name}: 파일 없음 — 건너뜀"); continue
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        for d in durs:
            w = waves[d][None, :]
            add(name, "CPU", d, bench(lambda: s.run(["pooled"], {"source": w}), A.reps))
        del s

    # ── 2B (torch) ──
    if os.path.exists(CKPT_2B):
        from country_classifier.model import build_model
        m = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=CKPT_2B,
                        fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).encoder.eval()
        from torch.nn.utils import remove_weight_norm
        for mod in m.modules():
            try: remove_weight_norm(mod)
            except (ValueError, AttributeError): pass

        def run(model, w, dev, dt):
            s = torch.from_numpy(w).unsqueeze(0).to(device=dev, dtype=dt)
            pm = torch.zeros(1, w.size, dtype=torch.bool, device=dev)
            with torch.inference_mode():
                res = model.extract_features(source=s, padding_mask=pm, mask=False, layer=31)
            f = res["layer_results"][31][0].transpose(0, 1)
            return f.mean(1)

        if torch.cuda.is_available():
            g = m.cuda().half()
            for d in durs:
                add("2B torch", "GPU", d,
                    bench(lambda: (run(g, waves[d], "cuda", torch.float16), torch.cuda.synchronize()), A.reps))
            m = g.cpu().float(); torch.cuda.empty_cache()
        if not A.skip_2b_cpu:
            m = m.cpu().float()
            for d in durs:
                add("2B torch", "CPU", d, bench(lambda: run(m, waves[d], "cpu", torch.float32), max(2, A.reps // 2), warm=1))
    else:
        print(f"  2B: {CKPT_2B} 없음 — 건너뜀")

    res = {"_설명": "인코더 forward + 평균풀링. 헤드(Linear)는 <0.1ms 라 제외.",
           "_주의": "ONNX 는 GPU provider 미설치라 CPU 만. 2B 는 torch.",
           "threads": A.threads, "측정": rows}
    try:
        json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {A.out}")
    except OSError as e:
        print(f"\n[저장 실패] {e} — 위 수치를 직접 옮길 것")


if __name__ == "__main__":
    main()

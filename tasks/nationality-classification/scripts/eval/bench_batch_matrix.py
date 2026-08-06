# -*- coding: utf-8 -*-
"""국적 분류 배치 처리량 격자 — 모델 3종 × 장치 2종 × 배치 크기.

**한 프로세스에서 순차 실행한다.** 벤치마크를 동시에 돌리면 CPU 를 나눠 써서 값이
몇 배씩 부풀려진다(실제로 2B CPU 를 21,254 ms 로 잘못 잰 적 있다 — 실측은 1,832 ms).

ONNX 는 이 격자에 못 들어간다:
  · batch=1 고정으로 수출됨 → 배치 2 이상에서 INVALID_ARGUMENT
  · onnxruntime 이 CPU 빌드 → GPU provider 없음
둘 다 재수출/재설치가 선행돼야 한다.

    $PY tasks/nationality-classification/scripts/eval/bench_batch_matrix.py --batches 128,256
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
from shared.paths import DATA_ROOT, MODEL_ROOT, FAIRSEQ_ROOT, NAT_OUTPUTS

import os, gc, json, time, argparse, statistics, resource
import numpy as np
import torch

SR = 16000
KD_CKPT = f"{DATA_ROOT}/kd_encoder/run2/encoder_kd_best.pt"
CKPT_2B = f"{DATA_ROOT}/models/xlsr2_2B_1000k_inference.pt"


def build(ckpt, state=None):
    from country_classifier.model import build_model
    from torch.nn.utils import remove_weight_norm
    enc = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=ckpt,
                      fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).encoder.eval()
    if state is not None:
        miss, _ = enc.load_state_dict(state, strict=False)
        print(f"    [가중치] 누락 {len(miss)}개 (층 18+ 절단분)", flush=True)
    for m in enc.modules():
        try: remove_weight_norm(m)
        except (ValueError, AttributeError): pass
    return enc


def run_case(enc, layer, dev, B, w, reps):
    dt = torch.float16 if dev == "cuda" else torch.float32
    src = torch.from_numpy(np.stack([w] * B)).to(device=dev, dtype=dt)
    pm = torch.zeros(B, w.size, dtype=torch.bool, device=dev)

    def once():
        with torch.inference_mode():
            r = enc.extract_features(source=src, padding_mask=pm, mask=False, layer=layer)
        out = r["layer_results"][layer][0].transpose(0, 1).mean(1)
        if dev == "cuda":
            torch.cuda.synchronize()
        return out

    once()                                            # 워밍업 1회 (배치가 커서 2회는 낭비)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); once(); ts.append(time.perf_counter() - t)
    mem = (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else \
          (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    del src, pm
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return statistics.median(ts) * 1000, mem


def main():
    ap = argparse.ArgumentParser(description="배치 처리량 격자")
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--batches", default="128,256")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu_reps", type=int, default=3)
    ap.add_argument("--cpu_reps", type=int, default=1)
    ap.add_argument("--out", default=f"{NAT_OUTPUTS}/bench_batch_matrix.json")
    A = ap.parse_args()

    torch.set_num_threads(A.threads)
    Bs = [int(x) for x in A.batches.split(",")]
    w = (np.random.default_rng(0).standard_normal(int(SR * A.dur)) * 0.05).astype(np.float32)

    kd_state = torch.load(KD_CKPT, map_location="cpu", weights_only=False)["encoder"]
    cases = [("2B", CKPT_2B, None, 31), ("300m", f"{MODEL_ROOT}/xlsr2_300m.pt", None, 17),
             ("KD", f"{MODEL_ROOT}/xlsr2_300m.pt", kd_state, 17)]

    rows = []
    print(f"발화 {A.dur:.0f}초 · 배치 {Bs} · 한 프로세스 순차 실행\n")
    print(f"{'모델':>6}{'장치':>6}{'배치':>6}{'전체_ms':>12}{'발화당_ms':>11}{'초당건수':>10}{'메모리_GB':>11}")
    for name, ckpt, state, layer in cases:
        enc = build(ckpt, state)
        for dev in ("cuda", "cpu"):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            enc = (enc.cuda().half() if dev == "cuda" else enc.cpu().float()).eval()
            reps = A.gpu_reps if dev == "cuda" else A.cpu_reps
            for B in Bs:
                try:
                    ms, mem = run_case(enc, layer, dev, B, w, reps)
                except RuntimeError as e:
                    msg = str(e).lower()
                    tag = "메모리 부족" if ("out of memory" in msg or "alloc" in msg) else str(e)[:40]
                    print(f"{name:>6}{dev:>6}{B:>6}   {tag}")
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    continue
                rows.append({"모델": name, "장치": dev.upper(), "배치": B,
                             "전체_ms": round(ms, 1), "발화당_ms": round(ms / B, 2),
                             "초당건수": round(B / (ms / 1000), 1), "메모리_GB": round(mem, 2)})
                print(f"{name:>6}{dev.upper():>6}{B:>6}{ms:>12.1f}{ms/B:>11.2f}"
                      f"{B/(ms/1000):>10.1f}{mem:>11.2f}", flush=True)
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    try:
        json.dump({"_설명": "인코더 forward + 평균풀링. 한 프로세스 순차 실행(오염 방지).",
                   "_제외": "ONNX 는 batch=1 고정 수출 + GPU provider 미설치라 격자 불가.",
                   "발화초": A.dur, "측정": rows}, open(A.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"→ {A.out}")
    except OSError as e:
        print(f"[저장 실패] {e} — 위 수치를 직접 옮길 것")


if __name__ == "__main__":
    main()

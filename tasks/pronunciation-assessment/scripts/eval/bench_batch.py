# -*- coding: utf-8 -*-
"""배치 크기별 인코더 처리량 — GPU 1장이 실제로 몇 건을 감당하는가.

`09_추론속도.md` 는 1건씩 부르는 현재 서버 구성을 쟀다. 그런데 `forward_lprobs` 는
**여러 발화를 한 번에** 받는다. GPU 는 배치를 키우면 처리량이 크게 오르므로,
"배치 큐를 넣을 가치가 있는가"를 판단하려면 이 곡선이 필요하다.

재는 것:
  ① 배치 크기별 발화당 지연과 초당 처리 건수
  ② 배치 전체 지연 (사용자가 기다리는 최악 시간)
  ③ GPU 메모리 사용량 — 배치 상한을 결정한다

주의 — 배치는 **같은 길이로 패딩**된다. 짧은 발화가 긴 발화에 묶이면 낭비가 생기므로,
실서비스에서는 길이별 버킷팅이 필요하다. 여기서는 상한을 보기 위해 동일 길이로 잰다.

    $PY tasks/pronunciation-assessment/scripts/eval/bench_batch.py --device cuda
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
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import os, json, time, argparse, statistics
import numpy as np
import torch

SR = 16000
DEFAULT_CKPT = f"{DATA_ROOT}/mdd_ckpt_tag_all_native/checkpoint_best.pt"


def main():
    ap = argparse.ArgumentParser(description="배치 크기별 처리량")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--dur", type=float, default=6.0, help="발화 길이(초)")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    torch.set_num_threads(A.threads)
    from mdd import mdd_eval as ME
    from mdd import gop_feats_v2 as GF
    ME.CKPT = A.ckpt
    dec = ME.Decoder()
    if A.device == "cuda":
        dec.model = dec.model.cuda().half()
    else:
        dec.model = dec.model.cpu().float()
    dec.model.eval()

    rng = np.random.default_rng(0)
    w = (rng.standard_normal(int(SR * A.dur)) * 0.05).astype(np.float32)

    import torch.nn.functional as TF

    def fwd(wavs):
        if A.device == "cuda":
            return GF.forward_lprobs(dec, wavs)
        mx = max(len(x) for x in wavs)
        src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
        for i, x in enumerate(wavs):
            t = torch.from_numpy(x)
            if dec.normalize:
                t = TF.layer_norm(t, t.shape)
            src[i, :len(x)] = t; pm[i, len(x):] = True
        with torch.inference_mode():
            net = dec.model(source=src, padding_mask=pm)
            lp = dec.model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float()
        return lp, [lp.shape[1]] * len(wavs)

    rows = []
    print(f"발화 {A.dur:.0f}초 · {A.device}"
          f"{f'/{A.threads}스레드' if A.device == 'cpu' else ' fp16'}\n")
    print(f"{'배치':>5}{'배치전체_ms':>13}{'발화당_ms':>11}{'초당건수':>10}{'GPU_GB':>9}")
    for B in [int(x) for x in A.batches.split(",")]:
        wavs = [w] * B
        try:
            if A.device == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
            for _ in range(2):
                fwd(wavs)
            ts = []
            for _ in range(A.reps):
                t0 = time.perf_counter()
                fwd(wavs)
                if A.device == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{B:>5}  메모리 부족 — 여기가 상한")
                if A.device == "cuda":
                    torch.cuda.empty_cache()
                break
            raise
        tot = statistics.median(ts) * 1000
        mem = (torch.cuda.max_memory_allocated() / 2**30) if A.device == "cuda" else 0.0
        rows.append({"배치": B, "배치전체_ms": round(tot, 1),
                     "발화당_ms": round(tot / B, 1), "초당건수": round(B / (tot / 1000), 1),
                     "GPU_GB": round(mem, 2)})
        print(f"{B:>5}{tot:>13.1f}{tot/B:>11.1f}{B/(tot/1000):>10.1f}{mem:>9.2f}")

    if rows:
        base = rows[0]["초당건수"]
        print(f"\n배치 {rows[-1]['배치']} → 처리량 {rows[-1]['초당건수']/base:.1f}배 "
              f"(배치 1 대비), 사용자 대기 {rows[-1]['배치전체_ms']:.0f} ms")

    res = {"_설명": "인코더 forward 만. 정렬·진단은 발화별 CPU 작업이라 배치 이득이 없다.",
           "_주의": "동일 길이 패딩 기준 상한. 실서비스는 길이 버킷팅 필요.",
           "device": A.device, "발화초": A.dur, "ckpt": A.ckpt, "측정": rows}
    out = A.out or f"{MDD_OUTPUTS}/bench_batch_{A.device}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {out}")


if __name__ == "__main__":
    main()

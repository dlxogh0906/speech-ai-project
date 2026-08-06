# -*- coding: utf-8 -*-
"""CPU 최대 처리량 — 워커 프로세스 수 × 워커당 스레드 수 격자 탐색.

CPU 는 배치가 안 먹힌다(1.2배). 코어가 이미 꽉 차 있어 묶어봐야 나눠 쓸 뿐이다.
CPU 에서 처리량을 올리는 유일한 방법은 **프로세스를 여러 개 띄우는 것**이고,
"스레드 16개짜리 1워커"와 "스레드 1개짜리 16워커" 중 어디가 최대인지는 재봐야 안다.

추정하지 않고 실제로 N개 프로세스를 띄워 총 처리량을 잰다.

    $PY tasks/pronunciation-assessment/scripts/eval/bench_cpu_workers.py --cores 16
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import os, json, time, argparse
import multiprocessing as mp

SR = 16000
CKPT = f"{DATA_ROOT}/mdd_ckpt_tag_all_native/checkpoint_best.pt"


def worker(threads, dur, n_utt, ckpt, q, go):
    """프로세스 1개: 모델을 올리고 n_utt 건을 처리한 뒤 소요 시간을 보고한다.

    출발 신호는 Event 로 준다. 같은 Queue 로 주고받으면 주 프로세스가 자기가 넣은
    "go" 를 도로 읽어가는 경합이 생긴다(실제로 겪었다).
    """
    os.environ["MDD_FORCE_CPU"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    _sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    import torch, numpy as np
    torch.set_num_threads(threads)
    from mdd import mdd_eval as ME, gop_feats_v2 as GF
    ME.CKPT = ckpt
    dec = ME.Decoder()
    w = (np.random.default_rng(0).standard_normal(int(SR * dur)) * 0.05).astype("float32")
    GF.forward_lprobs(dec, [w])                     # 워밍업
    q.put(("ready", os.getpid()))
    go.wait()                                        # 동시 출발 신호를 기다린다
    t0 = time.perf_counter()
    for _ in range(n_utt):
        GF.forward_lprobs(dec, [w])
    q.put(("done", time.perf_counter() - t0))


def main():
    ap = argparse.ArgumentParser(description="CPU 워커 격자 탐색")
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--n_utt", type=int, default=4, help="워커당 처리 건수")
    ap.add_argument("--grid", default="1,2,4,8,16", help="워커 수 목록")
    ap.add_argument("--out", default=f"{MDD_OUTPUTS}/bench_cpu_workers.json")
    A = ap.parse_args()

    ctx = mp.get_context("spawn")
    rows = []
    print(f"코어 {A.cores} · 발화 {A.dur:.0f}초 · 워커당 {A.n_utt}건\n")
    print(f"{'워커':>5}{'워커당스레드':>13}{'워커당_ms':>11}{'총_건/초':>10}{'RAM_GB':>9}")
    for W in [int(x) for x in A.grid.split(",")]:
        th = max(1, A.cores // W)
        q = ctx.Queue(); go = ctx.Event()
        ps = [ctx.Process(target=worker, args=(th, A.dur, A.n_utt, CKPT, q, go)) for i in range(W)]
        for p in ps:
            p.start()
        for _ in range(W):                           # 전부 준비될 때까지 (적재 시간 제외)
            q.get()
        t0 = time.perf_counter()
        go.set()
        durs = [q.get()[1] for _ in range(W)]
        wall = time.perf_counter() - t0
        for p in ps:
            p.join()
        total = W * A.n_utt
        per = sum(durs) / len(durs) / A.n_utt * 1000
        rows.append({"워커": W, "워커당스레드": th, "워커당_ms": round(per, 1),
                     "총_건per초": round(total / wall, 2),
                     "RAM_GB": round(W * 5.17, 1)})
        print(f"{W:>5}{th:>13}{per:>11.1f}{total/wall:>10.2f}{W*5.17:>9.1f}")

    best = max(rows, key=lambda r: r["총_건per초"])
    print(f"\n최대 {best['총_건per초']} 건/초 — 워커 {best['워커']} × 스레드 {best['워커당스레드']} "
          f"(RAM {best['RAM_GB']} GB)")
    json.dump({"_설명": "인코더 forward 기준. 프로세스를 실제로 띄워 총 처리량을 쟀다.",
               "코어": A.cores, "발화초": A.dur, "측정": rows, "최대": best},
              open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()

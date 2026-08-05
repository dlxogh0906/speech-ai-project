# -*- coding: utf-8 -*-
"""발음 진단 추론 속도 측정 — CPU/GPU, 구간별 병목, 동시 요청 한계.

서비스 탑재 가능 여부를 판단하기 위한 것이라 **실제 서빙 경로**(`deploy/mdd-api` 의
analyze → diagnose → 점수 특징)를 그대로 태운다. 마이크로벤치가 아니다.

재는 것:
  ① 발화 길이별 지연 (3 / 6 / 12초)
  ② 구간별 분해 — 인코더 forward / 강제정렬 / 특징·진단
  ③ CPU 스레드 수에 따른 변화 (동시 요청 한계의 근거)
  ④ GPU 대비 배수

GPU 는 `--device cuda`. 인코더가 단일 GPU 라 서버는 lock 으로 직렬화하므로,
동시 요청 처리량은 "1건 지연의 역수"가 상한이다 — 이 스크립트는 그 상한을 잰다.

    $PY tasks/pronunciation-assessment/scripts/eval/bench_latency.py --device cpu --threads 4
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

import os, json, time, argparse, itertools, statistics
import numpy as np
import torch
import editdistance

SR = 16000
DEFAULT_CKPT = f"{DATA_ROOT}/mdd_ckpt_tag_all_native/checkpoint_best.pt"


def main():
    ap = argparse.ArgumentParser(description="발음 진단 추론 속도 측정")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=4, help="CPU 스레드 수")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--scorer", default=f"{MDD_OUTPUTS}/score_models_v2_native.joblib")
    ap.add_argument("--durations", default="3,6,12")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    torch.set_num_threads(A.threads)
    import joblib
    from mdd import mdd_eval as ME
    from mdd import gop_feats_v2 as GF
    from mdd import pipeline as MD
    ME.CKPT = A.ckpt

    t0 = time.perf_counter()
    dec = ME.Decoder()
    if A.device == "cuda":
        dec.model = dec.model.cuda()
    else:
        dec.model = dec.model.cpu()
    dec.model.eval()
    bundle = joblib.load(A.scorer)
    load_sec = time.perf_counter() - t0

    # 표준발음은 실제 문장에서 뽑는다 — 길이에 비례해 늘려야 정렬 비용이 현실적이다
    import g2p
    MD.S["ri"], MD.S["ro"] = g2p.readRules(g2p.ver_info[0], MD.RULEBOOK)
    BASE = "동물을 키울 수 있는 크고 좋은 집에서 살고 싶어"

    # `GF.forward_lprobs` 는 `.half().cuda()` 가 박혀 있다(단일 GPU 서빙 전제).
    # CPU 경로는 fp16 이 느리거나 미지원이라 fp32 로 돈다 — 라이브러리를 건드리면
    # GPU 쪽 수치가 바뀌므로 여기서만 등가 구현을 쓴다.
    import torch.nn.functional as TF

    def forward_cpu(dec, wavs):
        lens = [len(w) for w in wavs]; mx = max(lens)
        src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
        for i, w in enumerate(wavs):
            t = torch.from_numpy(w)
            if dec.normalize:
                t = TF.layer_norm(t, t.shape)
            src[i, :len(w)] = t; pm[i, len(w):] = True
        with torch.inference_mode():          # GPU 경로는 Decoder 안에서 이미 꺼져 있다
            net = dec.model(source=src, padding_mask=pm)
            lp = dec.model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float()
        opm = net.get("padding_mask")
        Ls = [(int((~opm[b]).sum()) if opm is not None else lp.shape[1]) for b in range(len(wavs))]
        return lp, Ls

    if A.device == "cuda":
        dec.model = dec.model.half()
        forward = GF.forward_lprobs
    else:
        dec.model = dec.model.float()
        forward = forward_cpu

    rng = np.random.default_rng(0)
    rows = []
    for dur in [float(x) for x in A.durations.split(",")]:
        # 문장을 반복해 길이에 맞춘다 (12초 발화면 표준발음도 그만큼 길다)
        reps_txt = max(1, int(round(dur / 4.0)))
        canon, _ = MD.canonical(" ".join([BASE] * reps_txt))
        y = [dec.d.index(t) for t in canon]
        w = (rng.standard_normal(int(SR * dur)) * 0.05).astype(np.float32)

        for _ in range(2):                                   # 워밍업
            forward(dec, [w])

        t_fwd, t_align, t_rest = [], [], []
        for _ in range(A.reps):
            a = time.perf_counter()
            lp, Ls = forward(dec, [w])
            if A.device == "cuda":
                torch.cuda.synchronize()
            b = time.perf_counter()
            l = lp[0, :Ls[0]].detach().numpy()
            segs = GF.viterbi_align(l, y, blank=dec.blank)
            c = time.perf_counter()
            g = l.argmax(-1)
            pred_idx = [int(v) for v, _ in itertools.groupby(g) if v != dec.blank]
            editdistance.eval(pred_idx, y)
            if segs is not None:
                for si, fs in enumerate(segs):
                    if fs:
                        fa = np.asarray(fs)
                        float((l[fa, y[si]] - l[fa].max(-1)).mean())
            d = time.perf_counter()
            t_fwd.append(b - a); t_align.append(c - b); t_rest.append(d - c)

        med = lambda v: round(statistics.median(v) * 1000, 1)
        tot = med(t_fwd) + med(t_align) + med(t_rest)
        rows.append({
            "발화초": dur, "음소수": len(canon), "프레임수": int(l.shape[0]),
            "인코더_ms": med(t_fwd), "강제정렬_ms": med(t_align), "진단·특징_ms": med(t_rest),
            "합계_ms": round(tot, 1), "실시간배수": round(dur * 1000 / tot, 1),
        })
        print(f"  {dur:>4.0f}초  인코더 {med(t_fwd):>7.1f}  정렬 {med(t_align):>6.1f}  "
              f"진단 {med(t_rest):>6.1f}  = {tot:>7.1f} ms  (실시간 {dur*1000/tot:.1f}배)", flush=True)

    res = {"_설명": "실제 서빙 경로(analyze) 기준. 점수 GBM 은 세션 단위라 여기 없음(발화당 <1ms).",
           "device": A.device, "threads": A.threads if A.device == "cpu" else None,
           "ckpt": A.ckpt, "모델적재_초": round(load_sec, 1), "reps": A.reps, "측정": rows}

    out = A.out or f"{MDD_OUTPUTS}/bench_latency_{A.device}{A.threads if A.device=='cpu' else ''}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n모델 적재 {load_sec:.1f}초 · {A.device}"
          f"{f'/{A.threads}스레드' if A.device=='cpu' else ''}\n→ {out}")


if __name__ == "__main__":
    main()

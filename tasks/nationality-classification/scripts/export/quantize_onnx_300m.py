# -*- coding: utf-8 -*-
"""encoder_l17.onnx → int8 동적 양자화 + 서버-정확 레퍼런스로 수치검증 + 속도벤치."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, FFMPEG, MODEL_ROOT, NAT_OUTPUTS, REPO_ROOT

import subprocess, sys, time
import numpy as np

FP32 = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/encoder_l17.onnx"
INT8 = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/encoder_l17_int8.onnx"
HEAD = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/head.pt"
SR = 16000; LAYER = 17
FFMPEG = FFMPEG


def main():
    import onnx, os
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
    import torch, torch.nn as nn

    # ---- int8 동적 양자화 (Linear/MatMul 가중치 int8) ----
    print("[quantize] dynamic int8 (MatMul만; conv는 fp32 유지 — ConvInteger 회피) ...", flush=True)
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print(f"[quantize] {os.path.getsize(FP32)/1e6:.0f}MB -> {os.path.getsize(INT8)/1e6:.0f}MB")

    # ---- 서버-정확 레퍼런스 (mult=2 + padding_mask + 마스크평균) ----
    from country_classifier.model import build_model
    ref = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=f"{MODEL_ROOT}/xlsr2_300m.pt",
                      fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).encoder.eval()
    from torch.nn.utils import remove_weight_norm
    for m in ref.modules():
        try: remove_weight_norm(m)
        except (ValueError, AttributeError): pass
    def ref_pooled(w):
        s = torch.from_numpy(w).unsqueeze(0); pm = torch.zeros(1, w.size, dtype=torch.bool)
        with torch.inference_mode():
            res = ref.extract_features(source=s, padding_mask=pm, mask=False, layer=LAYER)
        f = res["layer_results"][LAYER][0].transpose(0, 1); fpm = res.get("padding_mask")
        if fpm is None: return f.mean(1).numpy()
        v = (~fpm).unsqueeze(-1).float(); return ((f*v).sum(1)/v.sum(1).clamp_min(1.)).numpy()

    def mk_sess(p):
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    s32, s8 = mk_sess(FP32), mk_sess(INT8)

    # ---- head (top5 비교용) ----
    d = torch.load(HEAD, map_location="cpu", weights_only=False)
    head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
    mu = torch.from_numpy(np.asarray(d["mu"], np.float32)); sd = torch.from_numpy(np.asarray(d["sd"], np.float32))
    labels = list(d["labels"])
    def classify(p):
        x = torch.from_numpy(p); x = torch.nan_to_num(x).clamp(-50, 50).half().float(); x = (x-mu)/sd
        pr = head(x).softmax(-1)[0]; i = pr.topk(5).indices.tolist()
        return [(labels[j], round(float(pr[j]), 4)) for j in i]

    # ---- 수치검증: 41개 클래스 샘플 wav 전부 top1 일치율 ----
    import glob
    samples = sorted(glob.glob(f"{REPO_ROOT}/deploy/nationality-latency-demo/class_samples/*.wav"))
    print(f"\n[verify] {len(samples)}개 샘플 top1 일치 검사 (ref=서버정확)")
    feat_err8 = []; top1_ref, top1_8, top1_32 = [], [], []
    for wp in samples:
        raw = subprocess.run([FFMPEG,"-hide_banner","-loglevel","error","-i",wp,"-f","f32le","-ac","1","-ar",str(SR),"-t","12","pipe:1"],
                             stdout=subprocess.PIPE).stdout
        w = np.frombuffer(raw, np.float32).copy()
        if w.size < SR//2: continue
        r = ref_pooled(w); b32 = s32.run(["pooled"], {"source": w[None,:]})[0]; b8 = s8.run(["pooled"], {"source": w[None,:]})[0]
        feat_err8.append(np.abs(r-b8).max())
        top1_ref.append(classify(r)[0][0]); top1_32.append(classify(b32)[0][0]); top1_8.append(classify(b8)[0][0])
    n = len(top1_ref)
    m32 = sum(a==b for a,b in zip(top1_ref,top1_32))
    m8 = sum(a==b for a,b in zip(top1_ref,top1_8))
    print(f"  fp32 ONNX  top1 == 서버ref : {m32}/{n}")
    print(f"  int8 ONNX  top1 == 서버ref : {m8}/{n}   (int8 feature max|Δ| 중앙 {np.median(feat_err8):.3f})")
    mism = [(s.split('/')[-1][:-4], r, e) for s,r,e in zip(samples,top1_ref,top1_8) if r!=e]
    if mism: print("  int8 top1 불일치:", mism)

    # ---- 속도 벤치 (12초) ----
    w12 = (np.random.default_rng(1).standard_normal(SR*12)*0.05).astype(np.float32)[None,:]
    for _ in range(2): s32.run(["pooled"],{"source":w12}); s8.run(["pooled"],{"source":w12})
    def bench(sess, N=8):
        t=time.perf_counter()
        for _ in range(N): sess.run(["pooled"],{"source":w12})
        return (time.perf_counter()-t)/N*1000
    t32, t8 = bench(s32), bench(s8)
    print(f"\n[bench 12s, 4 threads]  ONNX fp32 {t32:.0f} ms  |  ONNX int8 {t8:.0f} ms  |  int8 speedup {t32/t8:.2f}x")
    print(f"[전체 비교] PyTorch fp32 ~2045 | PyTorch int8 ~1280 | ONNX fp32 {t32:.0f} | ONNX int8 {t8:.0f} ms")


if __name__ == "__main__":
    main()

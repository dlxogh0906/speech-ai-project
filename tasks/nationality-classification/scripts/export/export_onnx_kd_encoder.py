# -*- coding: utf-8 -*-
"""KD 인코더(run2 최종) → ONNX 수출 + pt↔onnx 패리티 검증. CPU 배포용.

export_onnx_300m.py 와 같은 규격: ONNX 는 파형(1,T)→pooled(1,1024)만 담당,
clamp/(x−mu)/sd/head/softmax 는 서버 파이썬에서 수행(수치 민감부를 그래프 밖에).

⚠️ 순서 중요: state_dict 로드 → remove_weight_norm (학습 시 weight_norm 유지 상태로 저장됨)

출력: <ckpt 디렉터리>/deploy/ 에 encoder_kd_l17.onnx + head_np.npz (서버가 읽는 번들)

    $PY tasks/nationality-classification/scripts/export/export_onnx_kd_encoder.py
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
from shared.paths import FFMPEG, NAT_TASK, REPO_ROOT

import argparse, os, subprocess, time
import numpy as np

SR = 16000


def patch_pad_to_multiple():
    import fairseq.models.wav2vec.wav2vec2 as w2
    def safe(x, multiple, dim=-1, value=0):
        if x is None or multiple <= 1:
            return x, 0
        tsz = x.size(dim); remainder = (multiple - tsz % multiple) % multiple
        if remainder == 0:
            return x, 0
        pad = [0, 0] * x.dim(); pad[(x.dim() - 1 - (dim % x.dim())) * 2 + 1] = remainder
        import torch.nn.functional as F
        return F.pad(x, pad, value=value), remainder
    w2.pad_to_multiple = safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(NAT_TASK / "outputs/kd_encoder/run2/encoder_kd_best.pt"))
    ap.add_argument("--xlsr300", default=os.path.expanduser("~/models/xlsr2_300m.pt"))
    A = ap.parse_args()
    out_dir = _pl.Path(A.ckpt).parent / "deploy"
    out_dir.mkdir(exist_ok=True)
    onnx_out = str(out_dir / "encoder_kd_l17.onnx")

    import torch, torch.nn as nn
    from fairseq import checkpoint_utils
    patch_pad_to_multiple()

    ck = torch.load(A.ckpt, map_location="cpu")
    LAYER = int(ck["layer"])

    def build(load_kd=True):
        models, _ = checkpoint_utils.load_model_ensemble([A.xlsr300])
        enc = models[0]
        del enc.encoder.layers[LAYER + 1:]
        if load_kd:
            enc.load_state_dict(ck["encoder"])           # ← weight_norm 제거 '전'에 로드
        enc.eval()
        return enc

    enc = build()
    for mod in enc.modules():
        if hasattr(mod, "required_seq_len_multiple"):
            mod.required_seq_len_multiple = 1
    from torch.nn.utils import remove_weight_norm
    for mod in enc.modules():
        try: remove_weight_norm(mod)
        except (ValueError, AttributeError): pass

    class Wrap(nn.Module):
        def __init__(self, e): super().__init__(); self.enc = e
        def forward(self, source):
            res = self.enc.extract_features(source=source, padding_mask=None, mask=False, layer=LAYER)
            return res["layer_results"][LAYER][0].transpose(0, 1).mean(dim=1)   # (1,1024)

    wrap = Wrap(enc).eval()

    # 서버-정확 레퍼런스 (배수 기본 + 마스크 평균, weight_norm 제거만)
    ref = build()
    for mod in ref.modules():
        try: remove_weight_norm(mod)
        except (ValueError, AttributeError): pass

    def ref_pooled(w):
        s = torch.from_numpy(w).unsqueeze(0)
        pm = torch.zeros(1, w.size, dtype=torch.bool)
        with torch.inference_mode():
            res = ref.extract_features(source=s, padding_mask=pm, mask=False, layer=LAYER)
        f = res["layer_results"][LAYER][0].transpose(0, 1)
        fpm = res.get("padding_mask")
        if fpm is None: return f.mean(1).numpy()
        v = (~fpm).unsqueeze(-1).float()
        return ((f * v).sum(1) / v.sum(1).clamp_min(1.0)).numpy()

    rng = np.random.default_rng(0)
    w0 = (rng.standard_normal(SR * 11) * 0.05).astype(np.float32)
    with torch.inference_mode():
        p0 = wrap(torch.from_numpy(w0).unsqueeze(0)).numpy()
    print(f"[규격] wrap vs ref  max|Δ|={np.abs(p0-ref_pooled(w0)).max():.2e}")

    print("[export] tracing...", flush=True)
    torch.onnx.export(wrap, (torch.from_numpy(w0).unsqueeze(0),), onnx_out,
                      input_names=["source"], output_names=["pooled"],
                      dynamic_axes={"source": {1: "samples"}},
                      opset_version=14, do_constant_folding=True)
    import onnx
    onnx.checker.check_model(onnx_out)
    print(f"[export] OK → {onnx_out} ({os.path.getsize(onnx_out)/1e6:.1f} MB)")

    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_out, so, providers=["CPUExecutionProvider"])
    worst = 0
    for sec in [11, 6, 3, 12]:
        w = (rng.standard_normal(SR * sec) * 0.05).astype(np.float32)
        d = np.abs(ref_pooled(w) - sess.run(["pooled"], {"source": w[None, :]})[0])
        worst = max(worst, d.max())
        print(f"[verify {sec:>2}s] max|Δ|={d.max():.2e} mean|Δ|={d.mean():.2e}")
    assert worst < 1e-2, "패리티 실패 — 배포 금지"

    # 헤드를 서버용 npz 로 (torch 없는 서버 지원)
    head = ck["head"]
    np.savez(out_dir / "head_np.npz",
             W=head["weight"].numpy().astype(np.float32),
             b=head["bias"].numpy().astype(np.float32),
             mu=np.asarray(ck["mu"], np.float32), sd=np.asarray(ck["sd"], np.float32),
             labels=np.array(list(ck["labels"]), dtype=object))
    print(f"[head] → {out_dir/'head_np.npz'}")

    # 실발화 top1 pt↔onnx 일치 검사
    import glob
    labels = list(ck["labels"])
    W = head["weight"].numpy(); b = head["bias"].numpy()
    mu = np.asarray(ck["mu"], np.float32); sd = np.asarray(ck["sd"], np.float32)
    def top1(pooled):
        x = np.clip(np.nan_to_num(pooled), -50, 50)
        z = ((x - mu) / sd) @ W.T + b
        return labels[int(z.argmax())]
    agree = tot = 0
    for wav in sorted(glob.glob("/home/tta/test_samples/*.wav"))[:10]:
        raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", wav,
                              "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", "12", "pipe:1"],
                             stdout=subprocess.PIPE).stdout
        w = np.frombuffer(raw, np.float32).copy()
        if w.size < SR: continue
        with torch.inference_mode():
            pt = wrap(torch.from_numpy(w).unsqueeze(0)).numpy()
        on = sess.run(["pooled"], {"source": w[None, :]})[0]
        tot += 1; agree += (top1(pt) == top1(on))
        print(f"  {os.path.basename(wav)[:34]:34} pt={top1(pt):12} onnx={top1(on):12} {'✓' if top1(pt)==top1(on) else '✗'}")
    print(f"[실발화] top1 일치 {agree}/{tot}")

    # 속도
    w12 = (rng.standard_normal(SR * 12) * 0.05).astype(np.float32)
    for _ in range(2): sess.run(["pooled"], {"source": w12[None, :]})
    t = time.perf_counter()
    for _ in range(5): sess.run(["pooled"], {"source": w12[None, :]})
    print(f"[bench] ONNX CPU(4스레드) 12s 발화: {(time.perf_counter()-t)/5*1000:.0f} ms")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""XLS-R 300M encoder → ONNX (layer-17 mean-pool feature까지). CPU 배포용.

설계: ONNX는 raw 파형(1,T) → pooled fp32 (1,1024) 만 담당.
그 뒤 clamp/fp16 라운드트립/(x-mu)/sd/head/softmax 는 Python에서 서버(_classify)와 동일.
→ 수치 민감부를 ONNX 밖에 둬 규격 불변 보장.

검증: PyTorch pooled vs ONNX pooled 오차, 그리고 최종 top5 일치 + 속도.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, FFMPEG, MODEL_ROOT, NAT_OUTPUTS, REPO_ROOT

import sys, time
import numpy as np

CKPT = f"{MODEL_ROOT}/xlsr2_300m.pt"
HEAD = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/head.pt"
ONNX_OUT = f"{NAT_OUTPUTS}/0705_country41_sent043_300m/encoder_l17.onnx"
LAYER = 17
SR = 16000


def _patch_pad_to_multiple():
    """트레이스-안전 pad_to_multiple: multiple<=1이면 즉시 반환(동적 control flow 회피).
    required_seq_len_multiple=1과 함께 쓰면 패딩이 아예 없어 서버(배수2+마스크평균)와 동일."""
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


def load_encoder():
    import torch
    from country_classifier.model import build_model
    _patch_pad_to_multiple()
    m = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=CKPT,
                    fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True)
    enc = m.encoder
    enc.eval()
    # 패딩 배수 1 → pad_to_multiple no-op → 트레이스 안전 + 서버와 수치 동일
    n_set = 0
    for mod in enc.modules():
        if hasattr(mod, "required_seq_len_multiple"):
            mod.required_seq_len_multiple = 1; n_set += 1
    print(f"[patch] required_seq_len_multiple=1 on {n_set} module(s)")
    from torch.nn.utils import remove_weight_norm
    for mod in enc.modules():
        try: remove_weight_norm(mod)
        except (ValueError, AttributeError): pass
    return enc


class PoolWrapper:
    pass


def main():
    import torch, torch.nn as nn
    enc = load_encoder()

    class Wrap(nn.Module):
        def __init__(self, enc):
            super().__init__(); self.enc = enc
        def forward(self, source):
            # padding_mask=None (배치1·무패딩 추론 규격) → mask 분기 회피
            res = self.enc.extract_features(source=source, padding_mask=None, mask=False, layer=LAYER)
            feats = res["layer_results"][LAYER][0]          # (T', 1, C)
            feats = feats.transpose(0, 1)                   # (1, T', C)
            return feats.mean(dim=1)                        # (1, C)

    wrap = Wrap(enc).eval()

    # 서버-정확 레퍼런스: 별도 encoder(패딩배수 기본=2) + padding_mask + 마스크평균 (server _pool 동일)
    from country_classifier.model import build_model as _bm
    ref = _bm(backend="fairseq_xlsr", num_classes=41, checkpoint_path=CKPT,
              fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).encoder.eval()
    from torch.nn.utils import remove_weight_norm as _rwn
    for mod in ref.modules():
        try: _rwn(mod)
        except (ValueError, AttributeError): pass
    def ref_pooled(wav_np):
        s = torch.from_numpy(wav_np).unsqueeze(0)
        pm = torch.zeros(1, wav_np.size, dtype=torch.bool)
        with torch.inference_mode():
            res = ref.extract_features(source=s, padding_mask=pm, mask=False, layer=LAYER)
        feats = res["layer_results"][LAYER][0].transpose(0, 1)      # (1,T',C)
        fpm = res.get("padding_mask")
        if fpm is None:
            return feats.mean(1).numpy()
        v = (~fpm).unsqueeze(-1).float()
        return ((feats * v).sum(1) / v.sum(1).clamp_min(1.0)).numpy()

    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(SR * 11) * 0.05).astype(np.float32)
    src = torch.from_numpy(wav).unsqueeze(0)
    with torch.inference_mode():
        pt_pooled = wrap(src).numpy()
    r0 = ref_pooled(wav)
    print(f"[baseline] wrap(mult=1) vs 서버ref(mult=2+마스크): max|Δ|={np.abs(pt_pooled-r0).max():.2e} (0에 가까워야 = 규격동일)")

    # ---- ONNX export ----
    print("[export] onnx export ...", flush=True)
    torch.onnx.export(
        wrap, (src,), ONNX_OUT,
        input_names=["source"], output_names=["pooled"],
        dynamic_axes={"source": {1: "samples"}},
        opset_version=14, do_constant_folding=True,
    )
    import onnx
    onnx.checker.check_model(ONNX_OUT)
    import os
    print(f"[export] OK -> {ONNX_OUT} ({os.path.getsize(ONNX_OUT)/1e6:.1f} MB)")

    # ---- ONNX Runtime 로 재현 검증 (여러 길이) ----
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(ONNX_OUT, so, providers=["CPUExecutionProvider"])
    for sec in [11, 6, 3, 12]:
        w = (rng.standard_normal(SR * sec) * 0.05).astype(np.float32)
        b = sess.run(["pooled"], {"source": w[None, :]})[0]      # ONNX
        r = ref_pooled(w)                                        # 서버-정확 레퍼런스
        d = np.abs(r - b)
        print(f"[verify {sec:>2}s] ONNX vs 서버ref  max|Δ|={d.max():.2e} mean|Δ|={d.mean():.2e}  (< 1e-3 면 동일)")

    # ---- 최종 top5 일치 (head까지) ----
    d = torch.load(HEAD, map_location="cpu", weights_only=False)
    head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
    mu = torch.from_numpy(np.asarray(d["mu"], np.float32)); sd = torch.from_numpy(np.asarray(d["sd"], np.float32))
    labels = list(d["labels"])
    def classify(pooled_np):
        x = torch.from_numpy(pooled_np)
        x = torch.nan_to_num(x).clamp(-50, 50).half().float()
        x = (x - mu) / sd
        p = head(x).softmax(-1)[0]
        return [(labels[i], float(p[i])) for i in p.topk(5).indices.tolist()]
    # 실제 샘플 wav 로 (Iran.wav)
    import subprocess
    wavp = f"{REPO_ROOT}/deploy/nationality-latency-demo/class_samples/Iran.wav"
    raw = subprocess.run([FFMPEG,"-hide_banner","-loglevel","error",
                          "-i",wavp,"-f","f32le","-ac","1","-ar",str(SR),"-t","12","pipe:1"],
                         stdout=subprocess.PIPE).stdout
    w = np.frombuffer(raw, np.float32).copy()
    with torch.inference_mode():
        pt = wrap(torch.from_numpy(w).unsqueeze(0)).numpy()
    on = sess.run(["pooled"], {"source": w[None, :]})[0]
    print(f"\n[Iran.wav] PyTorch top5: {[(l,round(p,3)) for l,p in classify(pt)][:3]}")
    print(f"[Iran.wav] ONNX    top5: {[(l,round(p,3)) for l,p in classify(on)][:3]}")

    # ---- 속도 벤치 (12초 파형) ----
    w12 = (rng.standard_normal(SR * 12) * 0.05).astype(np.float32)
    s12 = torch.from_numpy(w12).unsqueeze(0)
    torch.set_num_threads(4)
    for _ in range(2):  # warmup
        with torch.inference_mode(): wrap(s12)
        sess.run(["pooled"], {"source": w12[None, :]})
    N = 5
    t = time.perf_counter()
    for _ in range(N):
        with torch.inference_mode(): wrap(s12)
    pt_ms = (time.perf_counter() - t) / N * 1000
    t = time.perf_counter()
    for _ in range(N):
        sess.run(["pooled"], {"source": w12[None, :]})
    on_ms = (time.perf_counter() - t) / N * 1000
    print(f"\n[bench 12s, 4 threads] PyTorch CPU {pt_ms:.0f} ms  |  ONNX CPU {on_ms:.0f} ms  |  speedup {pt_ms/on_ms:.2f}x")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""KD 인코더 → 고정 길이 + 프레임 마스크 TorchScript(.pt). Rebellions SDK 는 torch/TorchScript 프런트엔드만 있어(compile_from_torchscript),
NPU 서버에 fairseq 없이 컴파일하려면 이 파일을 보낸다. 규약은 export_onnx_kd_static.py 와 같다: source f32 [1,N] · frame_mask bool [1,T] → pooled [1,1024].

    $PY tasks/nationality-classification/scripts/export/export_torchscript_kd_static.py --ckpt <run>/ep0.pt --xlsr300 <xlsr2_300m.pt> [--sec 12]
    → <run>/deploy/encoder_kd_l17_static<sec>s.ts.pt  (같은 폴더의 정적 ONNX 와 출력 대조)
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src"))), str(_pl.Path(__file__).resolve().parent)]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
import argparse, os, numpy as np

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--xlsr300", required=True); ap.add_argument("--sec", type=int, default=12)
    A = ap.parse_args()
    out_dir = _pl.Path(A.ckpt).parent / "deploy"; out_dir.mkdir(exist_ok=True)
    N = SR * A.sec; ts_out = str(out_dir / f"encoder_kd_l17_static{A.sec}s.ts.pt")
    import torch, torch.nn as nn
    from fairseq import checkpoint_utils
    from export_onnx_kd_encoder import patch_pad_to_multiple
    patch_pad_to_multiple()
    import fairseq.models.wav2vec.wav2vec2 as w2
    w2.index_put = lambda tensor, indices, value: tensor.masked_fill(indices.unsqueeze(-1), value)
    ck = torch.load(A.ckpt, map_location="cpu"); LAYER = int(ck["layer"])
    models, _ = checkpoint_utils.load_model_ensemble([A.xlsr300]); enc = models[0]
    del enc.encoder.layers[LAYER + 1:]; enc.load_state_dict(ck["encoder"]); enc.eval()
    for mod in enc.modules():
        if hasattr(mod, "required_seq_len_multiple"): mod.required_seq_len_multiple = 1
    from torch.nn.utils import remove_weight_norm
    for mod in enc.modules():
        try: remove_weight_norm(mod)
        except (ValueError, AttributeError): pass

    class Wrap(nn.Module):
        def __init__(self, e): super().__init__(); self.enc = e
        def forward(self, source, frame_mask):
            f = self.enc.feature_extractor(source).transpose(1, 2)
            f = self.enc.layer_norm(f)
            if self.enc.post_extract_proj is not None: f = self.enc.post_extract_proj(f)
            _, lr = self.enc.encoder(f, padding_mask=frame_mask, layer=LAYER)
            x = lr[LAYER][0].transpose(0, 1)
            v = (~frame_mask).to(x.dtype).unsqueeze(-1)
            return (x * v).sum(1) / v.sum(1).clamp_min(1.0)
    wrap = Wrap(enc).eval()

    def conv_out_len(n):
        for k, st in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)): n = (n - k) // st + 1
        return n
    T = conv_out_len(N)
    rng = np.random.default_rng(0)
    ex_src = torch.from_numpy((rng.standard_normal(N) * 0.05).astype(np.float32))[None]
    ex_mask = torch.zeros(1, T, dtype=torch.bool); ex_mask[:, conv_out_len(SR * (A.sec - 4)):] = True
    with torch.inference_mode():
        ts = torch.jit.trace(wrap, (ex_src, ex_mask), check_trace=False)
    ts.save(ts_out)
    print(f"[export] → {ts_out} ({os.path.getsize(ts_out)/1e6:.0f} MB)")
    # 검증: 다시 로드한 TorchScript vs 정적 ONNX (같은 규약)
    import onnxruntime as ort
    loaded = torch.jit.load(ts_out).eval()
    onnx_path = str(out_dir / f"encoder_kd_l17_static{A.sec}s.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]) if os.path.exists(onnx_path) else None
    worst = 0.0
    for sec in (A.sec, 8, 4, 2):
        w = (rng.standard_normal(SR * sec) * 0.05).astype(np.float32)
        src = np.zeros((1, N), np.float32); msk = np.ones((1, T), bool); src[0, :w.size] = w; msk[0, :conv_out_len(w.size)] = False
        with torch.inference_mode():
            a = loaded(torch.from_numpy(src), torch.from_numpy(msk)).numpy()
        if sess is not None:
            b = sess.run(["pooled"], {"source": src, "frame_mask": msk})[0]; d = float(np.abs(a - b).max()); worst = max(worst, d)
            print(f"[verify {sec:>2}s] TorchScript vs 정적 ONNX max|Δ|={d:.2e}")
    assert sess is None or worst < 5e-2, "TorchScript/ONNX 불일치"


if __name__ == "__main__":
    main()

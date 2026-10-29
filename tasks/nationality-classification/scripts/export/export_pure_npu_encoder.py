# -*- coding: utf-8 -*-
"""KD 인코더 체크포인트 → 순수 PyTorch(country_classifier.npu_encoder) 가중치 파일 + 정적 ONNX 대조.

    $PY tasks/nationality-classification/scripts/export/export_pure_npu_encoder.py --ckpt <run>/ep0.pt --xlsr300 <xlsr2_300m.pt>
    → <run>/deploy/encoder_pure_l17.pt  (NPU 서버에서 PureEncoder 에 싣고 rebel.compile_from_torch)
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
import argparse, os, numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--xlsr300", required=True); A = ap.parse_args()
    import torch
    from fairseq import checkpoint_utils
    from torch.nn.utils import remove_weight_norm
    from country_classifier.npu_encoder import PureEncoder, conv_out_len
    ck = torch.load(A.ckpt, map_location="cpu"); LAYER = int(ck["layer"])
    (m,), _ = checkpoint_utils.load_model_ensemble([A.xlsr300])[:2] if False else (checkpoint_utils.load_model_ensemble([A.xlsr300])[0], None)
    del m.encoder.layers[LAYER + 1:]; m.load_state_dict(ck["encoder"]); m.eval()
    for mod in m.modules():
        try: remove_weight_norm(mod)
        except (ValueError, AttributeError): pass
    pure = PureEncoder(LAYER + 1)
    keys = set(pure.state_dict())
    sd = {k: v for k, v in m.state_dict().items() if k in keys}
    miss = keys - set(sd); assert not miss, sorted(miss)[:8]
    pure.load_state_dict(sd); pure.eval()
    out = _pl.Path(A.ckpt).parent / "deploy" / "encoder_pure_l17.pt"
    torch.save(sd, out); print(f"[export] → {out} ({os.path.getsize(out)/1e6:.0f} MB, 텐서 {len(sd)})")
    import onnxruntime as ort
    onnx = str(out.parent / "encoder_kd_l17_static12s.onnx")
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]) if os.path.exists(onnx) else None
    N = 192000; T = conv_out_len(N); rng = np.random.default_rng(0); worst = 0.0
    for sec in (12, 8, 4, 2):
        w = (rng.standard_normal(16000 * sec) * 0.05).astype(np.float32)
        src = np.zeros((1, N), np.float32); src[0, :w.size] = w
        valid = np.zeros((1, T), np.float32); valid[0, :conv_out_len(w.size)] = 1.0
        with torch.inference_mode():
            a = pure(torch.from_numpy(src), torch.from_numpy(valid)).numpy()
        if sess is not None:
            b = sess.run(["pooled"], {"source": src, "frame_mask": valid < 0.5})[0]; d = float(np.abs(a - b).max()); worst = max(worst, d)
            print(f"[verify {sec:>2}s] 순수 PyTorch vs 정적 ONNX max|Δ|={d:.2e} (|pooled| 평균 {np.abs(b).mean():.2f})")
    assert worst < 5e-2, "불일치"


if __name__ == "__main__":
    main()

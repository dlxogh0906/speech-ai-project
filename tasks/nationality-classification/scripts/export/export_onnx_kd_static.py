# -*- coding: utf-8 -*-
"""KD 인코더 → **고정 길이 + 패딩 마스크** ONNX. NPU(ATOM-Max 등) 컴파일러용.

NPU 컴파일러는 입력 shape 이 고정돼야 한다. 그래서 파형을 --sec 초(기본 12)로 고정하고, 짧은 발화는 0 으로 채운 뒤
`mask`(True = 채운 구간)를 같이 넣는다. 어텐션은 채운 프레임을 보지 않고, 풀링은 유효 프레임만 평균한다 — 서버의
동적 길이 ONNX(마스크 없음)와 유효 구간 출력이 같도록 검증한다. 헤드((x−mu)/sd·Linear)는 배포 ONNX 와 같이 그래프 밖.

    $PY tasks/nationality-classification/scripts/export/export_onnx_kd_static.py --ckpt <run>/ep0.pt --xlsr300 <xlsr2_300m.pt> [--sec 12] [--opset 17]
    → <run>/deploy/encoder_kd_l17_static<sec>s.onnx  (입력 source[1,N] f32 · frame_mask[1,T] bool → pooled[1,1024])

frame_mask 는 **프레임 단위**(True = 채움)다. fairseq 의 샘플→프레임 마스크 변환(cumsum·scatter)이 ONNX 로 깨져서 호스트가 만든다:
유효 샘플 n → 유효 프레임 conv_out_len(n) (커널 10,3,3,3,3,2,2 · 스트라이드 5,2,2,2,2,2,2). 12초 = 192,000샘플 → T = 599.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import NAT_TASK

import argparse, collections, glob, os, subprocess, time
import numpy as np

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(NAT_TASK / "outputs/kd_encoder/run2/encoder_kd_best.pt"))
    ap.add_argument("--xlsr300", default=os.path.expanduser("~/models/xlsr2_300m.pt"))
    ap.add_argument("--sec", type=int, default=12)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--sample_dir", default="", help="실발화 wav 폴더 — 정적 vs 동적 top-1 일치 검사(선택)")
    A = ap.parse_args()
    out_dir = _pl.Path(A.ckpt).parent / "deploy"; out_dir.mkdir(exist_ok=True)
    N = SR * A.sec
    onnx_out = str(out_dir / f"encoder_kd_l17_static{A.sec}s.onnx")

    import torch, torch.nn as nn
    from fairseq import checkpoint_utils
    from export_onnx_kd_encoder import patch_pad_to_multiple
    patch_pad_to_multiple()
    import fairseq.models.wav2vec.wav2vec2 as w2
    w2.index_put = lambda tensor, indices, value: tensor.masked_fill(indices.unsqueeze(-1), value)     # x[mask]=0 → 브로드캐스트 (ScatterND 회피)
    ck = torch.load(A.ckpt, map_location="cpu"); LAYER = int(ck["layer"])
    models, _ = checkpoint_utils.load_model_ensemble([A.xlsr300]); enc = models[0]
    del enc.encoder.layers[LAYER + 1:]
    enc.load_state_dict(ck["encoder"]); enc.eval()
    for mod in enc.modules():
        if hasattr(mod, "required_seq_len_multiple"): mod.required_seq_len_multiple = 1
    from torch.nn.utils import remove_weight_norm
    for mod in enc.modules():
        try: remove_weight_norm(mod)
        except (ValueError, AttributeError): pass

    class Wrap(nn.Module):
        def __init__(self, e): super().__init__(); self.enc = e
        def forward(self, source, frame_mask):
            f = self.enc.feature_extractor(source).transpose(1, 2)                   # (1,T,512)
            f = self.enc.layer_norm(f)
            if self.enc.post_extract_proj is not None: f = self.enc.post_extract_proj(f)
            _, lr = self.enc.encoder(f, padding_mask=frame_mask, layer=LAYER)
            x = lr[LAYER][0].transpose(0, 1)                                         # (1,T,C)
            v = (~frame_mask).to(x.dtype).unsqueeze(-1)
            return (x * v).sum(1) / v.sum(1).clamp_min(1.0)
    wrap = Wrap(enc).eval()

    def conv_out_len(n):
        for k, st in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
            n = (n - k) // st + 1
        return n
    T = conv_out_len(N)

    rng = np.random.default_rng(0)
    ex_src = torch.from_numpy((rng.standard_normal(N) * 0.05).astype(np.float32))[None]
    ex_mask = torch.zeros(1, T, dtype=torch.bool); ex_mask[:, conv_out_len(SR * (A.sec - 4)):] = True     # 예제 입력에 패딩 포함 → 마스크 경로 추적
    with torch.inference_mode():
        torch.onnx.export(wrap, (ex_src, ex_mask), onnx_out, input_names=["source", "frame_mask"], output_names=["pooled"],
                          opset_version=A.opset, do_constant_folding=True)
    import onnx
    m = onnx.load(onnx_out); onnx.checker.check_model(onnx_out)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(f"[export] → {onnx_out} ({os.path.getsize(onnx_out)/1e6:.0f} MB) · opset {A.opset} · 노드 {len(m.graph.node)}")
    print("[ops] " + ", ".join(f"{k} {v}" for k, v in ops.most_common()))
    print("[inputs] " + str([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]))

    # 검증: 동적 ONNX(배포 규격, 마스크 없음)로 원 길이 그대로 계산한 값 vs 정적 ONNX(0 채움 + 마스크)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    st = ort.InferenceSession(onnx_out, so, providers=["CPUExecutionProvider"])
    dyn_path = str(out_dir / "encoder_kd_l17.onnx")
    dyn = ort.InferenceSession(dyn_path, so, providers=["CPUExecutionProvider"]) if os.path.exists(dyn_path) else None
    def run_static(w):
        src = np.zeros((1, N), np.float32); msk = np.ones((1, T), bool)
        n = min(w.size, N); src[0, :n] = w[:n]; msk[0, :conv_out_len(n)] = False
        return st.run(["pooled"], {"source": src, "frame_mask": msk})[0]
    worst = 0.0
    for sec in (A.sec, 8, 4, 2):
        w = (rng.standard_normal(SR * sec) * 0.05).astype(np.float32)
        ref = dyn.run(["pooled"], {"source": w[None, :]})[0] if dyn else None
        if ref is None:
            with torch.inference_mode():
                ref = wrap(torch.from_numpy(w)[None], torch.zeros(1, conv_out_len(w.size), dtype=torch.bool)).numpy()
        d = np.abs(ref - run_static(w)); worst = max(worst, float(d.max()))
        print(f"[verify {sec:>2}s → {A.sec}s 패딩] max|Δ|={d.max():.2e} mean|Δ|={d.mean():.2e} (기준 |pooled| 평균 {np.abs(ref).mean():.2f})")
    assert worst < 5e-2, "정적/동적 불일치 — 마스크 경로 확인"

    if A.sample_dir:
        from shared.paths import FFMPEG
        head = ck["head"]; W = head["weight"].numpy(); b = head["bias"].numpy(); mu = np.asarray(ck["mu"], np.float32); sd = np.asarray(ck["sd"], np.float32); labels = list(ck["labels"])
        def top1(p):
            z = ((np.clip(np.nan_to_num(p), -50, 50) - mu) / sd) @ W.T + b; return labels[int(z.argmax())]
        agree = tot = 0
        for wav in sorted(glob.glob(os.path.join(A.sample_dir, "*.wav")))[:12]:
            raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", wav, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(A.sec), "pipe:1"], stdout=subprocess.PIPE).stdout
            w = np.frombuffer(raw, np.float32).copy()
            if w.size < SR: continue
            a = top1(dyn.run(["pooled"], {"source": w[None, :]})[0]) if dyn else None; s = top1(run_static(w)); tot += 1; agree += (a == s)
            print(f"  {os.path.basename(wav)[:30]:30} 동적 {a:14} 정적 {s:14} {'✓' if a == s else '✗'}")
        print(f"[실발화] top-1 일치 {agree}/{tot}")
    t = time.perf_counter()
    for _ in range(3): run_static((rng.standard_normal(N) * 0.05).astype(np.float32))
    print(f"[bench] 정적 ONNX CPU(4스레드) {A.sec}s: {(time.perf_counter()-t)/3*1000:.0f} ms")


if __name__ == "__main__":
    main()

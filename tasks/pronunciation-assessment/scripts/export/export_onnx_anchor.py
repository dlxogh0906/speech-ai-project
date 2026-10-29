# -*- coding: utf-8 -*-
"""음향 기준 모델(anchor) → ONNX — 판별기 파이프라인용. 로그사후확률과 **은닉 벡터 2종**을 함께 내보낸다.

`export_onnx_mdd.py` 는 lprobs 만 낸다. 음소 판별기(`mdd.phone_detector`)는 자리별 은닉 벡터(마지막 층 · 16층)도 쓰므로
출력이 셋이다:  lprobs[1,T,V] · h_last[1,T,C] · h_mid[1,T,C].  입력 규약은 같다(정규화한 16 kHz 파형 [1, samples]).
수출 시 처리도 같다: pos_conv weight_norm 제거 · 2의 배수 패딩 끄기(required_seq_len_multiple=1).

패리티: 같은 발화에서 PyTorch CPU fp32 와 출력 셋의 최대 절대 오차 · greedy 일치.

    $PY tasks/pronunciation-assessment/scripts/export/export_onnx_anchor.py --ckpt <anchor.pt> --dict_dir <dict.ltr.txt 폴더> \
        --out <dir> --wav_list <wav 경로 목록 txt> [--int8]
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq

import argparse, json, os, time, wave

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dict_dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--name", default="anchor_fp32.onnx")
    ap.add_argument("--wav_list", required=True, help="패리티용 wav 경로를 한 줄에 하나씩 적은 파일")
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--mid_layer", type=int, default=16)
    ap.add_argument("--opset", type=int, default=17); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--int8", action="store_true", help="동적 int8(MatMul · 텐서별)도 만들고 같은 패리티를 잰다")
    a = ap.parse_args()

    ensure_fairseq()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from fairseq import checkpoint_utils
    torch.set_num_threads(a.threads)
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([a.ckpt], arg_overrides={"data": a.dict_dir})
    m = models[0].eval().float()
    assert cfg.task.normalize, "normalize=False 체크포인트는 입력 규약이 다르다"
    w2v = m.w2v_encoder.w2v_model

    def load(p):
        with wave.open(p, "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        return F.layer_norm(torch.from_numpy(x), (len(x),))[None]

    wavs = [l.strip() for l in open(a.wav_list, encoding="utf-8") if l.strip()][: a.n]

    class Anchor(nn.Module):
        def __init__(self, m, mid):
            super().__init__(); self.m = m; self.mid = mid

        def forward(self, source):
            enc = self.m.w2v_encoder
            res = enc.w2v_model.extract_features(source=source, padding_mask=None, mask=False)
            h = res["x"]
            return F.log_softmax(enc.proj(h).float(), -1), h, res["layer_results"][self.mid - 1][0].transpose(0, 1)

    net = Anchor(m, a.mid_layer).eval()
    with torch.no_grad():
        refs = [[t.numpy() for t in net(load(p))] for p in wavs]           # weight_norm 제거 **전** = 서빙과 같은 모델

    for mod in w2v.encoder.pos_conv.modules():
        if isinstance(mod, nn.Conv1d) and hasattr(mod, "weight_g"):
            torch.nn.utils.remove_weight_norm(mod)
    from fairseq.models.wav2vec import wav2vec2 as _w2v
    w2v.encoder.required_seq_len_multiple = 1
    _w2v.pad_to_multiple = lambda x, multiple, dim=-1, value=0: (x, 0)

    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, a.name)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(net, (torch.randn(1, 64080),), onnx_path, input_names=["source"], output_names=["lprobs", "h_last", "h_mid"],
                          dynamic_axes={"source": {1: "samples"}, "lprobs": {1: "frames"}, "h_last": {1: "frames"}, "h_mid": {1: "frames"}},
                          opset_version=a.opset, do_constant_folding=True)
    print(f"[export] {onnx_path} {os.path.getsize(onnx_path)/2**20:.0f} MiB ({time.time()-t0:.0f}s)", flush=True)

    import onnxruntime as ort
    rep = {"ckpt": os.path.abspath(a.ckpt), "n_utts": len(wavs), "mid_layer": a.mid_layer, "variants": {}}

    def parity(path, tag):
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        mx = np.zeros(3); eq = 0; cos = []; t0 = time.time()
        for p, ref in zip(wavs, refs):
            out = sess.run(None, {"source": load(p).numpy()})
            n = min(out[0].shape[1], ref[0].shape[1])
            for k in range(3):
                mx[k] = max(mx[k], float(np.abs(out[k][0, :n] - ref[k][0, :n]).max()))
            eq += int((out[0][0, :n].argmax(-1) == ref[0][0, :n].argmax(-1)).all())
            a_, b_ = out[1][0, :n], ref[1][0, :n]
            cos.append(float(((a_ * b_).sum(-1) / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-9)).mean()))
        rep["variants"][tag] = {"file": path, "mib": round(os.path.getsize(path) / 2 ** 20), "max_abs_err": {"lprobs": mx[0], "h_last": mx[1], "h_mid": mx[2]},
                                "frame_argmax_all_equal_utts": eq, "h_last_mean_cosine": round(float(np.mean(cos)), 6), "sec_per_utt": round((time.time() - t0) / len(wavs), 3)}
        print(f"[{tag}] {rep['variants'][tag]}", flush=True)

    parity(onnx_path, "fp32")
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = onnx_path.replace(".onnx", "_pre.onnx"); q = os.path.join(a.out, a.name.replace("fp32", "int8"))
        quant_pre_process(onnx_path, pre, skip_symbolic_shape=True)
        quantize_dynamic(pre, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        parity(q, "int8")
    json.dump(rep, open(os.path.join(a.out, "anchor_parity.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

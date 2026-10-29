# -*- coding: utf-8 -*-
"""발음 진단(MDD) 체크포인트 → ONNX(fp32) 수출 + 서빙 경로 대비 패리티 검증.

경량화 2단계(형식 변환). 가중치·연산은 그대로고 실행 엔진만 바꾼다 — 정확도가 바뀌면 안 된다.
그래서 수출 직후 **서빙이 실제로 쓰는 경로**(`gop_feats_v2.forward_lprobs`, CPU fp32)와
실제 발화에서 출력을 대조한다.

그래프
    입력  source  (1, samples)     float32 — 16 kHz 파형에 발화 단위 layer_norm 을 **적용한 뒤**의 값
    출력  lprobs  (1, frames, 50)  float32 — CTC log-softmax (blank = index 0)

정규화를 그래프 밖에 둔 이유: Decoder / forward_lprobs 가 패딩 **전에** 발화별로 정규화한다.
나중에 배치로 확장할 때 그래프 안에 넣으면 패딩까지 정규화에 섞인다.

⚠️ fairseq TransformerEncoder 는 프레임 수를 2의 배수로 맞추려 패딩한다(required_seq_len_multiple).
   패딩 길이를 파이썬으로 계산해서 추적 중에는 `Tensor.is_integer()` 로 죽는다. 추출하는 동안만
   패딩을 끈다(국적 export_onnx_kd_encoder.py 도 required_seq_len_multiple=1 로 같은 처리).
   패딩 프레임은 어텐션에서 마스킹되므로 결과가 같아야 한다 — 레퍼런스는 **패딩을 켠 원래 모델**로
   먼저 뽑아 두고, 패리티 검사가 홀·짝 프레임 길이를 나눠 이를 확인한다.

    $PY tasks/pronunciation-assessment/scripts/export/export_onnx_mdd.py \\
        --wav_dir <16kHz wav 폴더> --out <출력 폴더>
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

import argparse, glob, json, os, time
import numpy as np


def greedy(lp, blank):
    """CTC greedy — Decoder.decode 와 같은 규칙(연속 중복 제거 → blank 제거)."""
    g = lp.argmax(-1)
    keep = np.ones(len(g), bool)
    keep[1:] = g[1:] != g[:-1]
    return [int(t) for t in g[keep] if t != blank]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("MDD_SERVE_CKPT"),
                    help="체크포인트 (기본: .env 의 MDD_SERVE_CKPT)")
    ap.add_argument("--out", required=True, help="출력 폴더 (.onnx 와 패리티 리포트)")
    ap.add_argument("--name", default="mdd_fp32.onnx")
    ap.add_argument("--wav_dir", required=True, help="패리티용 16 kHz mono wav 폴더")
    ap.add_argument("--n", type=int, default=0, help="패리티 발화 수 상한 (0 = 전부)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--skip_export", action="store_true", help="이미 있는 .onnx 로 패리티만 잰다")
    a = ap.parse_args()
    if not a.ckpt:
        ap.error("--ckpt 가 없고 .env 에 MDD_SERVE_CKPT 도 없다")

    os.environ["MDD_FORCE_CPU"] = "1"            # 레퍼런스는 CPU fp32 — ONNX 도 fp32 라 같은 조건
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import editdistance
    torch.set_num_threads(a.threads)

    from mdd import mdd_eval
    from mdd.gop_feats_v2 import forward_lprobs
    mdd_eval.CKPT = a.ckpt                       # import 가 끝난 **뒤에** 넣는다 (mdd.pipeline 이 덮어쓴다)
    dec = mdd_eval.Decoder()
    assert dec.device == "cpu" and next(dec.model.parameters()).dtype == torch.float32
    if not dec.normalize:
        raise SystemExit("cfg.task.normalize=False 인 체크포인트다 — 이 스크립트의 입력 규약과 다르다")

    wavs = sorted(glob.glob(os.path.join(a.wav_dir, "*.wav")))
    if a.n:
        wavs = wavs[:a.n]
    if not wavs:
        raise SystemExit(f"wav 가 없다: {a.wav_dir}")
    os.makedirs(a.out, exist_ok=True)
    onnx_path = os.path.join(a.out, a.name)

    # ── 1. 레퍼런스: 서빙 경로 그대로 (weight_norm 제거 **전** 모델) ──
    t0 = time.time()
    refs = []
    for p in wavs:
        w = mdd_eval.load_wav(p)
        lp, Ls = forward_lprobs(dec, [w])
        refs.append(lp[0, :Ls[0]].numpy())
    t_ref = time.time() - t0
    print(f"[ref] PyTorch CPU fp32 {len(wavs)}발화 {t_ref:.1f}s", flush=True)

    w2v = dec.model.w2v_encoder.w2v_model

    def n_frames(samples):
        return int(w2v._get_feat_extract_output_lengths(torch.tensor(samples)))

    # ── 2. 수출 ──
    if not a.skip_export:
        # weight_norm 훅은 매 forward 마다 weight 를 재계산한다. 상수로 굳혀야 그래프가 단순해진다.
        for mod in w2v.encoder.pos_conv.modules():
            if isinstance(mod, nn.Conv1d) and hasattr(mod, "weight_g"):
                torch.nn.utils.remove_weight_norm(mod)
        # 2의 배수 패딩 끄기 (docstring ⚠️). 모듈 전역 이름을 바꿔야 extract_features 가 이걸 부른다.
        from fairseq.models.wav2vec import wav2vec2 as _w2v
        w2v.encoder.required_seq_len_multiple = 1
        _w2v.pad_to_multiple = lambda x, multiple, dim=-1, value=0: (x, 0)

        class CTC(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, source):
                net = self.m(source=source, padding_mask=None)
                return self.m.get_normalized_probs(net, log_probs=True).transpose(0, 1)

        T = next(s for s in range(64000, 64400) if n_frames(s) % 2 == 0)
        dummy = torch.randn(1, T)
        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(CTC(dec.model).eval(), (dummy,), onnx_path,
                              input_names=["source"], output_names=["lprobs"],
                              dynamic_axes={"source": {1: "samples"}, "lprobs": {1: "frames"}},
                              opset_version=a.opset, do_constant_folding=True)
        t_exp = time.time() - t0
        print(f"[export] {onnx_path}  {os.path.getsize(onnx_path)/2**20:.0f} MiB  "
              f"(추적 입력 {T} samples = {n_frames(T)} frames) {t_exp:.0f}s", flush=True)
    else:
        t_exp = None

    # ── 3. ONNX Runtime (CPU) 로 같은 발화 ──
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    t_load = time.time() - t0

    rows = []
    t0 = time.time()
    for p, ref in zip(wavs, refs):
        w = mdd_eval.load_wav(p)
        x = F.layer_norm(torch.from_numpy(w), (len(w),)).numpy()[None, :]
        out = sess.run(["lprobs"], {"source": x})[0][0]
        row = {"wav": os.path.basename(p), "sec": round(len(w) / 16000, 2),
               "frames_ref": int(ref.shape[0]), "frames_onnx": int(out.shape[0])}
        if out.shape != ref.shape:
            row["shape_mismatch"] = True
            rows.append(row)
            continue
        d = np.abs(out - ref)
        s_ref, s_onx = greedy(ref, dec.blank), greedy(out, dec.blank)
        row.update(max_abs=float(d.max()), mean_abs=float(d.mean()),
                   frame_argmax_agree=float((out.argmax(-1) == ref.argmax(-1)).mean()),
                   seq_equal=s_ref == s_onx,
                   tok_edits=int(editdistance.eval(s_ref, s_onx)), tok_len=len(s_ref))
        rows.append(row)
    t_onnx = time.time() - t0

    ok = [r for r in rows if not r.get("shape_mismatch")]
    maxabs = np.array([r["max_abs"] for r in ok]) if ok else np.array([np.nan])

    def part(sel):
        sel = list(sel)
        if not sel:
            return None
        return {"n": len(sel),
                "seq_equal_rate": sum(r["seq_equal"] for r in sel) / len(sel),
                "frame_argmax_agree": float(np.mean([r["frame_argmax_agree"] for r in sel])),
                "max_abs_max": float(max(r["max_abs"] for r in sel))}

    rep = {
        "ckpt": a.ckpt, "onnx": onnx_path,
        "onnx_bytes": os.path.getsize(onnx_path), "opset": a.opset,
        "n_utts": len(rows), "shape_mismatch": sum(1 for r in rows if r.get("shape_mismatch")),
        "seq_equal_rate": (sum(r["seq_equal"] for r in ok) / len(ok)) if ok else None,
        "token_edit_rate": (sum(r["tok_edits"] for r in ok) / max(1, sum(r["tok_len"] for r in ok))) if ok else None,
        "frame_argmax_agree": float(np.mean([r["frame_argmax_agree"] for r in ok])) if ok else None,
        "max_abs": {"max": float(maxabs.max()), "p99": float(np.percentile(maxabs, 99)),
                    "median": float(np.median(maxabs))},
        "even_frames": part(r for r in ok if r["frames_ref"] % 2 == 0),
        "odd_frames": part(r for r in ok if r["frames_ref"] % 2 == 1),
        "seconds": {"export": t_exp, "ort_load": round(t_load, 1),
                    "torch_cpu_total": round(t_ref, 1), "ort_cpu_total": round(t_onnx, 1),
                    "_note": "참고용 — 단독 실행 벤치가 아니다. 속도 판단은 벤치 스크립트로"},
        "threads": a.threads,
        "rows": rows,
    }
    rpath = os.path.join(a.out, os.path.splitext(a.name)[0] + "_parity.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)

    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False, indent=1))
    print(f"[parity] → {rpath}")
    passed = ok and rep["shape_mismatch"] == 0 and rep["seq_equal_rate"] == 1.0
    print("[parity] PASS — 모든 발화에서 greedy 출력 동일" if passed
          else "[parity] ⚠️ 출력이 다른 발화가 있다 — rows 를 볼 것")
    return 0 if passed else 1


if __name__ == "__main__":
    _sys.exit(main())

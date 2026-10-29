# -*- coding: utf-8 -*-
"""발음 진단 ONNX(fp32) → 경량 변형 생성 + fp32 대비 출력 비교 (경량화 3단계).

변형
    dyn_matmul     int8 동적 양자화 — MatMul 가중치만 (국적 0705 와 같은 방식, conv 는 fp32)
    dyn_matmul_pc  위와 같되 채널별(per-channel) 스케일 — 정확도 손실을 줄이는 쪽
    static_qdq     int8 정적 양자화(QDQ) — Conv·MatMul 가중치+활성값. 앞단 conv 까지 int8 이라
                   CPU 에서 속도 여지가 가장 크지만, 활성값 이상치 때문에 정확도 위험도 가장 크다.
                   보정(calibration)은 **test 화자가 아닌** 발화만 쓴다(md5 화자 분할 기준).
    fp16           GPU 용 반정밀도 그래프 (입출력은 float32 유지). CPU 에선 쓰지 말 것 — 함정 7

여기서 재는 건 **fp32 ONNX 와 greedy 출력이 얼마나 같은가**뿐이다(빠른 1차 선별).
일치율은 정확도가 아니다(DECISIONS #12) — 채택 판단은 `decode_manifest_backend.py` 로 test 셋을
디코드해 정답 대비 F1 로 한다.

    $PY tasks/pronunciation-assessment/scripts/export/quantize_onnx_mdd.py \\
        --fp32 <mdd_fp32.onnx> --out <폴더> --wav_dir <16kHz wav 폴더> --variants dyn_matmul,static_qdq
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, glob, hashlib, json, os, time, wave
import numpy as np

VARIANTS = ["dyn_matmul", "dyn_matmul_pc", "static_qdq", "fp16"]


def load_norm(path):
    """int16/32768 → 발화 단위 layer_norm (Decoder 와 같은 식)."""
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-5)).astype(np.float32)


def is_test_speaker(wav_path):
    """rebuild_tag_manifest.split_of 와 같은 규칙 — 보정 데이터에서 test 화자를 뺀다."""
    spk = os.path.basename(wav_path).split("-")[0]
    return int(hashlib.md5(spk.encode()).hexdigest()[:8], 16) % 10 == 0


def greedy(lp):
    g = lp.argmax(-1)
    keep = np.ones(len(g), bool)
    keep[1:] = g[1:] != g[:-1]
    return [int(t) for t in g[keep] if t != 0]


def session(path, threads, cuda=False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=prov)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav_dir", required=True, help="비교(+보정)용 16 kHz wav 폴더")
    ap.add_argument("--variants", default="dyn_matmul,dyn_matmul_pc")
    ap.add_argument("--n_calib", type=int, default=60, help="static_qdq 보정 발화 수 (test 화자 제외)")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    variants = a.variants.split(",")
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        ap.error(f"모르는 변형: {bad} (가능: {VARIANTS})")

    import editdistance
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    os.makedirs(a.out, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(a.wav_dir, "*.wav")))
    if not wavs:
        raise SystemExit(f"wav 가 없다: {a.wav_dir}")

    # fp32 기준 출력 (한 번만)
    ref_sess = session(a.fp32, a.threads)
    xs = {p: load_norm(p) for p in wavs}
    ref = {p: ref_sess.run(None, {"source": xs[p][None, :]})[0][0] for p in wavs}
    del ref_sess

    pre = os.path.join(a.out, "mdd_fp32_pre.onnx")          # 양자화 전처리(모양 추론 + 그래프 최적화)
    report = {"fp32": a.fp32, "fp32_bytes": os.path.getsize(a.fp32), "n_compare": len(wavs), "variants": {}}
    for v in variants:
        dst = os.path.join(a.out, f"mdd_{v}.onnx")
        t0 = time.time()
        if v != "fp16" and not os.path.exists(pre):
            print("[pre] quant_pre_process ...", flush=True)
            quant_pre_process(a.fp32, pre, skip_symbolic_shape=False)
        if v == "dyn_matmul":
            quantize_dynamic(pre, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        elif v == "dyn_matmul_pc":
            quantize_dynamic(pre, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                             per_channel=True)
        elif v == "static_qdq":
            calib = [p for p in wavs if not is_test_speaker(p)][:a.n_calib]

            class Reader(CalibrationDataReader):
                def __init__(self):
                    self.it = iter(calib)

                def get_next(self):
                    p = next(self.it, None)
                    return None if p is None else {"source": xs[p][None, :]}

            # AVX2(VNNI 없음)에서 U8S8 은 포화가 날 수 있어 reduce_range 를 켠다(ORT 권고).
            quantize_static(pre, dst, Reader(), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            per_channel=True, reduce_range=True, op_types_to_quantize=["Conv", "MatMul"],
                            calibrate_method=CalibrationMethod.MinMax)
            report.setdefault("calib", {})["n"] = len(calib)
        elif v == "fp16":
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16
            m = onnx.load(a.fp32)
            onnx.save(convert_float_to_float16(m, keep_io_types=True), dst)
        t_make = time.time() - t0

        cuda = v == "fp16"
        try:
            sess = session(dst, a.threads, cuda=cuda)
        except Exception as e:
            report["variants"][v] = {"file": dst, "error": f"{type(e).__name__}: {e}"}
            print(f"[{v}] 세션 생성 실패: {e}", flush=True)
            continue
        eq = edits = toks = 0
        agree = []
        t0 = time.time()
        for p in wavs:
            out = sess.run(None, {"source": xs[p][None, :]})[0][0].astype(np.float32)
            r = ref[p]
            if out.shape != r.shape:
                continue
            s_r, s_o = greedy(r), greedy(out)
            eq += s_r == s_o
            edits += editdistance.eval(s_r, s_o)
            toks += len(s_r)
            agree.append(float((out.argmax(-1) == r.argmax(-1)).mean()))
        report["variants"][v] = {
            "file": dst, "bytes": os.path.getsize(dst), "make_sec": round(t_make, 1),
            "provider": sess.get_providers()[0],
            "seq_equal_rate_vs_fp32": eq / len(wavs), "token_edit_rate_vs_fp32": edits / max(1, toks),
            "frame_argmax_agree_vs_fp32": float(np.mean(agree)) if agree else None,
            "compare_sec": round(time.time() - t0, 1)}
        print(f"[{v}] {os.path.getsize(dst)/2**20:.0f} MiB · 출력 일치 {eq}/{len(wavs)} · "
              f"토큰 편집률 {edits/max(1,toks):.4f} · {sess.get_providers()[0]}", flush=True)
        del sess

    rpath = os.path.join(a.out, "quantize_report.json")
    if os.path.exists(rpath):                       # 변형을 나눠 돌려도 누적되게
        old = json.load(open(rpath, encoding="utf-8"))
        old.get("variants", {}).update(report["variants"])
        report["variants"] = old["variants"]
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"→ {rpath}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

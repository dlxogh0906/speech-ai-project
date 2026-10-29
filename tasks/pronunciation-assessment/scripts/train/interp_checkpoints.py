# -*- coding: utf-8 -*-
"""두 HF 체크포인트(같은 구조)의 가중치를 섞는다(WiSE-FT) — w = (1 − α)·base + α·ft. α = 이어 학습 모델 쪽 비율.

`interp_a0.5_step3000`(2026-10-16 NPU 운영)과 같은 방식이다. 설정·사전·전처리 파일은 ft 쪽을 그대로 복사하고
`interp.json` 에 두 모델과 α 를 남긴다. 가중치는 fp32 safetensors 로 쓴다.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:$PYTHONPATH $PY .../interp_checkpoints.py --alpha 0.75 \\
        --base $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 \\
        --ft $SPEECH_AI_DATA/checkpoints/w2vbert_synth_tense_v2/step_3000 \\
        --out $SPEECH_AI_DATA/checkpoints/w2vbert_synth_tense_v2/interp_a0.75_step3000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="이어 학습 모델 비율(0 = base, 1 = ft)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    b = load_file(os.path.join(a.base, "model.safetensors"))
    f = load_file(os.path.join(a.ft, "model.safetensors"))
    if set(b) != set(f):
        raise SystemExit("두 체크포인트의 가중치 이름이 다르다")
    out = {}
    for k, v in f.items():
        if v.dtype.is_floating_point:
            out[k] = ((1.0 - a.alpha) * b[k].float() + a.alpha * v.float()).contiguous()
        else:
            out[k] = v.clone()
    os.makedirs(a.out, exist_ok=True)
    for name in os.listdir(a.ft):
        if name.endswith(".json") or name.endswith(".txt"):
            shutil.copy(os.path.join(a.ft, name), os.path.join(a.out, name))
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"base": os.path.abspath(a.base), "ft": os.path.abspath(a.ft), "alpha": a.alpha},
              open(os.path.join(a.out, "interp.json"), "w"), ensure_ascii=False, indent=1)
    print("[끝] %s · 텐서 %d · α %.2f" % (a.out, len(out), a.alpha))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""omniASR-W2V-300M (fairseq2) → legacy fairseq1 wav2vec2 체크포인트 변환.

아키텍처가 XLS-R 300m과 동일(wav2vec2 large_lv60k)하므로 키 이름만 재작성한다.
템플릿으로 기존 xlsr2_300m.pt의 cfg를 그대로 쓰고 model state_dict만 교체 →
기존 학습 config에서 w2v_path 한 줄만 바꿔 쓸 수 있는 파일을 만든다.

실행: fairseq_xlsr env (torch 2.1, py3.8) — 저장 호환성 보장 목적.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MODEL_ROOT

import re, sys, torch

SRC = f"{DATA_ROOT}/hf_home/hub/models--facebook--omniASR-W2V-300M/snapshots/e20053922d00fec2aad44fb039560e23acf43a19/omniASR-W2V-300M.pt"
TEMPLATE = f"{MODEL_ROOT}/xlsr2_300m.pt"
OUT = f"{DATA_ROOT}/models_omni/omniASR-W2V-300M-fairseq1.pt"

RULES = [
    # (fairseq2 패턴, fairseq1 치환)
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.conv\.(weight|bias)$",
     r"feature_extractor.conv_layers.\1.0.\2"),
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.layer_norm\.(weight|bias)$",
     r"feature_extractor.conv_layers.\1.2.1.\2"),
    (r"^encoder_frontend\.post_extract_layer_norm\.(weight|bias)$", r"layer_norm.\1"),
    (r"^encoder_frontend\.model_dim_proj\.(weight|bias)$", r"post_extract_proj.\1"),
    (r"^encoder_frontend\.pos_encoder\.conv\.(bias|weight_g|weight_v)$", r"encoder.pos_conv.0.\1"),
    (r"^encoder\.layers\.(\d+)\.self_attn\.output_proj\.(weight|bias)$",
     r"encoder.layers.\1.self_attn.out_proj.\2"),
    (r"^encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)$",
     r"encoder.layers.\1.self_attn.\2.\3"),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$",
     r"encoder.layers.\1.self_attn_layer_norm.\2"),
    (r"^encoder\.layers\.(\d+)\.ffn\.inner_proj\.(weight|bias)$", r"encoder.layers.\1.fc1.\2"),
    (r"^encoder\.layers\.(\d+)\.ffn\.output_proj\.(weight|bias)$", r"encoder.layers.\1.fc2.\2"),
    (r"^encoder\.layers\.(\d+)\.ffn_layer_norm\.(weight|bias)$", r"encoder.layers.\1.final_layer_norm.\2"),
    (r"^encoder\.layer_norm\.(weight|bias)$", r"encoder.layer_norm.\1"),
    (r"^masker\.temporal_mask_embed$", r"mask_emb"),
    (r"^quantizer\.entries$", r"quantizer.vars"),
    (r"^quantizer\.entry_proj\.(weight|bias)$", r"quantizer.weight_proj.\1"),
    (r"^final_target_proj\.(weight|bias)$", r"project_q.\1"),
    (r"^final_proj\.(weight|bias)$", r"final_proj.\1"),
]
DROP = {"quantizer.num_updates"}  # legacy state_dict에 없는 카운터

def convert_key(k):
    if k in DROP:
        return None
    for pat, rep in RULES:
        if re.match(pat, k):
            return re.sub(pat, rep, k)
    raise KeyError(f"매핑 규칙 없음: {k}")

def main():
    print("[1] fairseq2 체크포인트 로드...", flush=True)
    src = torch.load(SRC, map_location="cpu")["model"]
    print(f"    {len(src)} tensors")

    print("[2] 키 변환...", flush=True)
    new_sd = {}
    for k, v in src.items():
        nk = convert_key(k)
        if nk is None:
            continue
        assert nk not in new_sd, f"중복 키: {nk}"
        new_sd[nk] = v

    print("[3] 템플릿(XLS-R) 로드 + 키 집합 대조...", flush=True)
    tpl = torch.load(TEMPLATE, map_location="cpu")
    tpl_sd = tpl["model"]
    tpl_keys, new_keys = set(tpl_sd), set(new_sd)
    missing = tpl_keys - new_keys      # 템플릿엔 있는데 변환본에 없음
    extra = new_keys - tpl_keys        # 변환본에만 있음
    print(f"    missing={sorted(missing)}")
    print(f"    extra  ={sorted(extra)}")
    shape_bad = [k for k in (tpl_keys & new_keys) if tuple(tpl_sd[k].shape) != tuple(new_sd[k].shape)]
    print(f"    shape 불일치={shape_bad}")
    if missing or extra or shape_bad:
        print("!! 불일치 존재 — 저장 중단"); sys.exit(1)

    print("[4] 템플릿 cfg 유지, model만 교체 후 저장...", flush=True)
    tpl["model"] = new_sd
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(tpl, OUT)
    print(f"[저장] {OUT}")

    print("[5] fairseq 정식 로드 검증...", flush=True)
    from fairseq import checkpoint_utils
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([OUT])
    m = models[0]
    n = sum(p.numel() for p in m.parameters())
    print(f"    로드 성공: {type(m).__name__}, {n:,} params")

    print("[6] 더미 forward (CPU)...", flush=True)
    m.eval()
    with torch.no_grad():
        wav = torch.randn(1, 32000)  # 2초
        out = m.extract_features(wav, padding_mask=None, mask=False)
        feat = out["x"] if isinstance(out, dict) else out[0]
        print(f"    출력: {tuple(feat.shape)}  (기대: (1, ~99, 1024))")
    print("[완료] 변환 성공")

if __name__ == "__main__":
    main()

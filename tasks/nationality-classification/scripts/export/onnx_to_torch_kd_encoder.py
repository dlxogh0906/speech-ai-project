# -*- coding: utf-8 -*-
"""배포 KD 인코더 ONNX → PyTorch 체크포인트 (재학습 초기값용).

왜
    배포 모델(encoder_kd_l17.onnx + head_np.npz)의 학습 체크포인트(encoder_kd_best.pt)가 이 서버에 없다.
    ONNX 가중치를 fairseq 300m(≤17층) 모듈에 되돌려 넣어 이어서 학습할 수 있게 한다.

어떻게
    - 이름이 남은 가중치(enc.*)는 그대로 넣는다. pos_conv 는 수출 때 weight_norm 을 걷어냈으므로
      weight_g/weight_v 로 다시 쪼갠다(g = ‖W‖, v = W → 유효 가중치가 W 로 같다).
    - 상수 접기로 이름을 잃은 가중치(q/k/v·fc1·fc2·post_extract_proj 의 전치 행렬, q/k/v bias,
      특징추출기 LayerNorm)는 **소비 노드의 위치**로 모듈을 정한다. 위치→모듈 대응은 추측하지 않고,
      KD 때 동결돼 원본 300m 과 값이 같아야 하는 0~11층·특징추출기에서 **값 일치로 확정**한 뒤
      같은 대응을 12~17층에 적용한다.
    - 마지막에 PyTorch 와 ONNX 의 pooled 출력을 대조한다(max|Δ| < 1e-3 이 아니면 실패).

출력
    encoder_kd_best.pt 와 같은 형식: encoder(state_dict) · head{weight,bias} · mu · sd · labels · layer

    $PY tasks/nationality-classification/scripts/export/onnx_to_torch_kd_encoder.py \
        --onnx $KD_MODEL_DIR/encoder_kd_l17.onnx --head $KD_MODEL_DIR/head_np.npz \
        --out /data/speech-ai/out/nationality/webmic_kd/init_kd_from_onnx.pt
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import XLSR_300M

import argparse, collections, os, re
import numpy as np

LAYER = 17
SR = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kd = os.environ.get("KD_MODEL_DIR", "")
    ap.add_argument("--onnx", default=os.path.join(kd, "encoder_kd_l17.onnx"))
    ap.add_argument("--head", default=os.path.join(kd, "head_np.npz"))
    ap.add_argument("--xlsr300", default=str(XLSR_300M))
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    import onnx
    from onnx import numpy_helper
    import torch
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils

    m = onnx.load(A.onnx)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    # ── 이름 없는 가중치 → (소비 노드 위치 키) ──
    pos = {}                                   # 초기화 이름 → (층 or None, 위치 키)
    for nd in m.graph.node:
        for x in nd.input:
            if x in inits and not x.startswith("enc.") and x not in pos:
                # conv_layers.N 은 인코더 층이 아니다 — /encoder/layers.N/ 만 층으로 본다
                mm = re.search(r"/encoder/layers\.(\d+)/", nd.name)
                key = re.sub(r"/encoder/layers\.\d+/", "/encoder/layers.N/", nd.name)
                pos[x] = (int(mm.group(1)) if mm else None, key)
    unnamed = [k for k in inits if not k.startswith("enc.")]
    assert len(pos) == len(unnamed), f"소비 노드를 못 찾은 가중치 {len(unnamed) - len(pos)}개"

    print(f"[load] 원본 300m: {A.xlsr300}", flush=True)
    models, _ = checkpoint_utils.load_model_ensemble([A.xlsr300])
    enc = models[0]
    del enc.encoder.layers[LAYER + 1:]
    ref = {k: v.detach().numpy() for k, v in enc.state_dict().items()}

    # ── 위치 키 → 모듈 이름: 동결 구간(원본과 값이 같아야 함)에서 값 일치로 확정 ──
    def candidates(layer, key):
        if key.startswith("/feature_extractor/"):
            ci = re.search(r"conv_layers\.(\d+)", key).group(1)
            return [f"feature_extractor.conv_layers.{ci}.2.1.{p}" for p in ("weight", "bias")]
        if key.startswith("/post_extract_proj/"):
            return ["post_extract_proj.weight"]
        p = f"encoder.layers.{layer}."
        return [p + s for s in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                                "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
                                "fc1.weight", "fc2.weight")]

    def match(arr, name):
        r = ref[name]
        if r.shape == arr.shape and np.array_equal(r, arr):
            return "same"
        if r.ndim == 2 and r.T.shape == arr.shape and np.array_equal(r.T, arr):
            return "T"
        return None

    keymap = {}                                # (위치 키, conv 인덱스 포함) → (모듈 접미사, 전치 여부)
    votes = collections.defaultdict(collections.Counter)
    for x, (layer, key) in pos.items():
        if layer is not None and layer >= 12:
            continue
        hits = [(n, t) for n in candidates(layer, key) for t in [match(inits[x], n)] if t]
        if len(hits) != 1:
            raise SystemExit(f"값 일치로 모듈을 못 정함: {x} {key} 층{layer} → {hits}")
        n, t = hits[0]
        suffix = n.split(f"encoder.layers.{layer}.", 1)[1] if layer is not None else n
        votes[key][(suffix, t)] += 1
    for key, c in votes.items():
        if len(c) != 1:
            raise SystemExit(f"위치 키 {key} 의 대응이 층마다 다르다: {dict(c)}")
        keymap[key] = next(iter(c))
    print("[map] 위치 → 모듈 (동결 구간 값 일치로 확정)")
    for key, (suffix, t) in sorted(keymap.items()):
        if "layers.N" in key:
            print(f"    {key:42} → {suffix}{' (전치)' if t == 'T' else ''}")

    # ── state_dict 조립 ──
    sd = {k: torch.from_numpy(v.copy()) for k, v in ref.items()}
    filled = set()
    for name, arr in inits.items():
        if name.startswith("enc."):
            k = name[4:]
            if k == "encoder.pos_conv.0.weight" and "encoder.pos_conv.0.weight_v" in sd:
                w = torch.from_numpy(arr.copy())
                g = torch.norm_except_dim(w, 2, 2)           # fairseq: weight_norm(conv, dim=2)
                assert g.shape == sd["encoder.pos_conv.0.weight_g"].shape
                sd["encoder.pos_conv.0.weight_g"] = g
                sd["encoder.pos_conv.0.weight_v"] = w
                filled |= {"encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"}
                continue
            assert k in sd and tuple(sd[k].shape) == arr.shape, f"이름 가중치 불일치: {k}"
            sd[k] = torch.from_numpy(arr.copy()); filled.add(k)
        else:
            layer, key = pos[name]
            suffix, t = keymap[key]
            k = f"encoder.layers.{layer}.{suffix}" if layer is not None else suffix
            v = arr.T if t == "T" else arr
            assert tuple(sd[k].shape) == v.shape, f"shape 불일치: {k}"
            sd[k] = torch.from_numpy(np.ascontiguousarray(v)); filled.add(k)
    used = {k for k in sd if k.startswith(("feature_extractor.", "layer_norm.", "post_extract_proj.", "encoder.pos_conv."))
            or re.match(r"encoder\.layers\.\d+\.", k)}
    missing = sorted(used - filled)
    assert not missing, f"ONNX 에서 못 채운 사용 가중치: {missing[:8]}"
    changed = [k for k in filled if not torch.equal(sd[k], torch.from_numpy(ref[k]))]
    by_layer = collections.Counter(re.match(r"encoder\.layers\.(\d+)", k).group(1) if k.startswith("encoder.layers")
                                   else k.split(".")[0] for k in changed)
    print(f"[fill] 채운 텐서 {len(filled)} · 원본과 달라진 텐서 {len(changed)} → {dict(sorted(by_layer.items()))}")
    enc.load_state_dict(sd)
    enc.eval()

    # ── 패리티: PyTorch(weight_norm 유지) vs ONNX ──
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    sess = ort.InferenceSession(A.onnx, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for sec in [3, 8, 12]:
        w = (rng.standard_normal(SR * sec) * 0.05).astype(np.float32)
        with torch.inference_mode():
            res = enc.extract_features(source=torch.from_numpy(w)[None], padding_mask=None, mask=False, layer=LAYER)
            pt = res["layer_results"][LAYER][0].transpose(0, 1).mean(1).numpy()
        on = sess.run(["pooled"], {"source": w[None]})[0]
        d = float(np.abs(pt - on).max())
        worst = max(worst, d)
        print(f"[parity {sec:>2}s] max|Δ|={d:.2e} (pooled 범위 ±{np.abs(on).max():.1f})")
    assert worst < 1e-3, "패리티 실패"

    h = np.load(A.head, allow_pickle=True)
    torch.save({"encoder": enc.state_dict(),
                "head": {"weight": torch.from_numpy(h["W"].astype(np.float32)),
                         "bias": torch.from_numpy(h["b"].astype(np.float32))},
                "mu": h["mu"].astype(np.float32), "sd": h["sd"].astype(np.float32),
                "labels": [str(x) for x in h["labels"]], "layer": LAYER,
                "source": {"onnx": os.path.abspath(A.onnx), "head": os.path.abspath(A.head),
                           "parity_max_abs": worst}}, A.out)
    print(f"[saved] {A.out}")


if __name__ == "__main__":
    main()

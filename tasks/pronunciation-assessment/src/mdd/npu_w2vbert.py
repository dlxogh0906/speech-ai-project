# -*- coding: utf-8 -*-
"""w2v-BERT 2.0 CTC(발음 평가 서비스 모델)를 NPU 컴파일용으로 감싼 모듈 — 불리언 마스크 없이 실수 마스크만 쓴다.

왜: Rebellions 컴파일러(rebel 0.10.2)는 불리언 마스크 연산(masked_fill(~mask), bool 인덱싱)에서
`x and y must have the same dtype: float32 vs bool` 로 멈춘다(국적 인코더에서 확인, 2026-09-26).
HF `Wav2Vec2BertForCTC` 의 하위 모듈(가중치)은 그대로 쓰고, 마스크를 쓰는 세 곳만 곱셈·덧셈으로 바꾼다:
  ① 인코더 입력의 패딩 0 채움  ② 셀프 어텐션의 키 마스크(가산 −1e4)  ③ 합성곱 모듈 앞의 패딩 0 채움.
합성곱은 왼쪽만 보는 인과 합성곱이라 오른쪽 패딩은 유효 프레임에 새지 않는다.

attn="relshift"(기본): HF relative_key 는 층마다 거리 임베딩 표를 [T,T,64] 로 펼쳐(상수) einsum 한다.
층 1개는 컴파일되지만 2층 이상 쌓으면 `Graph Generation: [DEVICE_GRAPH_CONVERSION]` 으로 실패했다(rebel 0.10.2).
그래서 q·Eᵀ([.,T,73])를 먼저 구하고, 양끝 값을 T 번 복제해 붙인 뒤 flatten→오프셋 슬라이스→reshape 로
(l, r) 자리에 E[clamp(r−l, −64, 8)+64] 항을 배치한다(Transformer-XL 의 rel-shift). 상수·gather 가 없고 값은 같다.
attn="hf" 는 HF 모듈을 그대로 호출(대조용).

입력: input_features f32 [1, T, 160] (SeamlessM4T 특징, 뒤쪽 채움) · valid f32 [1, T] (1 = 유효 프레임)
출력: logits f32 [1, T, V] (20 ms 프레임, 유효 구간만 쓴다) — log_softmax·디코딩·GOP 는 호스트에서.
"""
import math

import torch
import torch.nn as nn


class W2VBertCTCNPU(nn.Module):
    def __init__(self, hf_ctc_model, attn="relshift"):
        super().__init__()
        assert attn in ("relshift", "hf")
        self.attn = attn
        self.m = hf_ctc_model.eval()
        enc = self.m.wav2vec2_bert
        assert enc.intermediate_ffn is None and enc.adapter is None, "중간 FFN·어댑터가 있는 설정은 지원하지 않는다"
        assert enc.encoder.embed_positions is None, "relative_key 위치 표현만 지원한다"

    @staticmethod
    def _attn_relshift(sa, h, bias):
        B, L, _ = h.shape
        nh, hd = sa.num_heads, sa.head_size
        q = sa.linear_q(h).view(B, L, nh, hd).transpose(1, 2)            # [B, nh, L, hd]
        k = sa.linear_k(h).view(B, L, nh, hd).transpose(1, 2)
        v = sa.linear_v(h).view(B, L, nh, hd).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1))                           # [B, nh, L, L]
        E = sa.distance_embedding.weight                                   # [K, hd], K = 64 + 8 + 1
        K, left = E.shape[0], sa.left_max_position_embeddings
        qe = torch.matmul(q, E.t())                                        # [B, nh, L, K]
        F = torch.cat([qe[..., :1].expand(B, nh, L, L), qe, qe[..., K - 1:].expand(B, nh, L, L)], dim=-1)
        W = 2 * L + K                                                      # 행 l 의 열 c = L + (r − l + 64) → 양끝은 clamp 값
        flat = torch.nn.functional.pad(F.reshape(B, nh, L * W), (0, left))
        rel = flat[..., L + left: L + left + L * (W - 1)].reshape(B, nh, L, W - 1)[..., :L]
        p = torch.softmax((s + rel) / math.sqrt(hd) + bias, dim=-1)
        o = torch.matmul(p, v).transpose(1, 2).reshape(B, L, nh * hd)
        return sa.linear_out(o)

    @staticmethod
    def _conv(cm, h, vm):
        h = cm.layer_norm(h) * vm                                          # ③ 패딩 0
        h = cm.glu(cm.pointwise_conv1(h.transpose(1, 2)))
        h = torch.nn.functional.pad(h, (cm.depthwise_conv.kernel_size[0] - 1, 0))
        h = cm.depthwise_conv(h)
        h = cm.depthwise_layer_norm(h.transpose(1, 2)).transpose(1, 2)
        h = cm.pointwise_conv2(cm.activation(h))
        return h.transpose(1, 2)

    def forward(self, input_features, valid):
        enc = self.m.wav2vec2_bert
        h, _ = enc.feature_projection(input_features)
        vm = valid.unsqueeze(-1)
        h = h * vm                                                         # ① 패딩 0
        bias = ((valid - 1.0) * 1e4)[:, None, None, :]                     # ② 키 마스크
        for L in enc.encoder.layers:
            r = h; h = L.ffn1(L.ffn1_layer_norm(h)) * 0.5 + r
            r = h; x = L.self_attn_layer_norm(h)
            if self.attn == "relshift":
                h = self._attn_relshift(L.self_attn, x, bias) + r
            else:
                a, _ = L.self_attn(hidden_states=x, attention_mask=bias); h = a + r
            r = h; h = r + self._conv(L.conv_module, h, vm)
            r = h; h = L.ffn2(L.ffn2_layer_norm(h)) * 0.5 + r
            h = L.final_layer_norm(h)
        return self.m.lm_head(h)


def to_bucket(feats, buckets, n_valid=None):
    """feats [T, 160] (numpy) → (padded [1, Tb, 160], valid [1, Tb], n). Tb 는 T 이상인 가장 작은 버킷(없으면 앞 Tb 만).
    n_valid: 특징 추출기 attention_mask 의 합 — 추출기가 끝에 한 프레임을 채우는 경우가 있어 T 가 아니라 이걸 써야 한다."""
    import numpy as np
    T = feats.shape[0] if n_valid is None else int(n_valid)
    Tb = next((b for b in buckets if b >= T), buckets[-1])
    n = min(T, Tb)
    x = np.ones((1, Tb, feats.shape[1]), np.float32)                      # 서비스와 같은 padding_value=1
    x[0, :min(feats.shape[0], Tb)] = feats[:Tb]
    v = np.zeros((1, Tb), np.float32); v[0, :n] = 1.0
    return x, v, n


class RBLNDecoder:
    """`mdd.gop_feats_v2.HFDecoder` 자리에 끼우는 NPU 디코더 — 모델 대신 컴파일된 .rbln 런타임을 쓴다.

    `forward_lprobs(dec, wavs)` 가 kind="rbln" 이면 `dec.lprobs(wavs)` 를 부른다. 강제정렬·GOP·점수가 쓰는
    `.d`(자모 사전)·`.blank` 는 HFDecoder 와 같다. 특징 추출(SeamlessM4T)도 서비스와 같은 호출이다.
    mdir: jamo_vocab.json·preprocessor_config.json 이 있는 폴더(가중치는 안 읽는다)
    rbln_path: `npu_compile_check.py` 산출물. 버킷(프레임)은 파일 이름 `_f192_448_1024.rbln` 에서 읽는다.
    """

    kind = "rbln"

    def __init__(self, mdir, rbln_path, device=0, buckets=None):
        import json, os, re, threading
        import rebel
        from transformers import SeamlessM4TFeatureExtractor
        from mdd.gop_feats_v2 import _JamoDict
        v = json.load(open(os.path.join(mdir, "jamo_vocab.json"), encoding="utf-8"))
        if buckets is None:
            mt = re.search(r"_f(\d+(?:_\d+)*)\.rbln$", os.path.basename(rbln_path))
            assert mt, f"버킷을 파일 이름에서 못 읽었다: {rbln_path}"
            buckets = [int(x) for x in mt.group(1).split("_")]
        self.buckets = sorted(buckets)
        self.fe = SeamlessM4TFeatureExtractor.from_pretrained(mdir)
        self.d = _JamoDict(v["vocab"]); self.blank = v["blank"]; self.normalize = False
        self.rt = rebel.Runtime(rbln_path, device=int(device), tensor_type="np")
        self.device = f"npu:{int(device)}"
        self.model = None
        self._lock = threading.Lock()                # 런타임 1개를 여러 스레드가 동시에 부르지 않게
        print(f"[model] rbln {rbln_path} | npu {device} | 버킷 {self.buckets} | blank_idx={self.blank} vocab={len(self.d)}", flush=True)

    def lprobs(self, wavs):
        """wavs(16k float32 목록) → (log_softmax [B, T, V] torch.float32, 유효 프레임 수 목록)."""
        import numpy as np
        outs = []
        for w in wavs:
            enc = self.fe([w], sampling_rate=16000, return_tensors="np", padding=True)
            n = int(enc["attention_mask"][0].sum())
            x, v, k = to_bucket(enc["input_features"][0], self.buckets, n)
            if k < n:
                print(f"[warn] {n} 프레임 > 최대 버킷 {self.buckets[-1]} — 앞 {k} 프레임만 본다", flush=True)
            with self._lock:
                o = np.asarray(self.rt.run(x, v), np.float32).reshape(1, x.shape[1], -1)[0, :k]
            outs.append(torch.from_numpy(o).log_softmax(-1))
        T = max(o.shape[0] for o in outs)
        lp = torch.zeros(len(outs), T, outs[0].shape[1])
        for i, o in enumerate(outs):
            lp[i, :o.shape[0]] = o
        return lp, [o.shape[0] for o in outs]

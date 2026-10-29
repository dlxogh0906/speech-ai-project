# -*- coding: utf-8 -*-
"""사투리 앙상블(EAT Large + XLS-R 2B)의 두 인코더를 **순수 PyTorch** 로 다시 쓴 NPU 용 모듈 (2026-10-28).

왜: Rebellions 컴파일러(rebel 0.10.2)는 입력 모양이 고정이어야 하고, 불리언 마스크 연산에서 멈춘다
(국적 `country_classifier/npu_encoder.py` 와 같은 이유). 그래서 두 인코더 모두 12초 고정 입력과
실수 마스크(1 = 유효, 0 = 채움)를 받아 곱셈·덧셈만으로 가린다. 가중치 이름은 원래 모델
(fairseq XLS-R 2B · HF EAT 의 `.model`)과 같아서 `pure_state()` 로 옮긴 state_dict 를 그대로 싣는다.

  xlsr2b    : source f32 [1, 192000] · frame_valid f32 [1, 599]        → f32 [1, 2, 1920]  31 · 47층 유효 프레임 평균
  eat_large : fbank  f32 [1, 1, 1200, 128] · patch_valid f32 [1, 600]  → f32 [1, 2, 1024]  15 · 23층 유효 패치 평균

채움이 답을 바꾸지 않는 이유: 합성곱 프레임·패치는 채운 부분과 겹치지 않고(유효 프레임 수 = 채움 없이 잰 프레임 수),
위치 합성곱 앞에서 채움 프레임을 0 으로 만들며(fairseq 도 같다), 어텐션은 채움 키에 −1e4 를 더해 가린다.
fbank 와 헤드는 호스트에서 계산한다(`eat_inputs` · `Head`). NPU 서버에는 torchaudio 가 없어서 kaldi fbank 를 torch 연산으로 다시 썼다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
STATIC_SAMPLES = 12 * SR
CONV = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
EAT_PATCH, EAT_FREQ_PATCHES, EAT_MELS = 16, 8, 128
EAT_SHIFT, EAT_SCALE = 4.268, 2 * 4.569            # Collator(eat_large): (fbank + 4.268) / (2 * 4.569)

SPECS = {
    "xlsr2b": dict(width=1920, layers=48, heads=16, ffn=7680, out_layers=(31, 47)),
    "eat_large": dict(width=1024, layers=24, heads=16, ffn=4096, out_layers=(15, 23)),
}


def conv_out_len(n):
    for _, k, s in CONV:
        n = (n - k) // s + 1
    return n


def fbank_frames(n):
    """kaldi fbank(25 ms 창 · 10 ms 이동 · snip_edges) 프레임 수."""
    return 1 + (n - 400) // 160


XLSR_FRAMES = conv_out_len(STATIC_SAMPLES)                                           # 599
EAT_FRAMES = -(-fbank_frames(STATIC_SAMPLES) // EAT_PATCH) * EAT_PATCH               # 1198 → 1200
EAT_PATCHES = EAT_FRAMES // EAT_PATCH * EAT_FREQ_PATCHES                             # 600


def _attention(q, k, v, heads, bias):
    B, T, D = q.shape
    dh = D // heads
    q = (q * dh ** -0.5).view(B, T, heads, dh).transpose(1, 2)
    k = k.view(B, T, heads, dh).transpose(1, 2)
    v = v.view(B, T, heads, dh).transpose(1, 2)
    w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1)
    return torch.matmul(w, v).transpose(1, 2).reshape(B, T, D)


class _XlsrLayer(nn.Module):
    """fairseq TransformerSentenceEncoderLayer(layer_norm_first=True)."""

    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, n, nn.Linear(d, d))
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x, bias):
        a = self.self_attn
        y = self.self_attn_layer_norm(x)
        x = x + a.out_proj(_attention(a.q_proj(y), a.k_proj(y), a.v_proj(y), self.h, bias))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class PureXLSR(nn.Module):
    def __init__(self, width, layers, heads, ffn, out_layers):
        super().__init__()
        self.out_layers = tuple(out_layers)
        self.feature_extractor = nn.Module()
        self.feature_extractor.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1 if i == 0 else 512, c, k, s), nn.Identity(),
                          nn.Sequential(nn.Identity(), nn.LayerNorm(c), nn.Identity()), nn.GELU())
            for i, (c, k, s) in enumerate(CONV)])
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, width)
        self.encoder = nn.Module()
        self.encoder.pos_conv = nn.Sequential(nn.Conv1d(width, width, 128, padding=64, groups=16))
        self.encoder.layers = nn.ModuleList([_XlsrLayer(width, heads, ffn) for _ in range(max(out_layers) + 1)])

    def forward(self, source, frame_valid):
        x = source.unsqueeze(1)
        for seq in self.feature_extractor.conv_layers:
            x = F.gelu(seq[2][1](seq[0](x).transpose(1, 2)).transpose(1, 2))
        x = self.post_extract_proj(self.layer_norm(x.transpose(1, 2)))           # [B,T,D]
        m = frame_valid.unsqueeze(-1)
        x = x * m                                                                # fairseq: x[padding_mask] = 0
        x = x + F.gelu(self._pos_conv(x.transpose(1, 2))).transpose(1, 2)
        bias = ((frame_valid - 1.0) * 1e4)[:, None, None, :]
        n = m.sum(1).clamp_min(1.0)
        out = []
        for i, layer in enumerate(self.encoder.layers):
            x = layer(x, bias)
            if i in self.out_layers:
                out.append((x * m).sum(1) / n)
        return torch.stack(out, 1)

    def _pos_conv(self, x):
        """그룹 합성곱(16그룹 · 커널 128)을 그룹별 일반 합성곱 16개로 — 값은 같다(torch 에서 차이 0).
        rebel 0.10.2 는 그룹 합성곱을 호스트 CPU 로 넘겨 12초 한 번에 92 ms 가 걸렸다. 나누면 NPU 에서 4.5 ms (10-28 실측)."""
        conv = self.encoder.pos_conv[0]
        g = conv.in_channels // conv.groups
        y = torch.cat([F.conv1d(x[:, i * g:(i + 1) * g], conv.weight[i * g:(i + 1) * g], conv.bias[i * g:(i + 1) * g],
                                padding=conv.padding[0]) for i in range(conv.groups)], 1)
        return y[:, :, :-1]                                                      # SamePad: 마지막 1프레임 제거


class _EatBlock(nn.Module):
    """EAT AltBlock(layer_norm_first=False): 어텐션 → 잔차 → norm1 → MLP → 잔차 → norm2."""

    def __init__(self, d, h, ff, eps=1e-6):
        super().__init__()
        self.h = h
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=eps), nn.LayerNorm(d, eps=eps)
        self.attn = nn.Module()
        self.attn.qkv, self.attn.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x, bias):
        q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
        x = self.norm1(x + self.attn.proj(_attention(q, k, v, self.h, bias)))
        return self.norm2(x + self.mlp.fc2(F.gelu(self.mlp.fc1(x))))


class PureEAT(nn.Module):
    def __init__(self, width, layers, heads, ffn, out_layers):
        super().__init__()
        self.out_layers = tuple(out_layers)
        self.local_encoder = nn.Module()
        self.local_encoder.proj = nn.Conv2d(1, width, EAT_PATCH, EAT_PATCH)
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("positions", torch.zeros(1, EAT_PATCHES, width))    # 고정 sincos 의 앞 600 패치
        self.pre_norm = nn.LayerNorm(width, eps=1e-6)
        self.blocks = nn.ModuleList([_EatBlock(width, heads, ffn) for _ in range(max(out_layers) + 1)])

    def forward(self, fbank, patch_valid):
        x = self.local_encoder.proj(fbank).flatten(2).transpose(1, 2) + self.positions   # 패치 순서 = 시간 × 주파수 8
        x = self.pre_norm(torch.cat((self.extra_tokens.expand(x.shape[0], -1, -1), x), 1))
        valid = torch.cat((torch.ones_like(patch_valid[:, :1]), patch_valid), 1)          # CLS 는 늘 유효
        bias = ((valid - 1.0) * 1e4)[:, None, None, :]
        m = patch_valid.unsqueeze(-1)
        n = m.sum(1).clamp_min(1.0)
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x, bias)
            if i in self.out_layers:
                out.append((x[:, 1:] * m).sum(1) / n)
        return torch.stack(out, 1)


def build(kind):
    return (PureXLSR if kind == "xlsr2b" else PureEAT)(**SPECS[kind]).eval()


def pure_state(kind, model):
    """원래 모델(fairseq Wav2Vec2Model · HF EAT 의 `.model`) → 순수 모듈 state_dict. 빠진 키가 있으면 멈춘다."""
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    if kind == "xlsr2b":
        g, v = sd.get("encoder.pos_conv.0.weight_g"), sd.get("encoder.pos_conv.0.weight_v")
        if g is not None:
            sd["encoder.pos_conv.0.weight"] = torch._weight_norm(v, g, 2)
    else:
        sd["positions"] = sd["fixed_positional_encoder.positions"][:, :EAT_PATCHES].clone()
    want = build(kind).state_dict()
    missing = [k for k in want if k not in sd]
    assert not missing, "빠진 가중치: %s" % missing[:5]
    bad = [k for k in want if tuple(sd[k].shape) != tuple(want[k].shape)]
    assert not bad, "모양이 다르다: %s" % bad[:5]
    return {k: sd[k] for k in want}


def load_pure(kind, path):
    """저장된 state_dict(fp16 이어도 된다)를 fp32 순수 모듈로 싣는다."""
    m = build(kind)
    sd = torch.load(str(path), map_location="cpu")
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return m.eval()


def xlsr_inputs(wav):
    n = min(len(wav), STATIC_SAMPLES)
    source = np.zeros((1, STATIC_SAMPLES), np.float32)
    source[0, :n] = wav[:n]
    valid = np.zeros((1, XLSR_FRAMES), np.float32)
    valid[0, :conv_out_len(n)] = 1.0
    return source, valid


def eat_inputs(wav, mel_banks):
    """PreparedClips + Collator(eat_large) 와 같은 전처리. mel_banks: torchaudio kaldi get_mel_banks(128) 를 오른쪽 0 한 칸 채운 [128, 257]."""
    w = torch.from_numpy(np.ascontiguousarray(wav[:STATIC_SAMPLES], np.float32))
    w = w - w.mean()
    frames = w.unfold(0, 400, 160)                                                # snip_edges
    frames = frames - frames.mean(1, keepdim=True)                                # remove_dc_offset
    frames = frames - 0.97 * torch.cat((frames[:, :1], frames[:, :-1]), 1)       # preemphasis (replicate)
    frames = F.pad(frames * torch.hann_window(400, periodic=False), (0, 112))     # hanning → 512 점
    power = torch.fft.rfft(frames).abs().pow(2.0)
    banks = torch.as_tensor(np.asarray(mel_banks, np.float32))
    fb = torch.clamp_min(torch.mm(power, banks.t()), torch.finfo(torch.float32).eps).log()
    m = fb.shape[0]
    out = torch.zeros(EAT_FRAMES, EAT_MELS)
    out[:m] = fb
    out = (out + EAT_SHIFT) / EAT_SCALE                                           # 채움(0)도 정규화한다 — Collator 와 같다
    valid = np.zeros((1, EAT_PATCHES), np.float32)
    valid[0, :(m + EAT_PATCH - 1) // EAT_PATCH * EAT_FREQ_PATCHES] = 1.0
    return out.numpy()[None, None].astype(np.float32), valid


def inputs(kind, wav, mel_banks=None):
    return xlsr_inputs(wav) if kind == "xlsr2b" else eat_inputs(wav, mel_banks)


class Head:
    """ensemble.DialectEnsemble._logits 의 뒷부분: ±50 자르기 → fp16 반올림 → (f − mu) / sd → 선형."""

    def __init__(self, npz):
        self.w, self.b = npz["weight"].astype(np.float32), npz["bias"].astype(np.float32)
        self.mu, self.sd = npz["mu"].astype(np.float32), npz["sd"].astype(np.float32)

    def __call__(self, pooled):
        f = np.clip(np.asarray(pooled, np.float32).reshape(1, -1), -50, 50).astype(np.float16).astype(np.float32)
        return ((f - self.mu) / self.sd) @ self.w.T + self.b

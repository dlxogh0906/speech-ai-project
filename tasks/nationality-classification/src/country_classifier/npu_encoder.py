# -*- coding: utf-8 -*-
"""XLS-R 300m KD 인코더(≤17층)를 **순수 PyTorch** 로 다시 쓴 추론 전용 모듈 — fairseq·불리언 마스크 없음.

왜: Rebellions 컴파일러(rebel 0.10.2)가 fairseq TorchScript 의 불리언 마스크 연산에서
`x and y must have the same dtype: float32 vs bool` 로 멈춘다. 여기서는 마스크를 실수(frame_valid: 1=유효, 0=채움)로 받아
곱셈·덧셈만으로 처리한다. 가중치 이름은 fairseq 모델(remove_weight_norm 이후)과 같아서 state_dict 를 그대로 싣는다.

입력: source f32 [B, N] (0 채움) · frame_valid f32 [B, T] (T = conv_out_len(N)) → pooled f32 [B, 1024] (유효 프레임 평균)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2


def conv_out_len(n):
    for _, k, s in CONV:
        n = (n - k) // s + 1
    return n


class _Layer(nn.Module):
    def __init__(self, d=1024, h=16, ff=4096):
        super().__init__()
        self.h = h; self.dh = d // h
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, n, nn.Linear(d, d))
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff); self.fc2 = nn.Linear(ff, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x, bias):                     # x [B,T,D] · bias [B,1,1,T] (0 또는 −1e4)
        B, T, D = x.shape
        r = x; y = self.self_attn_layer_norm(x)
        a = self.self_attn
        q = (a.q_proj(y) * (self.dh ** -0.5)).view(B, T, self.h, self.dh).transpose(1, 2)
        k = a.k_proj(y).view(B, T, self.h, self.dh).transpose(1, 2)
        v = a.v_proj(y).view(B, T, self.h, self.dh).transpose(1, 2)
        w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1)
        y = torch.matmul(w, v).transpose(1, 2).reshape(B, T, D)
        x = r + a.out_proj(y)
        r = x; y = self.final_layer_norm(x)
        return r + self.fc2(F.gelu(self.fc1(y)))


class PureEncoder(nn.Module):
    def __init__(self, n_layers=18):
        super().__init__()
        self.feature_extractor = nn.Module()
        convs = []
        for i, (c, k, s) in enumerate(CONV):
            seq = nn.Sequential(nn.Conv1d(1 if i == 0 else 512, c, k, s), nn.Identity(),
                                nn.Sequential(nn.Identity(), nn.LayerNorm(c), nn.Identity()), nn.GELU())
            convs.append(seq)
        self.feature_extractor.conv_layers = nn.ModuleList(convs)
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 1024)
        self.encoder = nn.Module()
        self.encoder.pos_conv = nn.Sequential(nn.Conv1d(1024, 1024, 128, padding=64, groups=16))
        self.encoder.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])

    def forward(self, source, frame_valid):
        x = source.unsqueeze(1)
        for seq in self.feature_extractor.conv_layers:
            x = seq[0](x)
            x = seq[2][1](x.transpose(1, 2)).transpose(1, 2)
            x = F.gelu(x)
        x = self.post_extract_proj(self.layer_norm(x.transpose(1, 2)))       # [B,T,1024]
        m = frame_valid.unsqueeze(-1)
        x = x * m                                                            # fairseq: x[padding_mask] = 0
        pc = self.encoder.pos_conv[0](x.transpose(1, 2))[:, :, :-1]           # SamePad: 마지막 1프레임 제거
        x = x + F.gelu(pc).transpose(1, 2)
        bias = ((frame_valid - 1.0) * 1e4)[:, None, None, :]
        for layer in self.encoder.layers:
            x = layer(x, bias)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


def load_pure(state_path, n_layers=18):
    enc = PureEncoder(n_layers)
    sd = torch.load(state_path, map_location="cpu")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not missing, f"빠진 가중치: {missing[:5]}"
    return enc.eval()

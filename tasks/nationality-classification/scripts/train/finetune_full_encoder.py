# -*- coding: utf-8 -*-
"""XLS-R 인코더 전체 미세조정 — 300m 과 2B 를 같은 레시피로 학습해 크기만의 차이를 본다.

왜: 9/24 의 2B 미세조정은 300m(run_zeroth)과 출발점·손실·학습량이 달라 크기 비교가 되지 않았다
    (results/13_NPU_서빙_준비.md). 여기서는 두 백본 모두 사전학습 원본에서, 같은 데이터·같은 레시피로 학습한다.

레시피 (두 백본 공통)
  - 구조: 사전학습 XLS-R 을 --layer 층까지 자르고(300m 17 · 2B 31 — 배포 모델·9/24 2B 실험과 같은 층)
    그 층 출력을 마스크 평균 → clip ±50 → (x−mu)/sd → Linear(클래스). kd_student.forward_pooled 와 같다.
  - 학습하는 것: 트랜스포머 0~layer 층 전부 + 위치 합성곱 · 특징 투영 · 층 정규화.
    CNN 특징 추출기는 고정한다(wav2vec 2.0 미세조정 관례). --train_cnn 이면 그것까지 푼다.
  - 초기화: 동결 특징 --lp_rows 행(클래스별 같은 수)으로 mu/sd 와 선형 헤드를 먼저 맞춘다(LP-FT). mu/sd 는 고정.
  - 손실 CE · AdamW · warmup → cosine(최저 10%) · 가중치 EMA · bf16 autocast · 녹음 앞 --max_sec 초.
    --class_weight inv | sqrt 면 CE 에 클래스 가중치(N/(C·n_c) 또는 그 제곱근)를 준다 — 클래스 불균형 보정 실험(2026-10-19).
  - --eval_every 스텝마다 valid. 기준(웹마이크 top-5 + macro top-5)이 가장 좋은 EMA 가중치를 best.pt 로,
    --patience 번 연속 안 오르면 멈춘다. last.pt(옵티마이저 포함)로 이어서 학습할 수 있다(--resume).
  - --lora_rank r 이면 인코더를 고정하고 LoRA 만 학습한다. best.pt 에는 LoRA 를 합친 보통 가중치를 쓴다.
    --init_ckpt 로 앞 학습의 best.pt 에서 이어 갈 수 있다(LP 생략, 2026-10-20).
  - --wave_aug crop,speed,gain,eq,noise(또는 all) 면 학습 행에 파형 증강을, --cutmix p 면 배치 CutMix 를 건다
    (country_classifier.wave_aug, 2026-10-20). valid · test · LP 에는 안 건다.
  - --train_audio <목록> 이면 학습 행의 오디오만 그 목록의 경로에서 읽는다 — 미리 만든 증강(음성 변환 등, 2026-10-21).
    LP 초기화 · valid · 배치 길이는 원래 녹음 그대로라, 같은 목록 · 같은 seed 의 대조군과 배치가 똑같다.
  - 운율 실험(2026-10-23, results/42). 셋 다 --init_ckpt 가 있어야 하고, 새로 붙는 가중치는 0 에서 시작해 첫 순간은 원래 모델과 같다.
    · --pool stats: 프레임 평균에 표준편차를 붙인다(표준편차는 학습 행 일부로 잰 mu2/sd2 로 정규화). attn: 프레임 가중치를 배우는
      attentive statistics pooling (점수층 0 시작 = 처음엔 고른 가중치).
    · --pros_tracks <폴더>: 프레임별 음높이 · 유성 · 세기 궤적(prosody_tracks.py)을 작은 1D 합성곱으로 읽어 평균 · 표준편차 128차원을 붙인다.
    · --aux_feats <csv> --aux_w λ: 풀링 특징(평균 부분)에서 발화 운율 특징 22개(prosody_feats.py, 학습 행 기준 표준화)를 맞히는
      보조 회귀를 더한다(값이 없는 행 · 칸은 뺀다).
  - mel 융합 실험(2026-10-29, results/45). --init_ckpt 가 있어야 하고, 운율 실험과 같이 쓰지 않는다. 새 칸은 0 에서 시작.
    · --mel_fuse xlsr: 같은 파형에서 로그 mel 80 을 GPU 에서 만들어(발화별 정규화 · 학습 중 SpecAugment) 2D 합성곱 줄기로
      20 ms 프레임(XLS-R 과 같은 간격)으로 줄인다. XLS-R 층 출력 프레임과 한 시퀀스로 이어 자기 어텐션 --mel_layers 층을
      태운다(두 입력이 서로 본다, 시간 위치 공유 + 입력 종류 임베딩). 유효 토큰 평균 --mel_dim 차원을 헤드 입력에 붙인다.
    · --mel_fuse mel: mel 프레임만 같은 어텐션에 태운다 — 두 입력을 섞는 몫과 mel 정보만 더한 몫을 가르는 대조.
    · 학습 중 발화마다 --mel_drop 확률로 mel 줄기를 끈다(mel 에만 기대지 않게).

    $PY scripts/train/finetune_full_encoder.py --backbone 300m --out_dir $W/full_300m --lr 1e-5
    $PY scripts/train/finetune_full_encoder.py --backbone 2b   --out_dir $W/full_2b   --lr 5e-6
    $PY scripts/train/finetune_full_encoder.py --backbone 300m --out_dir /tmp/x --smoke          # 3천 행 시험
    $PY scripts/train/finetune_full_encoder.py --backbone 300m --data_dir <목록> --lora_rank 8 \
        --init_ckpt $W/full_300m/best.pt --out_dir <폴더>                                       # LoRA 만, 앞 학습에서 이어
    $PY scripts/train/finetune_full_encoder.py --eval_only --ckpt $W/full_300m/best.pt --test <csv> --out <dir>
  ($W = $NATIONALITY_OUTPUTS/webmic_kd, 경로는 tasks/nationality-classification/ 기준)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src"))),
                 str(_r / "tasks" / "nationality-classification" / "experiments")]
from shared.paths import MODEL_ROOT, NAT_OUTPUTS

import argparse, collections, json, math, os, random, time
import numpy as np

# 백본 → (사전학습 파일, 풀링 층)
BACKBONES = {"300m": ("xlsr2_300m.pt", 17), "2b": ("xlsr2_2B_1000k.pt", 31)}
PRETRAIN_ONLY = ("quantizer.", "project_q.", "final_proj.", "target_glu.", "project_inp.")   # 사전학습 손실에만 쓰는 모듈
KOREAN_DOMAINS = ("kspon", "kor001", "kor_aug_test", "zeroth_test")


def load_backbone(path, layer, device):
    import torch
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils
    models, _ = checkpoint_utils.load_model_ensemble([str(path)])
    enc = models[0]
    del enc.encoder.layers[layer + 1:]
    enc.encoder.layerdrop = 0.0                      # layer_results 인덱스가 밀리지 않게
    # 2B 설정은 활성값 재계산(checkpoint_activations)이 켜져 있다. fairseq 의 재계산은 역전파 때 fp16 autocast 로
    # 다시 돌아 bf16 과 충돌하고(layer_norm dtype 오류) 느리다 — 원래 forward 로 되돌린다 (finetune_partial.py 와 같다).
    for m in enc.modules():
        if hasattr(m, "precheckpoint_forward"):
            m.forward = m.precheckpoint_forward
    return enc.to(device)


def set_trainable(enc, train_cnn):
    names = []
    for n, p in enc.named_parameters():
        on = not n.startswith(PRETRAIN_ONLY) and (train_cnn or not n.startswith("feature_extractor."))
        p.requires_grad = on
        if on:
            names.append(n)
    return [p for p in enc.parameters() if p.requires_grad], names


LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2")


def add_lora(enc, rank, alpha):
    """인코더를 고정하고 트랜스포머 층 선형층(q·k·v·out·fc1·fc2)에 LoRA 를 단다 → 학습할 (이름, 파라미터).

    weight 를 W + (alpha/rank)·B·A 로 파라미터화한다(torch parametrize). fairseq 어텐션의 빠른 경로는 q/k/v 가중치를
    모듈 호출 없이 직접 읽으므로, 모듈을 감싸는 방식으로는 LoRA 가 반영되지 않는다. B 는 0 에서 시작한다(처음엔 원래 모델).
    """
    import torch
    from torch.nn.utils import parametrize

    class LoRA(torch.nn.Module):
        def __init__(self, out_f, in_f):
            super().__init__()
            self.A = torch.nn.Parameter(torch.empty(rank, in_f))
            torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = torch.nn.Parameter(torch.zeros(out_f, rank))

        def forward(self, W):
            return W + (alpha / rank) * (self.B @ self.A)

    for p in enc.parameters():
        p.requires_grad = False
    for lay in enc.encoder.layers:
        for t in LORA_TARGETS:
            lin = lay.get_submodule(t)
            parametrize.register_parametrization(lin, "weight", LoRA(*lin.weight.shape).to(lin.weight.device))
    return [(n, p) for n, p in enc.named_parameters() if p.requires_grad]


def plain_state(enc):
    """LoRA 를 원래 가중치에 합친 보통 state_dict — --eval_only 와 다른 스크립트가 그대로 읽는다."""
    from torch.nn.utils import parametrize
    sd = {k: v for k, v in enc.state_dict().items() if ".parametrizations." not in k}
    for name, m in enc.named_modules():
        if parametrize.is_parametrized(m, "weight"):
            sd[name + ".weight"] = m.weight.detach().clone()
    return sd


AUX_KEYS = ["f0_range", "f0_sd", "f0_vel", "f0_decl", "syl_f0_jump", "syl_f0_inner", "final_delta", "final_rise", "int_sd",
            "int_jump", "f0_int_corr", "npvi", "ioi_varco", "final_len", "speech_rate", "artic_rate", "pause_per10", "pause_ratio",
            "hnr", "alpha", "voiced_frac", "varco_voiced"]


def make_xhead(dim, pool, pros):
    """운율 실험의 풀링 · 운율 궤적 모듈 (--pool stats|attn, --pros_tracks). 평균 부분은 원래 정규화(mu/sd)를 그대로 쓴다."""
    import torch

    class ProsEnc(torch.nn.Module):
        def __init__(self, hid=64):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Conv1d(5, hid, 5, padding=2), torch.nn.GELU(),
                                           torch.nn.Conv1d(hid, hid, 5, padding=2), torch.nn.GELU())
            self.norm = torch.nn.LayerNorm(2 * hid)

        def forward(self, p, v):                         # p (B,T,3) · v (B,T) 유효 프레임
            st, vo, it = p[..., 0], p[..., 1], p[..., 2]
            dst = torch.zeros_like(st); dst[:, 1:] = (st[:, 1:] - st[:, :-1]) * vo[:, 1:] * vo[:, :-1]
            dit = torch.zeros_like(it); dit[:, 1:] = it[:, 1:] - it[:, :-1]
            h = self.net(torch.stack([st, vo, it, dst, dit], 1)).transpose(1, 2)    # (B,T,hid)
            w = v.float().unsqueeze(-1); n = w.sum(1).clamp_min(1.0)
            m = (h * w).sum(1) / n
            sd = (((h - m.unsqueeze(1)) ** 2) * w).sum(1).div(n).clamp_min(1e-6).sqrt()
            return self.norm(torch.cat([m, sd], 1))

    class XHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool, self.use_pros = pool, pros
            if pool == "attn":
                self.att = torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))
                torch.nn.init.zeros_(self.att[2].weight); torch.nn.init.zeros_(self.att[2].bias)
            if pros:
                self.pros = ProsEnc()

        def out_extra(self):
            return (dim if self.pool in ("stats", "attn") else 0) + (128 if self.use_pros else 0)

        def forward(self, x, fp, mu, sd, mu2, sd2, ptrack):
            """x (B,T,C) 층 출력 · fp (B,T) 패딩(True) 또는 None → 정규화된 [평균, (표준편차), (운율)]"""
            x = torch.nan_to_num(x.float())
            v = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if fp is None else ~fp
            xn = (x.clamp(-50, 50) - mu) / sd
            if self.pool == "attn":
                a = self.att(xn).squeeze(-1).masked_fill(~v, -1e4).softmax(1).unsqueeze(-1)
            else:
                a = v.float().unsqueeze(-1) / v.float().sum(1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
            m = (a * x).sum(1)
            parts = [(m.clamp(-50, 50) - mu) / sd]
            if self.pool in ("stats", "attn"):
                s = ((a * (x - m.unsqueeze(1)) ** 2).sum(1)).clamp_min(1e-6).sqrt().clamp(0, 50)
                parts.append((s - mu2) / sd2)
            if self.use_pros:
                parts.append(self.pros(ptrack, v))
            return torch.cat(parts, 1)

    return XHead()


def make_melfuse(dim, mode, d=256, layers=2, heads=4, drop=0.2):
    """mel 융합 모듈 (--mel_fuse xlsr|mel). 출력 = [원래 평균 풀링(정규화) dim, 융합 특징 d]."""
    import torch
    import torch.nn.functional as F
    import torchaudio

    def sinusoid(n, k):
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, k, 2, dtype=torch.float32) * (-math.log(10000.0) / k))
        pe = torch.zeros(n, k)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return pe

    class MelFuse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mode, self.drop = mode, drop
            self.mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, win_length=400, hop_length=160, f_min=20.0,
                                                            f_max=7600.0, n_mels=80, center=False, power=2.0)
            self.stem = torch.nn.ModuleList([torch.nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1),
                                             torch.nn.Conv2d(32, 32, 3, stride=(1, 2), padding=1)])
            self.proj_m = torch.nn.Linear(32 * 20, d)
            if mode == "xlsr":
                self.ln_x = torch.nn.LayerNorm(dim)
                self.proj_x = torch.nn.Linear(dim, d)
            self.kind = torch.nn.Parameter(torch.randn(2, d) * 0.02)       # 입력 종류 임베딩 (0 = XLS-R · 1 = mel)
            lay = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu", batch_first=True,
                                                   norm_first=True)
            self.tf = torch.nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
            self.norm = torch.nn.LayerNorm(d)
            self.register_buffer("pe", sinusoid(4096, d), persistent=False)

        def out_extra(self):
            return self.proj_m.out_features

        def logmel(self, wav, pmask):
            """→ 발화별 평균 · 분산 정규화한 로그 mel (B,F,80) · 유효 프레임 수 (B,)"""
            with torch.autocast(device_type=wav.device.type, enabled=False):
                m = torch.log(self.mel(wav.float()) + 1e-6).transpose(1, 2)
                ns = (torch.full((wav.shape[0],), wav.shape[1], device=wav.device) if pmask is None
                      else (~pmask).sum(1))
                nf = ((ns - 400) // 160 + 1).clamp(1, m.shape[1])
                w = (torch.arange(m.shape[1], device=m.device)[None] < nf[:, None]).float().unsqueeze(-1)
                n = w.sum(1, keepdim=True)
                mu = (m * w).sum(1, keepdim=True) / n
                sd = (((m - mu) ** 2) * w).sum(1, keepdim=True).div(n).clamp_min(1e-5).sqrt()
                m = (m - mu) / sd * w
                if self.training:                            # SpecAugment — 주파수 띠 2개(≤10) · 시간 구간 2개(≤40프레임)
                    for i in range(m.shape[0]):
                        L = int(nf[i])
                        for _ in range(2):
                            f = random.randint(0, 10); f0 = random.randint(0, 80 - f); m[i, :, f0:f0 + f] = 0
                            t = random.randint(0, min(40, L // 10)); t0 = random.randint(0, max(0, L - t)); m[i, t0:t0 + t] = 0
            return m, nf

        def forward(self, x, fp, mu, sd, wav, pmask):
            x = torch.nan_to_num(x.float())
            B, T = x.shape[:2]
            vx = torch.ones(B, T, dtype=torch.bool, device=x.device) if fp is None else ~fp
            w = vx.float().unsqueeze(-1)
            base = (((x * w).sum(1) / w.sum(1).clamp_min(1.0)).clamp(-50, 50) - mu) / sd
            m, nf = self.logmel(wav, pmask)
            h = F.gelu(self.stem[0](m.unsqueeze(1)))                      # (B, 32, T', 40) — 20 ms 프레임
            Tm = h.shape[2]
            vm = torch.arange(Tm, device=x.device)[None] < ((nf + 1) // 2)[:, None]
            h = h * vm[:, None, :, None].to(h.dtype)                      # 채운 칸의 bias 가 둘째 합성곱으로 새지 않게
            h = F.gelu(self.stem[1](h)).permute(0, 2, 1, 3).flatten(2)   # (B, T', 32·20)
            h = self.proj_m(h)
            h = h + self.pe[:Tm] + self.kind[1]
            keep = torch.ones(B, dtype=torch.bool, device=x.device)
            if self.training and self.drop > 0:
                keep = torch.rand(B, device=x.device) >= self.drop
            if self.mode == "xlsr":
                hx = self.proj_x(self.ln_x(x.clamp(-50, 50))) + self.pe[:T] + self.kind[0]
                tok, valid = torch.cat([hx, h], 1), torch.cat([vx, vm & keep[:, None]], 1)
            else:
                tok, valid = h, vm
            o = self.tf(tok, src_key_padding_mask=~valid)
            wv = valid.float().unsqueeze(-1)
            f = self.norm(torch.nan_to_num((o * wv).sum(1) / wv.sum(1).clamp_min(1.0)))
            if self.mode == "mel":
                f = f * keep[:, None].float()
            return torch.cat([base, f], 1)

    return MelFuse()


class Tracks:
    """prosody_tracks.py 출력 → 경로별 (T,3) 궤적을 배치 프레임 수에 맞춰 붙인다(없으면 0)."""

    def __init__(self, d):
        self.a = np.load(os.path.join(d, "tracks.npy"), mmap_mode="r")
        self.ix = json.load(open(os.path.join(d, "index.json")))

    def batch(self, paths, T, dev):
        import torch
        out = np.zeros((len(paths), T, 3), np.float32)
        for i, p in enumerate(paths):
            s = self.ix.get(p)
            if s:
                n = min(T, s[1]); out[i, :n] = self.a[s[0]:s[0] + n]
        return torch.from_numpy(out).to(dev)


def make_loss(kind, gamma, smoothing, cw, log_prior=None, tau=1.0):
    """학습 손실 — ce: 교차 엔트로피(라벨 스무딩 smoothing) · focal: −(1−p_t)^gamma·log p_t (Lin et al. 2017)
    · la: 로짓 보정 손실 CE(z + tau·log π, y) (Menon et al. 2021) — 추론은 보정 없는 z 그대로라 적은 클래스 쪽으로 기운다.
    cw(클래스 가중치)가 있으면 배치 안 가중 평균이다(F.cross_entropy 의 weight 와 같은 정의)."""
    import torch.nn.functional as F

    def ce(z, y):
        return F.cross_entropy(z, y, weight=cw, label_smoothing=smoothing)

    def la(z, y):
        return F.cross_entropy(z + tau * log_prior, y, weight=cw, label_smoothing=smoothing)

    def focal(z, y):
        lp = F.log_softmax(z.float(), -1).gather(1, y[:, None]).squeeze(1)
        l = -((1 - lp.exp()) ** gamma) * lp
        if cw is None:
            return l.mean()
        w = cw[y]
        return (l * w).sum() / w.sum()
    return {"focal": focal, "la": la}.get(kind, ce)


def grad_reverse(x, lam):
    """순전파는 그대로, 역전파 기울기에 −lam 을 곱한다 (DANN, Ganin & Lempitsky 2015)."""
    import torch

    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.view_as(t)

        @staticmethod
        def backward(ctx, g):
            return -lam * g
    return _GradReverse.apply(x)


def balanced_subset(rows, n, seed):
    """클래스마다 같은 수(가능한 만큼)를 무작위로 — LP 헤드 초기화용."""
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["label_en"]].append(i)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by)))
    out = []
    for c in sorted(by):
        idx = by[c][:]
        rng.shuffle(idx)
        out += idx[:per]
    return sorted(out)


def fit_linear(X, y, n_cls, device, wd=1e-4, iters=300):
    """표준화한 동결 특징 위 선형 헤드 (L-BFGS, 전체 배치)."""
    import torch
    import torch.nn.functional as F
    X = torch.from_numpy(X).to(device); Y = torch.from_numpy(y).to(device)
    W = torch.zeros(n_cls, X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W.t() + b, Y) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        acc = float(((X @ W.t() + b).argmax(1) == Y).float().mean())
    return W.detach(), b.detach(), acc


def score_all(rows, L, seen, c2i):
    """webmic_kd_train.score(웹 · 데이터셋별 · 문장/단어 · 131 · 한국어) + 한국어 원천별."""
    from webmic_kd_train import score
    out = score(rows, L, seen, c2i, None)
    dom = np.array([r["domain"] for r in rows], dtype=object)
    y = np.array([c2i.get(r["label_en"], -1) for r in rows])
    top5 = np.argsort(-L, 1)[:, :5]
    for d in KOREAN_DOMAINS:
        m = seen & (dom == d) & (y >= 0)
        if m.any():
            out["ko_" + d] = {"n": int(m.sum()), "top1": round(float((top5[m, 0] == y[m]).mean()), 4),
                              "top5": round(float((top5[m] == y[m][:, None]).any(1).mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="300m")
    ap.add_argument("--xlsr", default="", help="사전학습 파일 (기본 $SPEECH_AI_MODELS/<백본 기본 파일>)")
    ap.add_argument("--layer", type=int, default=-1, help="풀링 층 (기본: 300m 17 · 2b 31)")
    ap.add_argument("--data_dir", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth_r16k")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5, help="인코더 학습률")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--max_batch_sec", type=float, default=64.0)
    ap.add_argument("--max_batch", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch_sec", type=float, default=240.0)
    ap.add_argument("--eval_fp32", action="store_true",
                    help="평가(LP 특징 · valid · test)를 bf16 autocast 없이 fp32 로, TF32 도 끈다. "
                         "2B 는 bf16 평가와 fp32 평가의 top-1 이 짧은 발화에서 15%% 다르다 (2026-10-20)")
    ap.add_argument("--eval_every", type=int, default=2000, help="옵티마이저 스텝 단위")
    ap.add_argument("--patience", type=int, default=10, help="valid 기준이 이만큼 연속 안 오르면 멈춘다 (0 = 끝까지)")
    ap.add_argument("--save_last_every", type=int, default=3, help="valid 몇 번마다 last.pt(이어 학습용)를 쓸지")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--mask_prob", type=float, default=0.0, help="학습 중 wav2vec2 시간 마스킹 확률 (0 = 끔)")
    ap.add_argument("--mask_length", type=int, default=10)
    ap.add_argument("--train_cnn", action="store_true", help="CNN 특징 추출기까지 학습")
    ap.add_argument("--lp_rows", type=int, default=20000, help="LP 헤드 초기화에 쓸 학습 행 수")
    ap.add_argument("--class_weight", choices=("none", "inv", "sqrt"), default="none",
                    help="학습 손실의 클래스 가중치 — inv: N/(C·n_c) · sqrt: 그 제곱근. LP 초기화 · 평가에는 안 쓴다")
    ap.add_argument("--loss", choices=("ce", "focal", "la"), default="ce",
                    help="학습 손실 (2026-10-20) — la = 로짓 보정 손실. LP 초기화 · 평가에는 안 쓴다")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--la_tau", type=float, default=1.0, help="--loss la 의 τ (학습 행의 클래스 비율 π 로 z + τ·log π)")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="--loss ce · la 의 라벨 스무딩 ε")
    ap.add_argument("--spk_adv", type=float, default=0.0,
                    help="화자 적대 λ 최대 (0 = 끔). 같은 클래스 안에서 화자를 맞히는 헤드를 붙이고 그 기울기를 뒤집어 "
                         "인코더에 준다 — 몇 사람의 목소리를 외우는 지름길을 막는다. λ 는 0 에서 DANN 식으로 올린다")
    ap.add_argument("--spk_adv_mode", choices=("grl", "confusion"), default="grl",
                    help="grl = 화자 CE 의 기울기를 뒤집는다 (λ0.3 에서 발산했다, 2026-10-21). confusion = 화자 헤드는 따로 CE 로 배우고 "
                         "인코더는 그 헤드의 출력이 같은 클래스 화자들에 고르게 퍼지도록 KL(고름 ‖ p) 를 줄인다 — 0 이 바닥이라 발산하지 않는다")
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="0 = 인코더 전체 학습. >0 이면 인코더를 고정하고 트랜스포머 층 q·k·v·out·fc1·fc2 에 LoRA 만 학습 (2026-10-20)")
    ap.add_argument("--lora_alpha", type=float, default=0.0, help="LoRA 배율 alpha/rank 의 alpha (0 = rank, 배율 1)")
    ap.add_argument("--init_ckpt", default="",
                    help="이 스크립트가 쓴 체크포인트(best.pt)에서 인코더 · 헤드 · mu/sd 를 가져와 이어 학습 (LP 초기화 생략)")
    ap.add_argument("--wave_aug", default="",
                    help="학습 행 파형 증강 (쉼표) — crop 무작위 구간 · speed 속도 · gain 음량 · eq 음색 · noise 잡음, all = 전부. "
                         "valid · test 에는 안 건다 (2026-10-20)")
    ap.add_argument("--aug_p", type=float, default=0.5, help="--wave_aug 증강마다 걸릴 확률 (noise 는 --noise_p)")
    ap.add_argument("--speed", type=float, default=0.1, help="speed: 속도 U(1−s, 1+s) 배 (음높이 · 포먼트도 같이 바뀐다)")
    ap.add_argument("--crop_scale", type=float, default=0.4, help="crop: 길이를 원래(최대 --max_sec)의 U(이 값, 1) 배로")
    ap.add_argument("--crop_min_sec", type=float, default=1.0, help="crop: 이보다 짧게는 자르지 않는다")
    ap.add_argument("--gain_db", type=float, default=6.0, help="gain: ±dB")
    ap.add_argument("--eq_db", type=float, default=6.0, help="eq: 쉘빙 · 피킹 필터 이득 ±dB")
    ap.add_argument("--noise_p", type=float, default=0.3, help="noise: 걸릴 확률")
    ap.add_argument("--noise_snr", default="15,35", help="noise: SNR 범위 dB (백색 · 분홍 반반)")
    ap.add_argument("--cutmix", type=float, default=0.0,
                    help="배치 CutMix 확률 (0 = 끔) — 다른 발화의 한 구간을 오려 붙이고 라벨을 길이 비율로 섞는다. CE 전용")
    ap.add_argument("--cutmix_alpha", type=float, default=1.0, help="붙일 비율 ~ 1 − Beta(alpha, alpha)")
    ap.add_argument("--pool", choices=("mean", "stats", "attn"), default="mean",
                    help="풀링: mean(원래) · stats(평균+표준편차) · attn(attentive statistics) — 운율 실험 2026-10-23")
    ap.add_argument("--pros_tracks", default="", help="프레임 운율 궤적 폴더(prosody_tracks.py) — 주면 운율 1D 합성곱 특징을 붙인다")
    ap.add_argument("--aux_feats", default="", help="발화 운율 특징 csv(prosody_feats.py) — 주면 보조 회귀를 더한다")
    ap.add_argument("--aux_w", type=float, default=0.3, help="--aux_feats 보조 회귀 손실 가중치")
    ap.add_argument("--mel_fuse", choices=("", "xlsr", "mel"), default="",
                    help="mel 융합 — xlsr: XLS-R 프레임 + mel 프레임 자기 어텐션 · mel: mel 프레임만 (2026-10-29)")
    ap.add_argument("--mel_dim", type=int, default=256)
    ap.add_argument("--mel_layers", type=int, default=2)
    ap.add_argument("--mel_drop", type=float, default=0.2, help="학습 중 발화마다 mel 줄기를 끌 확률")
    ap.add_argument("--gpu_mem_frac", type=float, default=0.0,
                    help="이 프로세스의 GPU 메모리 상한(전체 대비 비율, 0 = 제한 없음) — 다른 작업과 GPU 를 나눠 쓸 때")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="학습 행 무작위 표본 수 (0 = 전부)")
    ap.add_argument("--train_audio", default="",
                    help="학습 행의 오디오를 이 목록의 audio_path 에서 읽는다 — 미리 만든 증강(음성 변환 등)을 학습에만 쓴다. "
                         "행 순서 · 라벨 · 화자가 data_dir/train.csv 와 같아야 한다. LP 초기화 · valid · 배치 길이는 원래 녹음 (2026-10-21)")
    ap.add_argument("--resume", action="store_true", help="out_dir/last.pt 에서 이어서")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--test", default="")
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from country_classifier.dataset import AudioCountryDataset, collate_audio_batch
    from country_classifier.kd_student import LengthBucketBatchSampler, forward_pooled, write_loader_csv
    from webmic_kd_train import read_rows

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    amp_eval = dict(amp, enabled=amp["enabled"] and not A.eval_fp32)
    if A.gpu_mem_frac > 0 and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(A.gpu_mem_frac)
    if A.eval_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def run_rows(enc, head, mu, sd, layer, rows, tmp_dir, tag, max_sec, batch_sec, n_cls, pooled_only=False, xh=None, xs=None,
                 std_only=False, mf=None):
        """rows → (logits 또는 pooled, seen). 학습 모드는 바꾸지 않는다 — 호출하는 쪽에서 eval().
        xh(운율 실험 모듈)가 있으면 프레임 출력에서 xh 로 특징을 만든다. xs = (mu2, sd2, tracks).
        mf(mel 융합 모듈)가 있으면 프레임 출력과 파형으로 mf 가 특징을 만든다.
        std_only 면 고른 가중치 표준편차 풀링(정규화 전)을 돌려준다 — mu2/sd2 추정용."""
        tmp = write_loader_csv(rows, os.path.join(tmp_dir, f"_{tag}_loader.csv"))
        ds = AudioCountryDataset(tmp, 16000, min_duration_sec=0.25, max_duration_sec=max_sec)
        bs = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows], max_sec, batch_sec, 128, shuffle=False)
        ld = DataLoader(ds, batch_sampler=bs, num_workers=A.workers, collate_fn=collate_audio_batch, pin_memory=True)
        out, seen = None, np.zeros(len(rows), bool)
        with torch.no_grad():
            for bt in ld:
                if bt is None:
                    continue
                idx = np.array([int(s) for s in bt["speaker_ids"]])
                wav_d, pm_d = bt["waveforms"].to(dev), bt["padding_mask"].to(dev)
                if xh is None and mf is None and not std_only:
                    with torch.autocast(**amp_eval):
                        pooled = forward_pooled(enc, wav_d, pm_d, layer)
                    v = pooled if pooled_only else head((pooled - mu) / sd)
                else:
                    with torch.autocast(**amp_eval):
                        res = enc.extract_features(source=wav_d, padding_mask=pm_d, mask=False, layer=layer)
                    x = torch.nan_to_num(res["layer_results"][layer][0].transpose(0, 1).float())
                    fp = res.get("padding_mask")
                    if mf is not None:
                        v = head(mf(x, fp, mu, sd, wav_d, pm_d))
                    elif std_only:
                        w = (torch.ones(x.shape[:2], device=x.device) if fp is None else (~fp).float()).unsqueeze(-1)
                        m = (x * w).sum(1) / w.sum(1).clamp_min(1.0)
                        v = ((((x - m.unsqueeze(1)) ** 2) * w).sum(1) / w.sum(1).clamp_min(1.0)).clamp_min(1e-6).sqrt().clamp(0, 50)
                    else:
                        pt = xs[2].batch([rows[i]["audio_path"] for i in idx], x.shape[1], dev) if xs[2] is not None else None
                        v = head(xh(x, fp, mu, sd, xs[0], xs[1], pt))
                v = v.float().cpu().numpy()
                if out is None:
                    out = np.zeros((len(rows), v.shape[1]), np.float32)
                out[idx] = v; seen[idx] = True
        return out, seen

    # ── 평가만 ──
    if A.eval_only:
        ck = torch.load(A.ckpt, map_location="cpu")
        # 이 스크립트가 쓴 체크포인트는 xlsr 경로를 담고 있다. KD 학생(run_zeroth 등, kd_student 형식)은 300m 이다.
        xlsr_ck = ck.get("xlsr") or os.path.join(str(MODEL_ROOT), BACKBONES[ck.get("backbone", "300m")][0])
        enc = load_backbone(xlsr_ck, ck["layer"], dev)
        enc.load_state_dict(ck["encoder"])
        labels = list(ck["labels"]); c2i = {c: i for i, c in enumerate(labels)}
        head = torch.nn.Linear(ck["head"]["weight"].shape[1], len(labels)).to(dev)
        head.load_state_dict(ck["head"])
        mu = torch.as_tensor(ck["mu"]).to(dev); sd = torch.as_tensor(ck["sd"]).to(dev)
        xh, xs = None, None
        if ck.get("xh_cfg"):                                 # 운율 실험 체크포인트
            c = ck["xh_cfg"]
            xh = make_xhead(mu.numel(), c["pool"], c["pros"]).to(dev); xh.load_state_dict(ck["xh"]); xh.eval()
            trk = Tracks(A.pros_tracks or c["pros_tracks"]) if c["pros"] else None
            xs = (torch.as_tensor(ck["mu2"]).to(dev) if ck.get("mu2") is not None else None,
                  torch.as_tensor(ck["sd2"]).to(dev) if ck.get("sd2") is not None else None, trk)
        mf = None
        if ck.get("mf_cfg"):                                 # mel 융합 체크포인트
            c = ck["mf_cfg"]
            mf = make_melfuse(mu.numel(), c["mode"], c["d"], c["layers"]).to(dev); mf.load_state_dict(ck["mf"]); mf.eval()
        enc.eval(); head.eval()
        rows = read_rows(A.test)
        os.makedirs(A.out, exist_ok=True)
        L, seen = run_rows(enc, head, mu, sd, ck["layer"], rows, A.out, "test", 12.0, A.eval_batch_sec, len(labels), xh=xh, xs=xs,
                           mf=mf)
        np.savez(os.path.join(A.out, "logits.npz"), ok=seen, L=L, labels=np.array(labels))
        r = score_all(rows, L, seen, c2i)
        json.dump({"ckpt": A.ckpt, "test": A.test, "n": len(rows), "decoded": int(seen.sum()),
                   "eval_dtype": "fp32" if A.eval_fp32 else "bf16", "results": r},
                  open(os.path.join(A.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        return

    # ── 학습 ──
    xlsr = A.xlsr or os.path.join(str(MODEL_ROOT), BACKBONES[A.backbone][0])
    layer = A.layer if A.layer >= 0 else BACKBONES[A.backbone][1]
    if A.smoke:
        A.epochs, A.eval_every, A.warmup, A.limit, A.lp_rows, A.patience = 1, 40, 10, 3000, 1000, 0
    if not A.out_dir:
        raise SystemExit("--out_dir 가 필요하다")
    os.makedirs(A.out_dir, exist_ok=True)
    torch.manual_seed(A.seed); random.seed(A.seed); np.random.seed(A.seed)

    rows_tr = read_rows(os.path.join(A.data_dir, "train.csv"))
    rows_va = read_rows(os.path.join(A.data_dir, "valid.csv"))
    rows_au = read_rows(A.train_audio) if A.train_audio else None
    if rows_au is not None and (len(rows_au) != len(rows_tr) or any(
            a["label_en"] != b["label_en"] or a["speaker"] != b["speaker"] for a, b in zip(rows_au, rows_tr))):
        raise SystemExit(f"--train_audio 의 행이 train.csv 와 다르다 (순서 · 라벨 · 화자) — {A.train_audio}")
    if A.limit and A.limit < len(rows_tr):
        rng = np.random.default_rng(A.seed)
        sel = sorted(rng.choice(len(rows_tr), A.limit, replace=False))
        rows_tr = [rows_tr[i] for i in sel]
        if rows_au is not None:
            rows_au = [rows_au[i] for i in sel]
    if A.smoke:
        rows_va = rows_va[::max(1, len(rows_va) // 500)]
    labels = sorted({r["label_en"] for r in rows_tr})
    c2i = {c: i for i, c in enumerate(labels)}
    rows_va = [r for r in rows_va if r["label_en"] in c2i]
    Y = torch.tensor([c2i[r["label_en"]] for r in rows_tr], dtype=torch.long, device=dev)
    print(f"[data] train {len(rows_tr):,} · valid {len(rows_va):,} · 클래스 {len(labels)} · {A.data_dir}", flush=True)
    if rows_au is not None:
        n_au = sum(a["audio_path"] != b["audio_path"] for a, b in zip(rows_au, rows_tr))
        print(f"[train_audio] 학습 행 {n_au:,}/{len(rows_tr):,} 의 오디오를 바꿔 읽는다 · {A.train_audio}", flush=True)
    cw = None                                            # 클래스 불균형 보정 — 학습 행 평균 가중치가 1 이 되게 맞춘다
    if A.class_weight != "none":
        n_c = torch.bincount(Y, minlength=len(labels)).float()
        cw = len(Y) / (len(labels) * n_c.clamp_min(1))
        cw = cw.sqrt() if A.class_weight == "sqrt" else cw
        cw = cw / cw[Y].mean()
        print(f"[class_weight] {A.class_weight} · {cw.min().item():.2f} ~ {cw.max().item():.2f}", flush=True)
    if A.loss == "focal" and A.label_smoothing > 0:
        raise SystemExit("--loss focal 과 --label_smoothing 은 같이 쓰지 않는다")
    from country_classifier.wave_aug import OPS, AugmentWaveDataset, cutmix_batch
    aug_ops = set(OPS) if A.wave_aug == "all" else set(filter(None, A.wave_aug.split(",")))
    if aug_ops - set(OPS):
        raise SystemExit(f"--wave_aug 는 {','.join(OPS)} 중에서 — {sorted(aug_ops - set(OPS))}")
    if A.cutmix > 0 and (A.loss != "ce" or A.class_weight != "none" or A.spk_adv > 0):
        raise SystemExit("--cutmix 는 --loss ce · 클래스 가중치 없음 · 화자 적대 없이만 쓴다")
    log_prior = None
    if A.loss == "la":                                   # 학습 행의 클래스 비율 π (이 학습이 실제로 보는 분포)
        prior = torch.bincount(Y, minlength=len(labels)).float() / len(Y)
        log_prior = prior.clamp_min(1e-12).log()
    loss_fn = make_loss(A.loss, A.focal_gamma, A.label_smoothing, cw, log_prior, A.la_tau)
    if A.loss != "ce" or A.label_smoothing > 0:
        print(f"[loss] {A.loss}" + {"focal": f" γ={A.focal_gamma}", "la": f" τ={A.la_tau}"}.get(A.loss, "")
              + (f" · 라벨 스무딩 {A.label_smoothing}" if A.label_smoothing > 0 else ""), flush=True)

    enc = load_backbone(xlsr, layer, dev)
    if A.mask_prob > 0:
        enc.mask_prob, enc.mask_length, enc.mask_channel_prob = A.mask_prob, A.mask_length, 0.0
    ck0 = None
    if A.init_ckpt:                                      # 앞 학습에서 이어서 — LoRA 를 달기 전에 보통 키로 읽는다
        ck0 = torch.load(A.init_ckpt, map_location="cpu")
        if list(ck0["labels"]) != labels or ck0["layer"] != layer or ck0.get("backbone", "300m") != A.backbone:
            raise SystemExit(f"--init_ckpt 의 클래스 · 층 · 백본이 이 학습과 다르다 — {A.init_ckpt}")
        enc.load_state_dict(ck0["encoder"])
    if A.lora_rank > 0:
        named = add_lora(enc, A.lora_rank, A.lora_alpha or A.lora_rank)
        enc_names, enc_params = [n for n, _ in named], [p for _, p in named]
    else:
        enc_params, enc_names = set_trainable(enc, A.train_cnn)
    n_all = sum(p.numel() for p in enc.parameters())
    n_tr = sum(p.numel() for p in enc_params)
    print(f"[model] {A.backbone} ≤{layer}층 · 학습 {n_tr / 1e6:.1f}M / 전체 {n_all / 1e6:.0f}M · "
          + (f"LoRA r{A.lora_rank} (나머지 고정)" if A.lora_rank > 0 else f"CNN {'학습' if A.train_cnn else '고정'}")
          + f" · 마스킹 {A.mask_prob}", flush=True)
    head = torch.nn.Linear(enc.encoder.layers[0].self_attn.embed_dim, len(labels)).to(dev)

    last_path = os.path.join(A.out_dir, "last.pt")
    resume = torch.load(last_path, map_location="cpu") if A.resume and os.path.exists(last_path) else None
    if resume:
        enc.load_state_dict(resume["encoder"]); head.load_state_dict(resume["head"])
        mu = torch.as_tensor(resume["mu"]).to(dev); sd = torch.as_tensor(resume["sd"]).to(dev)
        print(f"[resume] step {resume['step']} · ep {resume['ep']} · batch {resume['bi']}", flush=True)
    elif ck0 is not None:
        head.load_state_dict(ck0["head"])
        mu = torch.as_tensor(ck0["mu"]).to(dev); sd = torch.as_tensor(ck0["sd"]).to(dev)
        print(f"[init] {A.init_ckpt} 의 인코더 · 헤드 · mu/sd 에서 시작 (LP 생략)", flush=True)
        del ck0
    else:
        # LP: 동결 특징으로 mu/sd · 헤드 초기화
        t0 = time.time()
        sub = [rows_tr[i] for i in balanced_subset(rows_tr, A.lp_rows, A.seed)]
        enc.eval()
        X, seen = run_rows(enc, None, None, None, layer, sub, A.out_dir, "lp", A.max_sec, A.eval_batch_sec, 0, pooled_only=True)
        enc.train()
        X, yl = X[seen], np.array([c2i[r["label_en"]] for r in sub])[seen]
        mu_np, sd_np = X.mean(0), X.std(0) + 1e-5
        W, b, acc = fit_linear(((X - mu_np) / sd_np).astype(np.float32), yl, len(labels), dev)
        head.weight.data.copy_(W); head.bias.data.copy_(b)
        mu = torch.from_numpy(mu_np.astype(np.float32)).to(dev); sd = torch.from_numpy(sd_np.astype(np.float32)).to(dev)
        print(f"[lp] {int(seen.sum()):,}행 · 학습 정확도 {acc:.3f} · {time.time() - t0:.0f}s", flush=True)

    xh, xs, aux_head, AUXY = None, (None, None, None), None, None
    if A.pool != "mean" or A.pros_tracks:                # 운율 실험 — 새 가중치는 0 에서 시작(첫 순간은 원래 모델)
        if not A.init_ckpt or resume:
            raise SystemExit("--pool stats|attn · --pros_tracks 는 --init_ckpt 에서 새로 시작할 때만 쓴다")
        dim = head.in_features
        xh = make_xhead(dim, A.pool, bool(A.pros_tracks)).to(dev)
        new = torch.nn.Linear(dim + xh.out_extra(), len(labels)).to(dev)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :dim] = head.weight; new.bias.copy_(head.bias)
        head = new
        mu2 = sd2 = None
        if A.pool != "mean":                             # 표준편차 부분의 정규화 값 — 학습 행 일부(클래스 균형)로
            t0 = time.time()
            sub = [rows_tr[i] for i in balanced_subset(rows_tr, min(A.lp_rows, 8000), A.seed)]
            enc.eval()
            S, seen = run_rows(enc, None, None, None, layer, sub, A.out_dir, "std", A.max_sec, A.eval_batch_sec, 0, std_only=True)
            enc.train()
            S = S[seen]
            mu2 = torch.from_numpy(S.mean(0).astype(np.float32)).to(dev)
            sd2 = torch.from_numpy((S.std(0) + 1e-5).astype(np.float32)).to(dev)
            print(f"[pool] {A.pool} · 표준편차 정규화 {int(seen.sum()):,}행 · {time.time() - t0:.0f}s", flush=True)
        xs = (mu2, sd2, Tracks(A.pros_tracks) if A.pros_tracks else None)
        if A.pros_tracks:
            print(f"[pros] 궤적 {len(xs[2].ix):,}개 · 학습 행 중 있는 것 {sum(r['audio_path'] in xs[2].ix for r in rows_tr):,}/{len(rows_tr):,}", flush=True)
        print(f"[xhead] 헤드 입력 {dim} → {head.in_features} (늘어난 칸은 0 에서 시작)", flush=True)
    mf = None
    if A.mel_fuse:                                       # mel 융합 — 새 가중치 · 늘어난 헤드 칸은 0 에서 시작(첫 순간은 원래 모델)
        if not A.init_ckpt or resume or xh is not None or A.aux_feats:
            raise SystemExit("--mel_fuse 는 --init_ckpt 에서 새로 시작할 때만, 운율 실험(--pool · --pros_tracks · --aux_feats) 없이 쓴다")
        dim = head.in_features
        mf = make_melfuse(dim, A.mel_fuse, A.mel_dim, A.mel_layers, drop=A.mel_drop).to(dev)
        new = torch.nn.Linear(dim + mf.out_extra(), len(labels)).to(dev)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :dim] = head.weight; new.bias.copy_(head.bias)
        head = new
        print(f"[mel] {A.mel_fuse} · mel 80 → 20 ms 프레임 · 자기 어텐션 {A.mel_layers}층 d{A.mel_dim} · "
              f"{sum(p.numel() for p in mf.parameters()) / 1e6:.2f}M · 학습 중 끄기 {A.mel_drop} · "
              f"헤드 입력 {dim} → {head.in_features} (늘어난 칸은 0 에서 시작)", flush=True)
    if A.aux_feats:                                      # 운율 보조 회귀 — 학습 행 순서로 정답 · 마스크
        import pandas as pd
        fa = pd.read_csv(A.aux_feats).drop_duplicates("audio_path").set_index("audio_path")
        fa = fa[fa.ok == 1]
        fa["speech_rate"] = fa.n_syl / fa.span; fa["artic_rate"] = fa.n_syl / (fa.span - fa.pause_time).clip(lower=0.3)
        fa["pause_per10"] = 10 * fa.pause_n / fa.n_syl; fa["pause_ratio"] = fa.pause_time / fa.span
        Z = fa.reindex([r["audio_path"] for r in rows_tr])[AUX_KEYS].astype(float)
        lo, hi = Z.quantile(0.01), Z.quantile(0.99)
        Z = Z.clip(lo, hi, axis=1); Z = (Z - Z.mean()) / Z.std()
        AUXY = torch.tensor(Z.values, dtype=torch.float32, device=dev)
        aux_head = torch.nn.Linear(head.weight.shape[1] if xh is None else mu.numel(), len(AUX_KEYS)).to(dev)
        print(f"[aux] 운율 특징 {len(AUX_KEYS)}개 · 정답 있는 학습 행 {int((~torch.isnan(AUXY)).any(1).sum()):,}/{len(rows_tr):,} · λ {A.aux_w}", flush=True)
    xparams = ((list(xh.parameters()) if xh is not None else []) + (list(mf.parameters()) if mf is not None else [])
               + (list(aux_head.parameters()) if aux_head is not None else []))
    groups = [{"params": enc_params, "lr": A.lr}, {"params": list(head.parameters()) + xparams, "lr": A.head_lr}]
    base_lrs = [A.lr, A.head_lr]
    spk_head = None
    if A.spk_adv > 0:                                    # 화자 적대 — 같은 클래스 화자끼리만 고르게 해 국적 단서는 남긴다
        spks = sorted({r["speaker"] for r in rows_tr}); s2i = {s: i for i, s in enumerate(spks)}
        SPK = torch.tensor([s2i[r["speaker"]] for r in rows_tr], dtype=torch.long, device=dev)
        cls_of = [0] * len(spks)
        for r in rows_tr:
            cls_of[s2i[r["speaker"]]] = c2i[r["label_en"]]
        spk_cls = torch.tensor(cls_of, dtype=torch.long, device=dev)
        spk_head = torch.nn.Linear(head.in_features, len(spks)).to(dev)
        groups.append({"params": list(spk_head.parameters()), "lr": A.head_lr})
        base_lrs.append(A.head_lr)
        how = {"grl": "기울기 반전", "confusion": "고른 분포로 혼동 (KL)"}[A.spk_adv_mode]
        print(f"[spk_adv] 화자 {len(spks):,}명 · 같은 클래스 안 화자 맞히기를 {how}으로 방해 · λ 최대 {A.spk_adv}", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=A.wd)
    ema_src = enc_params + list(head.parameters()) + xparams
    ema_buf = [p.detach().clone() for p in ema_src] if A.ema > 0 else None
    step, start_ep, start_bi, best, bad, hist = 0, 0, 0, -1.0, 0, {"cfg": vars(A), "evals": []}
    if resume:
        opt.load_state_dict(resume["opt"])
        if ema_buf is not None:
            for e, v in zip(ema_buf, resume["ema"]):
                e.copy_(v.to(dev))
        step, start_ep, start_bi, best, bad = resume["step"], resume["ep"], resume["bi"], resume["best"], resume["bad"]
        hist = resume["hist"]
        del resume

    def ema_update():
        d = min(A.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema_buf, d)
            torch._foreach_add_(ema_buf, [p.detach() for p in ema_src], alpha=1 - d)

    def ema_swap():
        if ema_buf is None:
            return
        with torch.no_grad():
            for e, p in zip(ema_buf, ema_src):
                tmp = p.detach().clone(); p.copy_(e); e.copy_(tmp)

    def save_model(path, **extra):
        torch.save({"backbone": A.backbone, "xlsr": xlsr, "layer": layer,
                    "encoder": plain_state(enc) if A.lora_rank > 0 else enc.state_dict(),
                    "head": head.state_dict(), "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(), "labels": labels,
                    "cfg": vars(A), **xextra(), **extra}, path)

    def xextra():
        if mf is not None:
            return {"mf": mf.state_dict(), "mf_cfg": {"mode": A.mel_fuse, "d": A.mel_dim, "layers": A.mel_layers}}
        if xh is None:
            return {}
        return {"xh": xh.state_dict(), "xh_cfg": {"pool": A.pool, "pros": bool(A.pros_tracks), "pros_tracks": A.pros_tracks},
                "mu2": None if xs[0] is None else xs[0].cpu().numpy(), "sd2": None if xs[1] is None else xs[1].cpu().numpy()}

    tmp_tr = write_loader_csv(rows_au if rows_au is not None else rows_tr, os.path.join(A.out_dir, "_train_loader.csv"))
    if aug_ops:                                          # 자르지 않고 읽어 증강이 앞 max_sec 초로 자른다(crop 이면 무작위 구간)
        aug_cfg = {"ops": aug_ops, "p": A.aug_p, "speed": A.speed, "crop_scale": A.crop_scale, "crop_min_sec": A.crop_min_sec,
                   "gain_db": A.gain_db, "eq_db": A.eq_db, "noise_p": A.noise_p,
                   "snr": tuple(float(v) for v in A.noise_snr.split(","))}
        ds_tr = AugmentWaveDataset(AudioCountryDataset(tmp_tr, 16000, min_duration_sec=0.25, max_duration_sec=0),
                                   aug_cfg, A.max_sec, seed=A.seed)
        print(f"[aug] 파형 증강 {','.join(o for o in OPS if o in aug_ops)} · 확률 {A.aug_p} (잡음 {A.noise_p}) · "
              f"속도 ±{A.speed} · 자르기 ×U({A.crop_scale}, 1) ≥{A.crop_min_sec}s · 음량 ±{A.gain_db} dB · "
              f"음색 ±{A.eq_db} dB · SNR {A.noise_snr} dB", flush=True)
    else:
        ds_tr = AudioCountryDataset(tmp_tr, 16000, min_duration_sec=0.25, max_duration_sec=A.max_sec)
    mix_rng = random.Random(A.seed + 1)
    if A.cutmix > 0:
        print(f"[cutmix] 배치 확률 {A.cutmix} · 붙일 비율 1 − Beta({A.cutmix_alpha}, {A.cutmix_alpha})", flush=True)
    bs_tr = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows_tr], A.max_sec, A.max_batch_sec,
                                     A.max_batch, shuffle=True, seed=A.seed)
    steps_per_ep = math.ceil(len(bs_tr) / A.grad_accum)
    total = steps_per_ep * A.epochs
    print(f"[sched] 배치 {len(bs_tr):,}/epoch · 스텝 {steps_per_ep:,}/epoch · 총 {total:,} · warmup {A.warmup}", flush=True)

    def set_lr():
        if step < A.warmup:
            f = (step + 1) / A.warmup
        else:
            f = A.min_lr_ratio + (1 - A.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - A.warmup) / max(1, total - A.warmup))))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    def evaluate(tag):
        ema_swap(); enc.eval(); head.eval()
        for m_ in (xh, mf):
            if m_ is not None:
                m_.eval()
        L, seen = run_rows(enc, head, mu, sd, layer, rows_va, A.out_dir, "valid", 12.0, A.eval_batch_sec, len(labels),
                           xh=xh, xs=xs, mf=mf)
        r = score_all(rows_va, L, seen, c2i)
        crit = r["web"]["top5"] + r["web"]["macro5"] if "web" in r else -1.0
        print(f"    [{tag}] 기준 {crit:.4f} · " + " · ".join(
            f"{k} " + "/".join(str(v) for kk, v in d.items() if kk != "n") for k, d in r.items()), flush=True)
        return r, crit

    def after_eval(r, crit, ep, bi):
        """best 저장(EMA 가중치가 올라가 있는 상태) → 원래 가중치로 되돌림 → 기록·last.pt."""
        nonlocal best, bad
        hist["evals"].append({"step": step, "ep": ep, "crit": round(crit, 4), **r})
        if crit > best:
            best, bad = crit, 0
            save_model(os.path.join(A.out_dir, "best.pt"), step=step, valid=r)
            print(f"    ✓ best (step {step}, 기준 {crit:.4f})", flush=True)
        else:
            bad += 1
        ema_swap(); enc.train(); head.train()
        for m_ in (xh, mf):
            if m_ is not None:
                m_.train()
        json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if A.save_last_every and len(hist["evals"]) % A.save_last_every == 0:
            torch.save({"encoder": enc.state_dict(), "head": head.state_dict(), "mu": mu.cpu().numpy(),
                        "sd": sd.cpu().numpy(), "opt": opt.state_dict(),
                        "ema": [e.cpu() for e in ema_buf] if ema_buf is not None else [],
                        "step": step, "ep": ep, "bi": bi, "best": best, "bad": bad, "hist": hist}, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

    if not hist["evals"]:
        r, crit = evaluate("valid 시작 (LP 헤드)")
        after_eval(r, crit, 0, 0)
    stop = False
    enc.train(); head.train()
    for ep in range(start_ep, A.epochs):
        bs_tr.set_epoch(ep)
        if aug_ops:
            ds_tr.set_epoch(ep)                          # DataLoader 를 epoch 마다 새로 만들므로 작업자에게 전해진다
        batches = bs_tr._batches()
        skip = start_bi if ep == start_ep else 0
        ld = DataLoader(ds_tr, batch_sampler=batches[skip:], num_workers=A.workers, collate_fn=collate_audio_batch,
                        pin_memory=True, persistent_workers=False)
        t0, done, agg = time.time(), 0, collections.defaultdict(float)
        opt.zero_grad(set_to_none=True)
        for bi, bt in enumerate(ld, start=skip):
            if bt is None:
                continue
            idx = torch.tensor([int(s) for s in bt["speaker_ids"]], device=dev)
            y = Y[idx]
            wav = bt["waveforms"].to(dev, non_blocking=True)
            pmask = bt["padding_mask"].to(dev, non_blocking=True)
            if A.cutmix > 0:
                wav, perm, lam = cutmix_batch(wav, pmask, mix_rng, A.cutmix, A.cutmix_alpha)
            with torch.autocast(**amp):
                res = enc.extract_features(source=wav, padding_mask=pmask, mask=A.mask_prob > 0, layer=layer)
                x = res["layer_results"][layer][0].transpose(0, 1)
                fp = res.get("padding_mask")
                if fp is None:
                    pooled = x.mean(1)
                else:
                    v = (~fp).to(x.dtype).unsqueeze(-1)
                    pooled = (x * v).sum(1) / v.sum(1).clamp_min(1.0)
            if mf is not None:
                feat = mf(x, fp, mu, sd, wav, pmask)
            elif xh is None:
                pooled = torch.nan_to_num(pooled.float()).clamp(-50, 50)
                feat = (pooled - mu) / sd
            else:
                pt = xs[2].batch([rows_tr[i]["audio_path"] for i in idx.tolist()], x.shape[1], dev) if xs[2] is not None else None
                feat = xh(x, fp, mu, sd, xs[0], xs[1], pt)
            z = head(feat)
            if A.cutmix > 0:                                     # 원래 라벨 lam · 붙인 발화의 라벨 1 − lam
                ce_ = lambda t: F.cross_entropy(z, t, reduction="none", label_smoothing=A.label_smoothing)
                loss = (lam * ce_(y) + (1 - lam) * ce_(y[perm])).mean()
            else:
                loss = loss_fn(z, y)                             # weight 가 있으면 배치 안 가중 평균
            if spk_head is not None:
                lam_adv = A.spk_adv * (2.0 / (1.0 + math.exp(-10.0 * min(1.0, step / max(1, total)))) - 1.0)
                off = spk_cls[None, :] != y[:, None]             # 다른 클래스 화자는 가린다
                if A.spk_adv_mode == "grl":
                    zs = spk_head(grad_reverse(feat, lam_adv)).masked_fill(off, -1e4)
                    loss = loss + F.cross_entropy(zs, SPK[idx])
                else:                                            # 헤드는 떼어 낸 특징으로 배우고, 인코더는 멈춘 헤드를 통해 KL 만 받는다
                    zs = spk_head(feat.detach()).masked_fill(off, -1e4)
                    ze = F.linear(feat, spk_head.weight.detach(), spk_head.bias.detach()).masked_fill(off, -1e4)
                    n_in = (~off).sum(1).clamp_min(1).float()
                    kl = -(F.log_softmax(ze.float(), 1).masked_fill(off, 0.0).sum(1) / n_in) - n_in.log()
                    loss = loss + F.cross_entropy(zs, SPK[idx]) + lam_adv * kl.mean()
                    with torch.no_grad():
                        agg["kl"] += float(kl.sum())
                with torch.no_grad():
                    agg["spk_acc"] += float((zs.argmax(1) == SPK[idx]).sum())
            if aux_head is not None:                         # 운율 보조 회귀 (평균 부분 특징에서)
                ty = AUXY[idx]; ok = ~torch.isnan(ty)
                pa = aux_head(feat[:, :aux_head.in_features])
                aux = ((pa - ty.nan_to_num()) ** 2 * ok).sum() / ok.sum().clamp_min(1)
                loss = loss + A.aux_w * aux
                with torch.no_grad():
                    agg["aux"] += float(aux) * len(idx)
            (loss / A.grad_accum).backward()
            if (bi + 1) % A.grad_accum == 0:
                if step == 0:                                    # zero_grad 전에 봐야 한다
                    unused = [n for n, p in zip(enc_names, enc_params) if p.grad is None]
                    print(f"[note] 기울기가 안 오는 학습 파라미터 {len(unused)}개 {unused[:5]}", flush=True)
                set_lr()
                torch.nn.utils.clip_grad_norm_(ema_src, 5.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if ema_buf is not None:
                    ema_update()
                if A.eval_every and step % A.eval_every == 0:
                    r, crit = evaluate(f"valid ep{ep} step{step}")
                    after_eval(r, crit, ep, bi + 1)
                    if A.patience and bad >= A.patience:
                        print(f"[stop] {A.patience}번 연속 기준이 안 올랐다", flush=True)
                        stop = True
                        break
            B = len(idx); done += B
            with torch.no_grad():
                agg["loss"] += float(loss) * B; agg["acc"] += float((z.argmax(1) == y).sum())
            if bi % 200 == 0:
                sp = done / max(time.time() - t0, 1e-9)
                print(f"  ep{ep} batch {bi:,}/{len(batches):,} step {step:,} ({sp:.0f} utt/s) loss {agg['loss'] / max(done, 1):.3f}"
                      f" · acc {agg['acc'] / max(done, 1):.3f} · lr {opt.param_groups[0]['lr']:.2e}"
                      f" · GPU {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.1f} GB"
                      + (f" · 화자 acc {agg['spk_acc'] / max(done, 1):.3f}" if spk_head is not None else "")
                      + (f" · KL {agg['kl'] / max(done, 1):.3f}" if spk_head is not None and A.spk_adv_mode == "confusion" else "")
                      + (f" · 보조 MSE {agg['aux'] / max(done, 1):.3f}" if aux_head is not None else ""),
                      flush=True)
        if stop:
            break
        r, crit = evaluate(f"valid ep{ep} 끝")
        after_eval(r, crit, ep + 1, 0)
        if A.patience and bad >= A.patience:
            print(f"[stop] {A.patience}번 연속 기준이 안 올랐다", flush=True)
            break
    ema_swap()
    save_model(os.path.join(A.out_dir, "final_ema.pt"), step=step)
    print(f"[done] best 기준 {best:.4f} · {A.out_dir}", flush=True)


if __name__ == "__main__":
    main()

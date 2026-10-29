# -*- coding: utf-8 -*-
"""공개 영어 음소 인식 모델(HF CTC)을 같은 모양으로 감싼다 — 어떤 모델이든 ARPAbet 음소열과 음소 사후확률을 낸다.

    rec = PhoneRecognizer(model_dir, kind="espeak")      # kind: espeak · arpa39 · arpa · slplab
    lp = rec.log_probs(wav16k)                           # (프레임, 모델 어휘) — 영어가 아닌 토큰은 막았다
    phones, spans = rec.greedy(lp)                       # ARPAbet 음소열 + 음소마다 (시작, 끝) 프레임
    P, names = rec.phone_post(lp)                        # (프레임×2, blank + ARPAbet …) 로 합친 사후확률(로그)

모델마다 출력 기호가 다르다(espeak IPA 392종 · TIMIT 39 · CMU 39 · 한국인 학습자용 확장). `mdd_en.phones` 가
ARPAbet 으로 옮긴다. espeak 모델은 다국어 어휘라 **영어 토큰만 남기고 막는다**(나머지 확률을 0 으로).
장치는 박지 않는다 — `device` 를 안 주면 GPU 가 있으면 GPU.
"""

import json
import os

import numpy as np

from mdd_en.phones import ARPA39, token_map

FRAME_SEC = 0.02                                          # wav2vec2 계열 출력 간격
POST_FRAME_SEC = FRAME_SEC / 2                            # phone_post 는 프레임을 2배로 늘린다
PHONE_NAMES = ["<b>"] + ARPA39 + ["DX", "Q", "EU", "X*"]   # phone_post 의 열(0 = blank)


def _find(path, name):
    p = os.path.join(path, name)
    return p if os.path.exists(p) else None


class PhoneRecognizer:
    def __init__(self, model_dir, kind, device=None, vocab_dir=None):
        import torch
        from transformers import AutoModelForCTC
        self.torch = torch
        self.kind = kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCTC.from_pretrained(model_dir).to(self.device).eval()
        vp = _find(vocab_dir or model_dir, "vocab.json")
        if vp is None:
            raise FileNotFoundError("vocab.json 이 없다: %s (charsiu 는 tokenizer 저장소를 vocab_dir 로)" % model_dir)
        self.vocab = json.load(open(vp, encoding="utf-8"))
        pre = _find(model_dir, "preprocessor_config.json")
        self.normalize = json.load(open(pre)).get("do_normalize", True) if pre else True
        self.blank = int(self.model.config.pad_token_id if self.model.config.pad_token_id is not None else 0)
        self.map = token_map(self.vocab, kind)            # id → [ARPAbet …]
        V = self.model.config.vocab_size
        allow = np.zeros(V, bool)
        allow[self.blank] = True
        for i in self.map:
            if i < V:
                allow[i] = True
        self.allow = torch.tensor(allow, device=self.device)
        # ARPAbet 음소 사후확률 합치기용: 토큰 → 음소.
        # espeak 은 'ɔːɹ'(AO R)·'ɑːɹ'·'əl' 같은 **두 음소 토큰을 한 프레임**에만 낸다. 확률을 반씩 나누면 CTC 경로가
        # 두 음소를 한 프레임에 못 넣어 한쪽이 "빠진 것"이 되고, 맞게 읽은 for·or·are 가 틀렸다고 나왔다(리뷰 2026-10-04).
        # → 구성 음소마다 토큰 확률을 **그대로** 주고 다시 정규화한 뒤, 프레임을 2배로 늘려(같은 프레임 반복)
        #   두 음소가 이어진 두 프레임에 들어갈 수 있게 한다. 그래서 사후확률의 프레임 간격은 POST_FRAME_SEC(10 ms).
        # 마지막 'X*' 칸은 표준 기호가 아닌 소리(slplab 의 '_err' 등)를 모아 확률이 사라지지 않게 한다.
        self.names = list(PHONE_NAMES)
        idx = dict((p, k) for k, p in enumerate(self.names))
        M = np.zeros((V, len(self.names)), np.float32)
        M[self.blank, 0] = 1.0
        for i, ph in self.map.items():
            if i >= V:
                continue
            for p in ph:
                M[i, idx.get(p, idx["X*"])] = 1.0
        self.M = torch.tensor(M, device=self.device)
        self.post_frame_sec = POST_FRAME_SEC

    def log_probs(self, wav):
        """16 kHz float 파형 → (T, V) 로그확률. 영어가 아닌 토큰은 -inf 로 막고 다시 정규화한다."""
        torch = self.torch
        x = np.asarray(wav, np.float32)
        if self.normalize:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x)[None].to(self.device)).logits[0].float()
            logits = logits.masked_fill(~self.allow[: logits.shape[-1]], float("-inf"))
            return torch.log_softmax(logits, -1)

    def greedy(self, lp):
        """→ (ARPAbet 음소열, [(시작 프레임, 끝 프레임)]). 여러 음소 토큰은 같은 구간을 나눠 갖는다."""
        ids = lp.argmax(-1).tolist()
        phones, spans, prev, start = [], [], None, 0
        for t, i in enumerate(ids + [-1]):
            if i != prev:
                if prev is not None and prev != self.blank and prev in self.map:
                    for p in self.map[prev]:
                        phones.append(p)
                        spans.append((start, t))
                prev, start = i, t
        return phones, spans

    def phone_post(self, lp):
        """(T, V) 로그확률 → (2T, blank + ARPAbet + DX/Q/EU/X*) 로그확률.

        같은 음소로 가는 토큰 확률을 더하고(여러 음소 토큰은 각 음소에 전부), 프레임마다 다시 정규화한 뒤
        프레임을 2배로 늘린다(POST_FRAME_SEC 간격).
        """
        torch = self.torch
        P = torch.exp(lp) @ self.M
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-10)
        P = P.repeat_interleave(2, dim=0)
        return torch.log(P.clamp_min(1e-10)), self.names

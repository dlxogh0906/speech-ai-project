# -*- coding: utf-8 -*-
"""CTC 강제 정렬 — 녹음마다 대사 텍스트로 직접 정렬해 음절 경계를 찾는다.

계획서 6.4 는 원본 경계를 DTW 로 녹음에 옮기지만, 대사를 알고 있으니 녹음에도 정렬을 바로 돌린다
(DTW 와의 비교는 `scripts/eval/validate_synthetic.py` 의 경계 오차 항목).
CTC 출력은 뾰족해서(토큰 하나가 1~2 프레임) 토큰 위치는 "앵커"로만 쓰고, 음절 사이 경계는
`analyze.py` 가 앵커 사이에서 쉼·에너지 골짜기로 다시 잡는다.

모델: 발음 task 의 원어민 Stage 1 w2v-BERT 2.0 CTC(KsponSpeech, KoG2P 자모 표기).
transformers 는 학습 환경 핀을 지키려고 별도 폴더로 얹는다 —
    PYTHONPATH=/data/speech-ai/tools/hf_py38:...  (README 의 실행 절)
"""
import json
import os

import numpy as np

FRAME = 0.02            # w2v-BERT 2.0 출력 프레임 = 필터뱅크 10 ms × stride 2
OFFSET = 0.0075         # 프레임 k 의 시간 = k·20ms + 7.5ms (25ms 창 2개를 쌓은 중심 − 10ms)


def default_ckpt():
    v = os.environ.get("MIMIC_ALIGN_CKPT")
    if v:
        return v
    from shared.paths import DATA_ROOT
    return os.fspath(DATA_ROOT / "checkpoints" / "w2vbert_stage1" / "step_10000")


class Aligner:
    def __init__(self, ckpt=None, device=None):
        import torch
        try:
            from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC
        except ImportError as e:
            raise ImportError("transformers 가 없습니다. 학습 환경엔 일부러 안 넣었다 — "
                              "PYTHONPATH 에 hf_py38 폴더를 얹어 실행하세요(README 실행 절).") from e
        self.ckpt = ckpt or default_ckpt()
        if device is None:
            device = "cuda" if torch.cuda.is_available() and not os.environ.get("MIMIC_FORCE_CPU") else "cpu"
        self.device = device
        self.fe = SeamlessM4TFeatureExtractor.from_pretrained(self.ckpt)
        self.model = Wav2Vec2BertForCTC.from_pretrained(self.ckpt).eval().to(device)
        voc = json.load(open(os.path.join(self.ckpt, "jamo_vocab.json"), encoding="utf-8"))
        self.vocab, self.blank = voc["vocab"], int(voc.get("blank", 0))

    def forward(self, wav, layers=()):
        """wav(16 kHz float32) → (log 확률 [T, V] cpu float32, {층: 은닉 [T, 1024] numpy}).

        은닉 층은 경계 투영용 DTW 특징이다(analyze.crosscheck) — 화자가 바뀌어도 발음 내용이 비슷하게 보인다."""
        import torch
        enc = self.fe(np.asarray(wav, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = self.model(input_features=enc["input_features"].to(self.device),
                             attention_mask=enc["attention_mask"].to(self.device), output_hidden_states=bool(layers))
        lp = torch.log_softmax(out.logits[0].float(), -1).cpu()
        hid = {L: out.hidden_states[L][0].float().cpu().numpy() for L in layers}
        return lp, hid

    def log_probs(self, wav):
        """wav(16 kHz float32) → log 확률 [T, V] (cpu float32)."""
        return self.forward(wav)[0]

    def token_ids(self, syl_tokens):
        """음절별 토큰 목록 → (토큰 id, 그 토큰의 음절 번호, 토큰 이름) 목록 셋."""
        ids, owner, names = [], [], []
        for si, toks in enumerate(syl_tokens):
            for t in toks:
                if t in self.vocab:                     # 사전에 없는 겹받침 등은 건너뛴다(모음은 늘 있다)
                    ids.append(self.vocab[t])
                    owner.append(si)
                    names.append(t)
        return ids, owner, names

    def align(self, wav, syl_tokens, lp=None):
        """음절별 토큰 목록 → 토큰마다 dict(syl, tok, start, end, prob). 시간은 초.

        `prob` 는 그 토큰 프레임들의 평균 사후확률 — 정렬이 억지로 끼워 넣은 토큰이면 낮다."""
        import torch
        import torchaudio.functional as AF
        if lp is None:
            lp = self.log_probs(wav)
        ids, owner, names = self.token_ids(syl_tokens)
        T = int(lp.shape[0])
        need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)   # 같은 토큰 연속엔 blank 1개 필요
        if need > T:
            raise ValueError("녹음(%.2f초)이 대사(%d토큰)에 비해 너무 짧습니다" % (T * FRAME, len(ids)))
        path, scores = AF.forced_align(lp[None], torch.tensor([ids], dtype=torch.int32), blank=self.blank)
        spans = AF.merge_tokens(path[0], scores[0].exp(), blank=self.blank)
        if len(spans) != len(ids):
            raise RuntimeError("정렬 토큰 수 불일치 %d≠%d" % (len(spans), len(ids)))
        return [dict(syl=owner[k], tok=names[k], start=sp.start * FRAME + OFFSET, end=sp.end * FRAME + OFFSET,
                     prob=float(sp.score)) for k, sp in enumerate(spans)]

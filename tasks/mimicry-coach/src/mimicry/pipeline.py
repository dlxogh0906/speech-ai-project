# -*- coding: utf-8 -*-
"""원본·녹음 한 쌍을 끝까지 — 대사 찾기 → 분석 → 경계 교차 확인 → 비교 → 교정 문장. CLI·보고서·서버가 같이 쓴다.

원본은 한 번만 분석해 두고(`Coach.analyze`) 녹음마다 `Coach.compare` 를 부르면 된다. 녹음은 먼저 통째로 받아 적어 대사 자리를
찾고(locate.locate_line), 앞뒤에 다른 말이 있으면 그 사이 조용한 곳에서 잘라 대사 부분만 분석한다(locate.trim_bounds).
잘랐으면 녹음 분석에 `crop`(원래 녹음에서의 시작·끝 초)이 붙는다 — 보고서는 그 구간만 들려준다.
"""
from .align import FRAME, OFFSET, Aligner
from .analyze import analyze, crosscheck, line_ids
from .compare import compare
from .feedback import describe, retake, unsure_note
from .locate import locate_line, trim_bounds

DTW_LAYER = 12        # 경계 교차 확인용 DTW 특징 층 (scripts/eval/validate_synthetic.py 결과로 고른다)
SR = 16000
TRIM_NOTE = {"before": "앞에 붙은 말은 빼고 대사 부분만 봤어요", "after": "뒤에 붙은 말은 빼고 대사 부분만 봤어요",
             "both": "앞뒤에 붙은 말은 빼고 대사 부분만 봤어요"}
NOISY_NOTE = 20.0     # 녹음의 추정 신호대잡음비(analyze 의 snr_db)가 이보다 낮으면 알린다 — phone 조건 11 dB · 깨끗한 낭독 44 dB
NOISY_TEXT = "주변 소리가 커서 쉼·세기는 확실히 못 잴 수 있어요 — 조용한 곳에서 녹음하면 더 정확해요"


class Coach:
    def __init__(self, aligner=None, layer=DTW_LAYER):
        self.al = aligner or Aligner()
        self.layer = layer

    def forward(self, wav):
        return self.al.forward(wav, [self.layer] if self.layer else [])

    def analyze(self, wav, text, baseline_hz=None, fwd=None):
        """→ (분석 dict, DTW 특징 [T, D] 또는 None). fwd: 이미 돌린 (log 확률, 은닉) — 다시 안 돌린다."""
        lp, hid = fwd if fwd is not None else self.forward(wav)
        return analyze(wav, text, self.al, lp=lp, baseline_hz=baseline_hz), hid.get(self.layer)

    def take(self, wav, text, baseline_hz=None):
        """녹음 → (분석, DTW 특징, 분석한 소리). 대사 자리를 찾아 앞뒤 군말을 잘라 내고, 대사 확인 값(line_jer · line_heard ·
        line_absent)은 통째 받아 적기의 대사 부분 매칭으로 채운다(군말은 세지 않는다)."""
        fwd = self.forward(wav)
        ids, owner, n_syl = line_ids(text, self.al)
        loc = locate_line(fwd[0], ids, owner, n_syl, self.al.blank)
        a, b = trim_bounds(wav, loc, SR)
        cut_a, cut_b = a > 0.0, b < len(wav) / float(SR)
        if cut_a or cut_b:
            wav = wav[int(round(a * SR)):int(round(b * SR))]
            fwd = None                                               # 자른 소리로 다시 돌린다
        B, FB = self.analyze(wav, text, baseline_hz=baseline_hz, fwd=fwd)
        B.update(line_jer=round(loc["jer"], 3), line_heard=round(loc["heard"], 3), line_absent=loc["absent"],
                 line_extra=len(loc["before"]) + len(loc["after"]) >= 2)
        if cut_a or cut_b:
            B["crop"] = [round(a, 3), round(b, 3)]
            B["trimmed"] = "both" if cut_a and cut_b else ("before" if cut_a else "after")
        return B, FB, wav

    def compare(self, ref_wav, take_wav, text, ref=None, top=3, baseline_hz=None, check=True):
        """→ (원본 분석, 녹음 분석, 비교 결과). 비교 결과의 selected 항목엔 교정 문장(`say`)이, 다시 녹음해야 하면
        quality 에 안내 문장(`say`)이, 녹음 앞뒤를 잘라 냈으면 quality 에 알림(`note`)이 붙는다."""
        A, FA = ref if ref is not None else self.analyze(ref_wav, text)
        B, FB, take_wav = self.take(take_wav, text, baseline_hz=baseline_hz)
        if check:
            feats = (FA, FB) if (FA is not None and FB is not None) else None
            crosscheck(ref_wav, A, take_wav, B, feats=feats, hop=FRAME, off=OFFSET)
        C = compare(A, B, top=top)
        syl = [s["text"] for s in A["syllables"]]
        word_of = [s["word"] for s in A["syllables"]]
        for iss in C["issues"]:
            iss["say"] = describe(iss, syl, word_of=word_of)
        if not C["quality"]["ok"]:
            C["quality"]["say"] = retake(C["quality"], A["words"], syl, word_of)
        notes = [TRIM_NOTE[B["trimmed"]]] if B.get("trimmed") else []
        unsure = [row["i"] for row in C["rows"] if row["uncertain"]]
        if C["quality"]["grade"] == "pass" and unsure:              # "거의 같아요"는 잰 음절 얘기다 — 못 잰 곳을 밝힌다
            notes.append(unsure_note(unsure, syl, word_of))
        if B.get("snr_db") is not None and B["snr_db"] < NOISY_NOTE:
            notes.append(NOISY_TEXT)
        if notes:
            C["quality"]["note"] = " · ".join(notes)
        return A, B, C

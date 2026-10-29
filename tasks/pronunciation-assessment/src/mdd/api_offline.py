# -*- coding: utf-8 -*-
"""발음 API(`deploy/pronunciation-demo/app.py` · `/sent.eval.kr`)의 채점 경로를 서버 없이 그대로 돌린다 — 평가용.

API 가 화면에 보내는 판정(errorType · uncertain · 점수)을 오프라인에서 재려고 만들었다. 계산은 app.py 의
`load_all` · `canonical` · `check_audio` · `align` · `analyze` 와 SpeechSuperFormatter.build 를 그대로 옮겼다.
**app.py 를 바꾸면 여기도 맞춰야 한다**(이 모듈은 app.py 를 import 하지 않는다 — 데모 코드에 기대지 않기 위해).

다른 점은 하나: `assess(..., canon=…)` 로 표준발음을 바꿔 넣을 수 있다(예: KoG2P 대신 사람 라벨 표준발음으로 채점).

    emu = ApiEmulator(hf_ckpt, scorer=…)          # 기본값은 8932 서버와 같은 보정표·눈금·유창성 규칙
    body, raw = emu.assess(wav16k, "안녕하세요")    # body = /sent.eval.kr 응답과 같은 모양
"""

import os
import re

import numpy as np

from shared.paths import G2P_TOOLS, MDD_OUTPUTS, ensure_kog2p

SR = 16000
MIN_VOICED_RATIO, MIN_VOICED_SEC = 0.30, 1.0          # app.py 와 같은 값
MIN_VOICED_SEC_PER_SYL, MIN_VOICED_SEC_FLOOR = 0.08, 0.2   # 2026-10-18 — 최소 말한 시간을 기준 문장 음절 수에 맞춘다(app.py)


def min_voiced_sec(text):
    """app.py min_voiced_sec 과 같다. 한글 음절이 없으면(또는 text=None) 예전 기준 1.0초."""
    n = len(re.findall(r"[가-힣]", text or ""))
    return MIN_VOICED_SEC if n == 0 else min(MIN_VOICED_SEC, max(MIN_VOICED_SEC_FLOOR, MIN_VOICED_SEC_PER_SYL * n))
_SP = os.path.join(str(MDD_OUTPUTS), "score_phoneme")


class ApiEmulator:
    def __init__(self, hf_ckpt, scorer=None, norms=os.path.join(_SP, "norms.json"),
                 calib=os.path.join(_SP, "calibration.json"), flu_rule=os.path.join(_SP, "fluency_rule.json"),
                 utt_pct=os.path.join(_SP, "utt_percentile.json"), utt_comp=False, decoder=None):
        import sys
        ensure_kog2p()
        argv, sys.argv = sys.argv, sys.argv[:1]            # g2p.py 는 import 시점에 argv 를 파싱한다
        try:
            from g2p import readRules, graph2prono, ver_info
        finally:
            sys.argv = argv
        from mdd.jamo_to_kog2p import phones_to_jamo, jamo_to_phones, INV, VOWEL
        from mdd.gop_feats_v2 import HFDecoder, forward_lprobs, utt_features
        from mdd.session import SessionScorer, phone_context
        from mdd.speechsuper import SpeechSuperFormatter
        from mdd.vad import trim, voiced_mask
        from mdd.pipeline import ctype
        self.g2p, self.rules = graph2prono, readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))
        self.p2j, self.j2p, self.inv, self.vowels = phones_to_jamo, jamo_to_phones, INV, set(VOWEL)
        self.dec = decoder or HFDecoder(hf_ckpt)
        self.fwd, self.feats, self.trim, self.voiced_mask = forward_lprobs, utt_features, trim, voiced_mask
        self.scorer = SessionScorer(scorer, norms)
        self.ss = SpeechSuperFormatter(norms, jamo_to_phones=jamo_to_phones, flu_rule_path=flu_rule, utt_comp=utt_comp,
                                       phone_context=phone_context, ctype=ctype, calib_path=calib, utt_pct_path=utt_pct)
        self.sz = None                                     # ㅅㅆ·ㅈㅉㅊ 판별기 — load_sz() 로 켠다(speechsuper.SZ_RULE 도 켜야 화면에 쓴다)

    def load_sz(self, path):
        from mdd.sz_detector import Detector
        self.sz = Detector(path) if path else None
        return self

    def canonical(self, sentence):
        """app.py canonical() 과 같은 식 — 문장 → KoG2P 표준발음 자모열(ㅙ·ㅞ 는 둘로 나눈다)."""
        text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", sentence or "")).strip()
        if not text:
            return []
        ri, ro = self.rules
        out = []
        for j in self.p2j([p for p in self.g2p(text, ri, ro).split() if p in self.inv]):
            out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
        return out

    def check_audio(self, wav, text=None):
        """text = 기준 문장(app.py 처럼 최소 말한 시간을 정한다). None 이면 예전 기준(1.0초) — 10-18 전 평가와 같은 값."""
        m, _ = self.voiced_mask(wav, SR)
        ratio = float(m.mean()) if m.size else 0.0
        sec = float(m.sum()) * 0.03
        return (sec < min_voiced_sec(text) or ratio < MIN_VOICED_RATIO), {"voiced_ratio": round(ratio, 3), "voiced_sec": round(sec, 2)}

    @staticmethod
    def align(canon, hyp):
        import edlib
        c, h = "".join(canon), "".join(hyp)
        if not c:
            return []
        if not h:
            return [{"i": i, "canon": x, "pred": None, "j": None, "verdict": "삭제"} for i, x in enumerate(canon)]
        a = edlib.getNiceAlignment(edlib.align(c, h, task="path"), c, h)
        out, i, j = [], 0, 0
        for x, y in zip(a["query_aligned"], a["target_aligned"]):
            if x == "-":
                j += 1
                continue
            out.append({"i": i, "canon": x, "pred": None if y == "-" else y, "j": None if y == "-" else j,
                        "verdict": "일치" if x == y else ("삭제" if y == "-" else "다름")})
            i += 1
            j += y != "-"
        return out

    def lprobs(self, wav):
        """무음 다듬기 → 로그확률(numpy). 같은 녹음을 표준발음만 바꿔 여러 번 채점할 때 한 번만 부른다."""
        w, vad = self.trim(wav[: SR * 20], SR)
        lp, Ls = self.fwd(self.dec, [w])
        return lp[0, :Ls[0]].numpy(), len(w) / SR, vad

    def assess(self, wav, sentence, canon=None, lp=None, gate=True):
        """→ (응답 body, raw). check_audio 에 걸리면 (None, {"rejected": 정보}). lp = lprobs() 결과를 넘기면 재사용.
        gate=False 면 말소리 검사를 건너뛴다(자리별 판정만 재고 싶을 때 — API 는 거절한다)."""
        bad, info = self.check_audio(wav[: SR * 20], sentence)
        if bad and gate:
            return None, {"rejected": info}
        l, dur, vad = lp if lp is not None else self.lprobs(wav)
        dec = self.dec
        canon = list(canon) if canon is not None else self.canonical(sentence)
        y = [dec.d.index(t) for t in canon]
        if canon and dec.d.unk() in y:
            canon, y = [], []
        if canon:
            feat, extra = self.feats(dec, l, y, canon, dur)
        else:
            import itertools
            g = l.argmax(-1)
            toks = [int(v) for v, _ in itertools.groupby(g) if v != dec.blank]
            feat, extra = {}, {"pred": [dec.d[t] for t in toks if 0x3131 <= ord(dec.d[t][0]) <= 0x3163], "gops_m": []}
        jamo = [t for t in extra["pred"] if len(t) == 1 and 0x3131 <= ord(t[0]) <= 0x3163]
        sites = self.align(canon, jamo)
        from mdd.gop_feats_v2 import attach_pred_probs
        attach_pred_probs(dec, l, sites, jamo)                 # 화면 지적 예외(speechsuper.STRONG_MIN_PROB)용
        if getattr(self, "sz", None) is not None and canon and extra.get("segs"):   # ㅅㅆ·ㅈㅉㅊ 판별기(app.py --sz-detector 와 같다)
            from mdd.sz_detector import annotate
            w, _ = self.trim(wav[: SR * 20], SR)                   # 로그확률을 낸 바로 그 녹음
            annotate(sites, w, l, extra["segs"], canon, dec.d.index, extra["frame_sec"], self.sz, blank=dec.blank)
        cph, pph = (self.j2p(canon) if canon else []), self.j2p(jamo)
        for s in sites:
            s["canon_ph"] = cph[s["i"]] if len(cph) == len(canon) else None
            s["pred_ph"] = pph[s["j"]] if (s["j"] is not None and len(pph) == len(jamo)) else None
        scores = self.scorer.utt_scores(feat) if feat else None
        body = self.ss.build(sentence, canon, sites, extra.get("gops_m") or [], utt_scores=scores, vowels=self.vowels,
                             feats=feat)
        return body, {"canon": canon, "sites": sites, "scores": scores, "vad": vad}

    @staticmethod
    def phonemes(body):
        """응답의 글자별 음소를 표준발음 순서대로 편다(build 는 표준발음 자리마다 하나씩 만든다)."""
        return [p for w in body["result"]["words"] for p in w["phonemes"]]

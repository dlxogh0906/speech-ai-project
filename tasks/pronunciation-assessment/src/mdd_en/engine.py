# -*- coding: utf-8 -*-
"""영어 발음 평가 엔진 — 문장 + 녹음 → 음소·단어·문장 판정(가공 전 결과). 응답 모양은 `mdd_en.response`.

    eng = EnglishAssessor(model_dir, "espeak", scorer_dir)
    raw = eng.assess(wav16k, "I think the weather is nice today.")

흐름(한국어 엔진과 같은 순서)
  문장 →(CMUdict/g2p_en)→ 단어별 발음 후보
  녹음 →(공개 음소 CTC)→ 음소 사후확률 ─┬→ 단어마다 소리에 맞는 발음 후보 고르기(the: ðə/ði …)
                                         ├→ GOP 특징(강제정렬 + 정렬 없는 CTC-GOP) → 음소 점수(사람 점수 회귀)
                                         ├→ 받아쓰기(greedy) ↔ 표준발음 정렬 → 들린 소리 · 빠뜨림 · 덧붙임
                                         └→ 문장 특징(속도·쉼·점수 분포) → 문장 점수(정확도·유창성·운율·종합)
문장 점수: 점수기 폴더에 `sent_scorer_ko.joblib` 이 있으면 **한국인 037 사람 점수 기준**(`mdd_en.sent_score`)으로
  응답 4칸을 채운다(raw["scores_ko"]). 없으면 speechocean762 기준(예전 — 한국인에게 발음이 96±2.5 로 몰렸다).
지적 규칙(policy v2, 2026-10-06 — 점수기 폴더의 `flag_policy.json` 이 있을 때)
  ① 음소 점수 < T  (speechocean762 기준 문턱 — `phone_scorer.joblib` 의 flag_rule)
  ② **한국인 대표 오류 자리**(f→p · v→b · r↔l · z→s · θ→s · ɪ↔iː · ɛ↔æ …, `mdd_en.l1_ko_errors`)에서
     받아쓰기가 그 한국인식 소리였고 LLR(한국인식 − 표준)이 규칙별 문턱을 넘으면 — 점수와 상관없이 지적.
     문턱은 037 고득점(조음 ≥ 4.5) 화자에서 5% 만 걸리게(`en_calibrate_pair_margins.py`, 0 아래로는 안 내림).
     ð→d 는 뺀다(구분력 AUC 0.50 — 사람 주석과 소리가 무관).
  보여 줄 소리(actual): ② 는 그 한국인식 소리. ① 만 걸렸을 때는 빠뜨림이면 빠뜨림, 대표 오류 소리로 들렸으면
  그 소리, 그 밖에는 **소리를 지어내지 않는다**(actual_src = unclear). 지적하지 않은 자리는 대표 오류 소리로
  들렸을 때만 힌트(uncertain)를 남기고, 나머지 불일치(예: 맞게 읽은 the 의 ð→d)는 보여 주지 않는다.
  `flag_policy.json` 이 없으면 예전 규칙(v1: ① 만, 불일치는 모두 uncertain 힌트)으로 돈다.

거절하는 입력(ValueError 계열 — 서버가 400 으로 바꾼다)
  NoWords       refText 에 읽을 영어 단어가 없다
  TextTooLong   단어 MAX_WORDS · 음소 MAX_PHONES 초과 — 정렬 없는 GOP 가 음소 수의 제곱으로 커진다
  AlignError    녹음이 문장보다 너무 짧아 정렬이 안 된다(그대로 채점하면 특징이 상수가 되어 엉뚱한 점수가 나왔다)
"""

import os
import re
import threading

import numpy as np

from mdd_en.align import align
from mdd_en.g2p import EnglishG2P, strip_stress
from mdd_en.gop import AlignError, choose_variants, gop_features      # AlignError: 서버가 400 으로 바꾼다
from mdd_en.phones import ARPA39, VOWELS, fold, same_sound
from mdd_en.recognizer import PhoneRecognizer
from mdd_en.scorer import COLS, phone_table
from mdd_en.utterance import utt_features


MAX_WORDS = 60          # 30초 녹음 상한에 맞춘 값(분당 120단어)
MAX_PHONES = 240


class NoWords(ValueError):
    pass


class TextTooLong(ValueError):
    pass


class EnglishAssessor:
    def __init__(self, model_dir, kind, scorer_dir, vocab_dir=None, device=None, policy=None, display_scale="human"):
        import json
        import joblib
        self.rec = PhoneRecognizer(model_dir, kind, device=device, vocab_dir=vocab_dir)
        self.kind = kind
        self.g2p = EnglishG2P()
        ps = joblib.load(os.path.join(scorer_dir, "phone_scorer.joblib"))
        self.reg = ps["reg"]
        rule = ps["flag_rule"]
        if not str(rule.get("name", "")).startswith("score<T") or rule.get("T") is None:
            raise ValueError("이 엔진은 '점수 < T' 지적 규칙만 안다: %s" % rule)
        self.flag_T = float(rule["T"])
        self.flag_rule = rule
        sp = os.path.join(scorer_dir, "sent_scorer.joblib")
        self.sent = joblib.load(sp)["regs"] if os.path.exists(sp) else {}
        kp = os.path.join(scorer_dir, "sent_scorer_ko.joblib")     # 한국인 037 사람 점수 기준(있으면 응답 4칸에 쓴다)
        if os.path.exists(kp):
            from mdd_en.sent_score import KoreanSentenceScorer
            self.ko = KoreanSentenceScorer(kp, display=display_scale)
        else:
            self.ko = None
        fp = os.path.join(scorer_dir, "flag_policy.json")
        self.pair_margins, self.policy_info = {}, None
        if policy != "v1" and os.path.exists(fp):
            pol = json.load(open(fp, encoding="utf-8"))
            floor = float(pol.get("llr_floor", 0.0))
            off = set(pol.get("disabled", []))
            self.pair_margins = dict((k, max(floor, float(v))) for k, v in pol["pair_margins"].items() if k not in off)
            self.policy_info = pol
        self.policy = "v2" if self.pair_margins else "v1"
        names = self.rec.names
        self.idx = dict((p, i) for i, p in enumerate(names))
        self.names = names
        self.alt = sorted(set(self.to_ids(ARPA39)))
        self.lock = threading.Lock()                     # 모델 하나를 여러 요청이 같이 쓴다

    def to_ids(self, phones):
        return [self.idx[p] for p in fold(phones, self.kind)]

    def warmup(self):
        self.assess(np.zeros(16000, np.float32) + 1e-4 * np.random.RandomState(0).randn(16000).astype(np.float32),
                    "hello world")

    def _flag(self, phones, T):
        """지적과 보여 줄 소리(actual)를 정한다 — 규칙은 모듈 설명(policy v1 / v2)."""
        if self.policy == "v1":
            for q in phones:
                q["flagged"] = bool(q["score"] < T)
                actual, src = q["asr"], "asr"
                if q["flagged"] and not q["mismatch"]:
                    actual, src = q["likely"], "gop"
                elif actual is not None and same_sound(fold([q["phone"]], self.kind)[0], actual, q["stressed"]):
                    actual = q["phone"]
                q["actual"], q["actual_src"] = actual, src
            return phones
        for q in phones:
            by_score, by_pair = bool(q["score"] < T), bool(q.get("pair_hit"))
            q["flagged"] = by_score or by_pair
            if by_pair:
                actual, src = q["pair_alt"], "pair"
            elif by_score:
                if q["asr"] is None:
                    actual, src = None, "asr"                    # 받아쓰기도 소리를 못 들었다 → 빠뜨림
                elif q.get("pair_cand"):
                    actual, src = q["pair_alt"], "asr"           # 대표 오류 소리로 들렸다(근거는 문턱 아래)
                else:
                    actual, src = q["phone"], "unclear"          # 무엇으로 냈는지 근거가 없다 — 지어내지 않는다
            else:
                actual, src = (q["pair_alt"], "asr") if q.get("pair_cand") else (q["phone"], "asr")
            q["actual"], q["actual_src"] = actual, src
        return phones

    def reflag(self, raw, T):
        """같은 채점 결과를 다른 지적 문턱으로 — 문턱별 비교(평가)용. raw 는 바꾸지 않는다."""
        import copy
        out = copy.deepcopy(raw)
        self._flag(out["phones"], T)
        return out

    def assess(self, wav, ref_text):
        """→ dict(words, phones, sentence, timing). ref_text 에 읽을 단어가 없으면 ValueError."""
        words = self.g2p.words(ref_text)
        if not words:
            raise NoWords("읽을 영어 단어가 없다")
        n_ph = sum(len(w["variants"][0]) for w in words)
        if len(words) > MAX_WORDS or n_ph > MAX_PHONES:
            raise TextTooLong("단어 %d개 · 음소 %d개 (상한 %d · %d)" % (len(words), n_ph, MAX_WORDS, MAX_PHONES))
        with self.lock:
            lp = self.rec.log_probs(wav)
            post, _ = self.rec.phone_post(lp)
            hyp, _ = self.rec.greedy(lp)
            variants = [[[strip_stress(p) for p in v] for v in w["variants"]] for w in words]
            pick = choose_variants(post, variants, self.to_ids)
            canon, stressed, word_of = [], [], []
            for wi, (w, k) in enumerate(zip(words, pick)):
                for p in w["variants"][k]:
                    canon.append(strip_stress(p))
                    d = re.search(r"\d", p)
                    stressed.append(None if d is None else d.group(0) != "0")
                    word_of.append(wi)
            G, best, spans = gop_features(post, self.to_ids(canon), self.alt)    # 프레임 모자라면 AlignError
            n_frames = int(post.shape[0])
            sites, _ = align(fold(canon, self.kind), fold(hyp, self.kind))
            pair = self._pair_sites(post, words, pick, canon, sites)
        mism = [0.0 if (s["actual"] is not None and same_sound(fold([canon[i]], self.kind)[0], s["actual"], stressed[i]))
                else 1.0 for i, s in enumerate(sites)]
        X = phone_table(G, canon, stressed, word_of, [w["word"] for w in words], mism)
        score = np.clip(self.reg.predict(X), 0, 2) * 50.0
        deleted = [s["actual"] is None for s in sites]
        phones = [{"phone": p, "stressed": stressed[i], "word": word_of[i], "score": float(score[i]),
                   "asr": sites[i]["actual"], "likely": self.names[int(best[i])], "mismatch": bool(mism[i]),
                   "inserted": list(sites[i]["inserted"]),
                   "span": spans[i] if spans[i][1] > spans[i][0] else None} for i, p in enumerate(canon)]
        for i, (name, alt, llr) in pair.items():
            phones[i].update(pair_cand=name, pair_alt=alt, pair_llr=round(llr, 3),
                             pair_hit=bool(llr > self.pair_margins[name]))
        self._flag(phones, self.flag_T)
        U = utt_features(score, word_of, spans, n_frames, deleted=deleted, mismatch=mism,
                         frame_sec=self.rec.post_frame_sec)
        sent = dict((k, float(np.clip(r.predict(U[None])[0], 0, 10))) for k, r in self.sent.items())
        raw = {"words": [w["word"] for w in words], "phones": phones, "sentence": sent,
               "utt_feats": [float(v) for v in U],
               "n_frames": n_frames, "frame_sec": self.rec.post_frame_sec, "vowels": VOWELS, "policy": self.policy}
        if self.ko is not None:
            raw["scores_ko"], raw["human_ko"] = self.ko.predict(raw)
            raw["scale_ko"] = self.ko.info.get("display_note")
        return raw

    def _pair_sites(self, post, words, pick, canon, sites):
        """한국인 대표 오류 자리 중 받아쓰기가 그 한국인식 소리였던 곳 → {자리: (규칙, 한국인식 소리, LLR)}."""
        if not self.pair_margins:
            return {}
        from mdd_en.pseudo import candidates, site_llrs
        ws = [{"w": w["word"], "ph": w["variants"][k]} for w, k in zip(words, pick)]
        heard = [c for c in candidates(ws) if c[3] not in (None, "+AH") and c[1] in self.pair_margins
                 and sites[c[0]]["actual"] == fold([c[3]], self.kind)[0]]
        if not heard:
            return {}
        llr = site_llrs(post, canon, heard, self.to_ids)
        if llr is None:
            return {}
        out = {}
        for (pos, name, c, alt), v in zip(heard, llr):
            if pos not in out or v > out[pos][2]:
                out[pos] = (name, alt, v)
        return out

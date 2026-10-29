# -*- coding: utf-8 -*-
"""문장 점수 — 한국인 학습자 사람 점수(AIHub 037)에 맞춘 점수기.

speechocean762(중국인 학습자) 사람 점수로 학습한 문장 점수기는 한국인 037 에서 발음(음소 점수 평균)이 96±3 에 몰려
사람을 거의 가르지 못했다(results/23 §11). 여기서는
  특징 = 엔진이 이미 계산한 값: 문장 특징 23종(`mdd_en.utterance`) + speechocean762 문장 예측 5종
         + 음소 점수 기반 발음 + 지적·한국인 대표 오류 비율
  목표 = 037 사람 점수(조음 articulationScore · 운율 prosodyScore, 1~5, 평가자 2명 평균)
  눈금 = 기본 "human": 회귀가 평균 쪽으로 오그라든 예측을 **사람 점수 분포의 평균·폭으로 되편 뒤** 5점 만점 비율(×20).
         사람이 4점 준 수준이 80 근처에 온다. (선택 "pct": 037 학습자 분포 백분위 — 한국어 API 유창·이해 방식.
         037 사람 점수는 평균 3.8/5 로 높아서 백분위로 펴면 사람이 4점 준 발화가 45 근처로 내려간다.)
         표시 눈금 "gentle"(서버 --display-scale gentle, 2026-10-14 제품 요청): human 눈금 칸(발음·운율)을 40 + 0.6×값 으로
         올려 보인다 — 사람이 조음 2·3·4·5점 준 037 발화의 평균이 71 · 80 · 87 · 93(human 눈금 51 · 66 · 79 · 89).
         순서·판별력은 그대로, 숫자만 앱처럼 너그럽다.
학습·평가: `scripts/score/en_fit_sentence_scorer_037.py` (특징 추출 `scripts/eval/en_extract_037_feats.py`).
학습 스크립트와 서버가 **같은 `sentence_features`·`to_display`** 를 쓴다.
"""

import numpy as np

from mdd_en.utterance import UTT_COLS

SO762_KEYS = ("accuracy", "fluency", "prosodic", "total", "completeness")
EXTRA_COLS = ["so_" + k for k in SO762_KEYS] + ["pron_phone", "flag_frac", "pair_hit_frac", "pair_cand_frac",
                                                 "unclear_frac", "word_err_frac"]
FEAT_COLS = UTT_COLS + EXTRA_COLS
W_OVERALL = (("pronunciation", 0.6), ("fluency", 0.2), ("rhythm", 0.2))   # 한국어 API 와 같은 비율(발음 0.6)
PCT_STEPS = np.arange(0, 101, 2)                                          # 분위수 51개 = 0, 2, …, 100


def sentence_features(raw):
    """엔진 결과(`EnglishAssessor.assess`) → 이름 붙은 특징 dict."""
    f = dict(zip(UTT_COLS, (float(v) for v in raw["utt_feats"])))
    s = raw.get("sentence") or {}
    for k in SO762_KEYS:
        f["so_" + k] = float(s.get(k, 0.0))
    ph = raw["phones"]
    n = max(1, len(ph))
    by_word, word_flag = {}, {}
    for p in ph:
        by_word.setdefault(p["word"], []).append(p["score"])
        word_flag[p["word"]] = word_flag.get(p["word"], False) or bool(p.get("flagged"))
    ws = [float(np.mean(v)) for v in by_word.values()]
    f["pron_phone"] = float(np.mean(ws)) if ws else 0.0          # 예전 응답의 pronunciation(반올림 전)
    f["flag_frac"] = sum(bool(p.get("flagged")) for p in ph) / n
    f["pair_hit_frac"] = sum(bool(p.get("pair_hit")) for p in ph) / n
    f["pair_cand_frac"] = sum(bool(p.get("pair_cand")) for p in ph) / n
    f["unclear_frac"] = sum(p.get("actual_src") == "unclear" for p in ph) / n
    f["word_err_frac"] = sum(word_flag.values()) / max(1, len(word_flag))
    return f


GENTLE = (40.0, 0.6)          # gentle 표시: 40 + 0.6 × human 눈금 값


def to_display(v, sc, gentle=False):
    """예측값 → 화면 점수(0~100). sc = 눈금 정의(학습 스크립트가 만든다).

    human: 100·(mu_h + (v − mu_p)·sd_h/sd_p)/top — 예측 분포를 사람 점수 분포의 평균·폭으로 되편다
           gentle=True 면 그 값(0~100으로 자른 뒤)을 40 + 0.6×값 으로 올려 보인다(앱 표시용)
    pct:   학습자 분포 백분위(q = 0,2,…,100 분위수 51개)
    x10:   0~10 값 ×10 (speechocean762 눈금 그대로)
    """
    if sc["kind"] == "pct":
        x = np.interp(float(v), sc["q"], PCT_STEPS)
    elif sc["kind"] == "human":
        x = 100.0 * (sc["mu_h"] + (float(v) - sc["mu_p"]) * sc["sd_h"] / sc["sd_p"]) / sc["top"]
        if gentle:
            x = GENTLE[0] + GENTLE[1] * float(np.clip(x, 0, 100))
    else:
        x = float(v) * 10.0
    return int(round(float(np.clip(x, 0, 100))))


class KoreanSentenceScorer:
    """`sent_scorer_ko.joblib` 을 읽어 문장 점수 4칸(overall · pronunciation · fluency · rhythm)을 낸다.

    파일 구조: {"regs": {칸: [회귀기, …]}, "direct": {칸: 특징 이름}, "cols": FEAT_COLS, "scale": {칸: 눈금}, "info": {...}}
      regs   — 037 사람 점수(1~5)를 예측한다(pronunciation ← 조음, rhythm ← 운율). 여러 개면 평균
      direct — 회귀 없이 특징값을 그대로 쓴다(fluency ← speechocean762 유창성 예측; 037 에 유창성 라벨이 없다)
    """

    def __init__(self, path, display="human"):
        import joblib
        obj = joblib.load(path)
        self.regs, self.direct = obj["regs"], obj.get("direct", {})
        self.cols, self.scale, self.info = obj["cols"], obj["scale"], dict(obj.get("info", {}))
        if display not in ("human", "gentle"):
            raise ValueError("display 는 human · gentle: %s" % display)
        self.gentle = display == "gentle"
        if self.gentle:
            self.info["display_note"] = ("%s · 표시 눈금 gentle(발음·운율 = 40 + 0.6×기존 눈금 — 사람 3·4점 발화 평균 80 · 87)"
                                         % (self.info.get("display_note") or "한국인 학습자(AIHub 037) 사람 점수 기준"))

    def predict(self, raw):
        """엔진 결과 → (0~100 점수 dict, 눈금 바꾸기 전 값 dict)."""
        return self.predict_feats(sentence_features(raw))

    def predict_feats(self, f):
        """`sentence_features` 결과 → (0~100 점수 dict, 눈금 바꾸기 전 값 dict). 평가 스크립트가 직접 부른다."""
        x = np.array([[float(f.get(c, 0.0)) for c in self.cols]], np.float32)
        val = dict((k, float(np.clip(np.mean([m.predict(x)[0] for m in ms]), 1, 5))) for k, ms in self.regs.items())
        val.update((k, float(f.get(c, 0.0))) for k, c in self.direct.items())
        out = dict((k, to_display(v, self.scale[k], gentle=self.gentle)) for k, v in val.items())
        parts = [(w, out[k]) for k, w in W_OVERALL if out.get(k) is not None]
        out["overall"] = int(round(sum(w * v for w, v in parts) / sum(w for w, _ in parts))) if parts else None
        return out, dict((k, round(v, 3)) for k, v in val.items())

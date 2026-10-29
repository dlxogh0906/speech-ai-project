# -*- coding: utf-8 -*-
"""세션 단위 발음 평가 — 발화 하나 진단 + 여러 발화를 모아 점수.

제품 화면이 필요로 하는 네 가지를 한곳에서 낸다.
  발화마다   실제 발음 자모/음소열 · 표준발음과의 자리별 판정 · 국적별(L1) 피드백
  세션 누적  ① 점수 3종(발음·이해가능·유창성, 1~5) ② 음소별 점수(0~100)와 약점

왜 세션인가 — 점수는 **화자 단위로만 유효**하다. 발화 1개 PCC 는 발음·이해 0.43,
유창성 0.61 이고 20~30발화를 모아야 0.67~0.77 로 포화한다(`results/03_점수_3종.json`).
음소별 점수도 자리가 몇 개 없으면 흔들려서 `MIN_POS` 미만은 내지 않는다.

점수 환산 기준(`norms.json`)은 `scripts/score/phoneme_scores.py` 가 만든다.
"""

import collections
import json
import os

import numpy as np

# 실측(화자 1,196명): 세션 점수 vs 사람 점수 상관 — 5발화 0.44 · 10발화 0.52 · 20발화 0.55 · 30발화 0.58 · 전체 0.67.
# 5발화는 순서가 자주 뒤집혀 쓸 수 없다 → 10발화부터 내고, 화면에는 30발화 이상을 권한다.
MIN_UTT_SCORE = 10
RECOMMEND_UTT = 30
# 음소 묶음은 자리 20개 이상에서만 검증했다(점수 30점 미만 묶음 실제 오류율 7.6% vs 70점 초과 3.4%).
MIN_POS = 20
SCORE_KEYS = ("pron", "comp", "flu")


def _to100(norm, x):
    return float(np.clip(np.interp(x, norm["q"], np.arange(0, 101, 2)), 0, 100))


class SessionScorer:
    """발화별 진단 결과를 받아 세션 점수를 누적한다(상태는 호출자가 보관).

    scorer_path: `train_score_models.py` 가 만든 joblib (없으면 점수 3종을 건너뛴다)
    norms_path:  `phoneme_scores.py` 가 만든 묶음별 분포
    """

    def __init__(self, scorer_path=None, norms_path=None):
        self.models = self.feat_names = None
        if scorer_path and os.path.exists(scorer_path):
            if str(scorer_path).endswith(".npz"):          # sklearn 없이 도는 이식본(mdd.hgbr_np, 예측값 동일)
                from mdd.hgbr_np import load_scorer_npz
                obj = load_scorer_npz(scorer_path)
            else:
                import joblib
                obj = joblib.load(scorer_path)
            self.models = obj.get("models", obj)
            self.feat_names = obj.get("feature_names")
        self.norms = json.load(open(norms_path, encoding="utf-8")) if (norms_path and os.path.exists(norms_path)) else {}

    # ── 발화 하나 ──
    def utt_scores(self, feat):
        """특징 41종 → 점수 3종(1~5). 모델이 없으면 None."""
        if not self.models or not self.feat_names:
            return None
        x = np.array([[float(feat.get(k, 0.0)) for k in self.feat_names]])
        out = {}
        for k in SCORE_KEYS:
            m = self.models.get(k)
            if m is not None:
                out[k] = float(np.clip(m.predict(x)[0], 1, 5))
        return out or None

    # ── 세션 누적 ──
    def session(self, utts):
        """utts: [{"scores": {...}|None, "phone": [(묶음, margin), ...]}] → 화면용 요약."""
        sc = [u["scores"] for u in utts if u.get("scores")]
        avg = {k: round(float(np.mean([s[k] for s in sc if k in s])), 2)
               for k in SCORE_KEYS if any(k in s for s in sc)} if sc else {}
        pool = collections.defaultdict(list)
        for u in utts:
            for k, v in u.get("phone") or []:
                pool[k].append(v)
        phone = {}
        for k, v in pool.items():
            if k in self.norms and len(v) >= MIN_POS:
                phone[k] = {"score": round(_to100(self.norms[k], float(np.mean(v))), 1), "n": len(v)}
        weak = sorted(phone.items(), key=lambda kv: kv[1]["score"])[:5]
        return {"n_utt": len(utts), "enough": len(sc) >= MIN_UTT_SCORE, "min_utt": MIN_UTT_SCORE,
                "recommend_utt": RECOMMEND_UTT,
                "scores": avg, "phone": phone,
                "weak": [{"key": k, "score": v["score"], "n": v["n"]} for k, v in weak],
                "note": f"점수는 화자 단위로만 유효하다 — {RECOMMEND_UTT}발화 이상 모을수록 사람 채점에 가까워진다"}


def phone_context(canon, i, vowels):
    """자리 i 의 묶음 이름(모음 / 초성 / 받침). `phoneme_scores.context` 와 같은 규칙."""
    c = canon[i]
    if c in vowels:
        return "모음 " + c
    coda = c == "ㅇ" or not (i + 1 < len(canon) and canon[i + 1] in vowels)
    return ("받침 " if coda else "초성 ") + c

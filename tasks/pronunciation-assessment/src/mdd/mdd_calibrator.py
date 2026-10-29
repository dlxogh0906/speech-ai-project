# -*- coding: utf-8 -*-
"""자리 단위 MDD 보정기 — 한 자리를 **correct / uncertain / incorrect** 로 판정한다.

왜 필요한가(2026-09-27 측정, `results/17_MDD_오류분석_2026-09-27.md`):
  · 현재 판정은 "전사가 표준발음과 다르면 오류"다 — 정밀도 0.458 / 재현율 0.533.
  · margin GOP 한 개로도 같은 수준이 나오고(P 0.444 / R 0.525), 두 신호는 서로 다른 실수를 한다.
  · 캐시된 특징으로 분류기를 학습하면 같은 지적 수에서 P 0.495 / R 0.576 (F1 0.493 → 0.532).
  · 중요한 성질: **모델이 잘못 들은 자리의 margin 이 진짜 오발음보다 더 낮다**(−1.94 vs 0.04).
    그래서 확신이 낮은 자리를 "판단 보류"로 빼면 학습자에게 억울한 지적을 크게 줄일 수 있다.

학습·임계값 산출은 `scripts/score/train_mdd_calibrator.py`. 이 모듈은 **특징 정의와 판정만** 담는다
(학습과 서비스가 같은 식을 쓰도록). 모델 파일이 없으면 `load()` 가 None 을 돌려주고, 호출측은
예전 규칙(전사 불일치 + 점수 임계값)으로 물러나면 된다.
"""

import os

STATE_CORRECT, STATE_UNCERTAIN, STATE_INCORRECT = "correct", "uncertain", "incorrect"
# uncertain 의 뜻 — **"오류일 가능성이 중간인 자리"** 다(안쪽 실제 오류율 약 33%).
# "모델이 잘 못 들었다"로 검증된 것이 아니다: margin 만으로는 학습자 오발음과 모델 인식 실수를
# 가를 수 없음이 측정됐다(17_MDD_오류분석 §10). 서비스에서는 이 자리를 **지적하지 않는 칸**으로만 쓴다.

# 발화 수준 특징 — `mdd.gop_feats_v2.utt_features` 의 키 이름 그대로여야 한다
UTT_FEATS = ["gop", "gop_p10", "gop_p25", "gop_med", "gop_min", "gop_std", "gop_bad", "neg_nll", "neg_per",
             "dur", "n_canon", "rate", "speech_rate", "artic_rate", "blank_frac", "lead_sil", "trail_sil",
             "speech_span", "n_pause", "pause_frac", "max_pause"]
SITE_FEATS = ["margin_gop", "site_gop", "phone", "position", "index", "n_canon_seq", "prev_phone", "next_phone"]
ALL_COLUMNS = SITE_FEATS + UTT_FEATS + ["mismatch"]           # site_features() 가 내는 열 순서

# 특징 묶음 — 축소 실험(ablation) 결과로 고른다. 모델은 이 중 하나만 쓴다.
#   mismatch = 전사(CTC 그리디)가 그 자리에서 표준발음과 다른가(0/1)
FEATURE_SETS = {
    "full": SITE_FEATS + UTT_FEATS,                       # 29 — 처음 학습한 구성
    "full_mism": SITE_FEATS + UTT_FEATS + ["mismatch"],   # 30
    "tiny": ["margin_gop", "mismatch", "phone", "position"],
    "minimal": ["margin_gop", "mismatch"],
}
FEATURE_NAMES = FEATURE_SETS["full"]                      # 하위 호환

VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JAMO = sorted(set(u"ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ") | VOWELS | set(u"ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ"))
PHONE_ID = dict((c, i + 1) for i, c in enumerate(_JAMO))     # 0 = 모르는 자모


def position_code(canon, i):
    """1 중성 · 2 초성 · 3 종성. ㅇ 은 종성(`mdd.pipeline` 규칙)."""
    c = canon[i]
    if c in VOWELS:
        return 1
    if c == u"ㅇ" or not (i + 1 < len(canon) and canon[i + 1] in VOWELS):
        return 3
    return 2


def site_features(canon, margins, site_gops, utt_feats, mismatch=None):
    """자리별 특징 행렬(list of list) — 열 순서는 ALL_COLUMNS. 학습기와 서비스가 이 함수만 쓴다.

    mismatch 를 안 주면 0 으로 채운다(그 열을 안 쓰는 특징 묶음이면 상관없다).
    """
    uf = [float((utt_feats or {}).get(k) or 0.0) for k in UTT_FEATS]
    rows = []
    n = len(canon)
    for i in range(n):
        m = margins[i] if (margins and i < len(margins) and margins[i] is not None) else 0.0
        g = site_gops[i] if (site_gops and i < len(site_gops) and site_gops[i] is not None) else 0.0
        mm = float(mismatch[i]) if (mismatch is not None and i < len(mismatch)) else 0.0
        rows.append([float(m), float(g), PHONE_ID.get(canon[i], 0), position_code(canon, i), i, n,
                     PHONE_ID.get(canon[i - 1], 0) if i else 0,
                     PHONE_ID.get(canon[i + 1], 0) if i + 1 < n else 0] + uf + [mm])
    return rows


def column_index(names):
    """특징 이름 목록 → ALL_COLUMNS 에서의 열 번호."""
    return [ALL_COLUMNS.index(c) for c in names]


class MddCalibrator(object):
    """joblib 번들(모델 + 임계값 + 특징 순서)을 읽어 자리별 상태를 낸다."""

    def __init__(self, bundle):
        self.model = bundle["model"]
        self.feature_set = bundle.get("feature_set", "full")
        self.names = bundle["feature_names"]
        self.thr_incorrect = float(bundle["thresholds"]["incorrect"])
        self.thr_uncertain = float(bundle["thresholds"]["uncertain"])
        self.meta = bundle.get("meta", {})
        if self.names != FEATURE_SETS.get(self.feature_set):   # 순서가 어긋나면 조용히 틀린다
            raise ValueError("특징 순서가 모듈과 다르다 — 번들을 다시 만들 것")
        self.cols = column_index(self.names)

    def probs(self, canon, margins, site_gops, utt_feats, mismatch=None):
        if not canon:
            return []
        import numpy as np
        # 학습(train_mdd_calibrator.py)과 같은 float32 — float64 로 넣으면 임계값 근처 자리가 뒤집힌다
        # (표본 9,101자리 중 7자리 불일치를 확인하고 맞췄다)
        X = np.asarray(site_features(canon, margins, site_gops, utt_feats, mismatch), dtype=np.float32)
        return [float(p) for p in self.model.predict_proba(X[:, self.cols])[:, 1]]

    def states(self, canon, margins, site_gops, utt_feats, transcription_mismatch=None):
        """자리별 (상태, 확률). `transcription_mismatch` 를 주면 **둘 다 오류라고 할 때만** incorrect.

        전사 불일치를 함께 요구하면 정밀도가 오르고(측정: 0.495 → 0.661) 재현율은 내려간다.
        서비스는 "억울한 지적"을 줄이는 쪽이 맞아 기본으로 이 인자를 넘긴다.
        """
        out = []
        for i, p in enumerate(self.probs(canon, margins, site_gops, utt_feats, transcription_mismatch)):
            mism = True if transcription_mismatch is None else bool(transcription_mismatch[i])
            if p >= self.thr_incorrect and mism:
                out.append((STATE_INCORRECT, p))
            elif p >= self.thr_uncertain or (mism and p >= self.thr_uncertain * 0.5):
                out.append((STATE_UNCERTAIN, p))
            else:
                out.append((STATE_CORRECT, p))
        return out


# 발화 단위 "다시 말해주세요" 기준 — uncertain 자리가 이 비율 이상이면 재녹음 안내.
# test 178,049발화 실측(uncertain 임계값 0.25): 비율 ≥0.3 인 발화 1.98% · ≥0.5 는 0.54%.
# 자리 하나가 보류라고 재녹음을 권하면 발화의 19.6%가 걸린다 — 너무 잦다.
RERECORD_RATIO = 0.3


def utterance_advice(states):
    """자리별 (상태, 확률) 목록 → 발화 단위 안내 하나.

    "rerecord"  : 판단 보류 자리가 많다 → "발음을 정확히 판단하기 어려워요. 다시 말해주세요."
    "feedback"  : 확정 지적이 있다     → 지적 자리만 보여준다(보류 자리는 옅게)
    "pass"      : 지적도 보류 비율도 없음
    """
    if not states:
        return "pass"
    n = float(len(states))
    n_unc = sum(1 for st, _ in states if st == STATE_UNCERTAIN)
    if n_unc / n >= RERECORD_RATIO:
        return "rerecord"
    if any(st == STATE_INCORRECT for st, _ in states):
        return "feedback"
    return "pass"


def load(path):
    """번들 경로 → MddCalibrator 또는 None(파일 없음·joblib 없음)."""
    if not path or not os.path.exists(path):
        return None
    try:
        import joblib
        return MddCalibrator(joblib.load(path))
    except Exception:
        return None

# -*- coding: utf-8 -*-
"""국적 맞춤 발음 피드백 — 진단된 오류 자리에 L1 prior 가중치와 교정 문장을 붙인다.

데이터 (둘 다 `$MDD_ANALYSIS/l1_prior_product/`, s6_l1_prior_product.py 산출물)
  l1_prior_weights.json               국가(81) × 오류유형 → prior_weight(전체 평균 대비 배수)
  nationality_feedback_enriched.json  국가(34) → summary · focus_areas[{tag, feature, problem, how_to_fix}]
                                       (원본 nationality_pronunciation_feedback.json 을 전부 포함)

유형 판정은 `mdd.pipeline.ctype` 과 같은 규칙을 쓴다(한 곳에서 관리).
교정 문장은 **그 모국어에 흔한 오류(prior_weight ≥ typical_min)** 에만 붙인다 — 모델이 정상 발음을
틀리게 들은 자리에 그럴듯한 조언이 붙는 것을 줄이기 위해서다.
삭제형(모델이 그 소리를 못 잡음)은 인식 실패일 수 있어 조언을 붙이지 않는다(DECISIONS #10 과 같은 취지).
"""

import json
import os

TYPICAL_MIN = 1.3


def _rules():
    from mdd.pipeline import ctype, DESC, PRIOR_KEY, VOWELS   # pipeline 의 유형 규칙을 그대로 쓴다
    return ctype, DESC, PRIOR_KEY, VOWELS


class L1Feedback:
    def __init__(self, weights_path, enriched_path, typical_min=TYPICAL_MIN):
        self.priors = json.load(open(weights_path, encoding="utf-8"))["priors"]
        self.text = json.load(open(enriched_path, encoding="utf-8"))["feedback"] if os.path.exists(enriched_path) else {}
        self.typical_min = typical_min
        self.ctype, self.desc, self.prior_key, self.vowels = _rules()

    def countries(self):
        rows = []
        for k, v in self.priors.items():
            t = self.text.get(k, {})
            rows.append({"key": k, "ko": t.get("nationality_ko"), "has_text": k in self.text,
                         "n_files": v.get("n_files", 0), "signal": v.get("signal_strength")})
        return sorted(rows, key=lambda r: (-int(r["has_text"]), -r["n_files"]))

    def _advice(self, country, key):
        for fa in self.text.get(country, {}).get("focus_areas", []):
            if fa.get("tag") == key:
                return {"feature": fa.get("feature"), "problem": fa.get("problem"),
                        "how_to_fix": fa.get("how_to_fix"), "residual": fa.get("residual")}
        return None

    def annotate(self, canon, sites, country=None):
        """canon: 표준발음 자모열, sites: [{i, canon, pred, verdict}] (표준발음 자리별 판정).

        반환: {country, has_text, summary, strengths, errors:[{i, target, actual, kind, type, desc,
               l1_weight, l1_typical, advice}]}
        """
        weights = self.priors.get(country or "", {}).get("weights", {})
        t = self.text.get(country or "", {})
        errors = []
        for s in sites:
            if s.get("verdict") == "일치":
                continue
            i, tgt, act = s["i"], s["canon"], s.get("pred")
            kind = "del" if act is None else "sub"
            is_coda = False
            if tgt not in self.vowels:
                is_coda = tgt == "ㅇ" or not (i + 1 < len(canon) and canon[i + 1] in self.vowels)
            et = self.ctype(tgt, act or "", kind, is_coda)
            pk = self.prior_key.get(et)
            w = weights.get(pk, {}).get("prior_weight") if pk else None
            typical = w is not None and w >= self.typical_min
            e = {"i": i, "target": tgt, "actual": act or "∅", "kind": kind, "type": et,
                 "desc": self.desc.get(et, ""), "l1_weight": round(w, 2) if w is not None else None,
                 "l1_typical": typical, "advice": None}
            if typical and kind == "sub":
                e["advice"] = self._advice(country, pk)
            errors.append(e)
        return {"country": country, "has_text": bool(t), "nationality_ko": t.get("nationality_ko"),
                "summary": t.get("summary"), "strengths": t.get("strengths"),
                "typical_min": self.typical_min, "errors": errors}

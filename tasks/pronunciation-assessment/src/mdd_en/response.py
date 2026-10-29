# -*- coding: utf-8 -*-
"""엔진 결과 → 응답 JSON. **한국어 API(`sent.eval.kr` 호환)와 같은 모양**이라 앱이 그대로 읽는다.

한국어 응답과 같은 칸(이름·뜻 동일)
    result.{overall, pronunciation, fluency, rhythm, comprehensibility, words[]}
    words[].{word, readType(0 정상·1 누락·2 대치·3 오독·4 삽입), score, phonemes[], inserted[]}
    phonemes[].{alpha, pronunciation, position, actual, errorType, uncertain}
    0 점은 null (한국어 API 2026-09-26 규칙과 같다)

영어에서 뜻이 달라지는 칸
    alpha · actual       ARPAbet 소문자(강세 숫자 없음) — 한국어의 KoG2P 기호 자리. 한글 대신 `alpha_ipa`·`actual_ipa`
    position             "모음" · "자음" (한국어: 초성·중성·종성)
    errorType            "모음오류" · "자음오류" · "음운탈락"(빠뜨림). 한국인 대표 오류면 errorRule(f_p …)·errorHint("f → p")
                         policy v2 에서는 모음 삽입(words[].inserted)을 내보내지 않는다(정밀도가 낮다)
    comprehensibility    영어에는 없다 → null

SpeechSuper 영어(sent.eval.promax)에서 쓰는 이름도 같이 싣는다(모르는 키는 앱이 무시한다)
    result.{integrity, speed, pause_count} · words[].{scores{overall, pronunciation}, span}
    phonemes[].{phoneme, sound_like('-' = 빠뜨림), span, stress_mark}
    ⚠️ readType 은 **우리 코드(0~4)** 다 — SpeechSuper 영어의 0/3/4·0/1/2/3/6 과 다르다.
"""

from mdd_en.phones import VOWELS, to_ipa

READ_NORMAL, READ_MISS, READ_SUB, READ_BAD, READ_INS = 0, 1, 2, 3, 4
BAD_SCORE = 20


def nz(v):
    return None if v is None or v == 0 else v


def _span(span, frame_sec, offset_sec=0.0):
    """정렬 구간(프레임) → SpeechSuper 와 같은 10 ms 단위. offset_sec 로 **보낸 녹음 기준** 시각으로 옮긴다
    (서버가 앞뒤 무음을 잘라 채점하므로)."""
    if not span:
        return None
    k = frame_sec * 100.0
    o = offset_sec * 100.0
    return {"start": max(0, int(round(span[0] * k + o))), "end": max(0, int(round(span[1] * k + o)))}


def build_response(raw, ref_text, record_id=None, flag_rule=None, offset_sec=0.0):
    fs = raw["frame_sec"]
    v2 = raw.get("policy") == "v2"
    words_out = []
    by_word = {}
    for p in raw["phones"]:
        by_word.setdefault(p["word"], []).append(p)
    for wi, word in enumerate(raw["words"]):
        ph_out, ins = [], []
        for p in by_word.get(wi, []):
            sc = int(round(p["score"]))
            is_v = p["phone"] in VOWELS
            q = {"alpha": p["phone"].lower(), "alpha_ipa": to_ipa(p["phone"], p["stressed"]),
                 "position": "모음" if is_v else "자음", "pronunciation": sc,
                 "actual": p["actual"].lower() if p["actual"] else None,
                 "actual_ipa": to_ipa(p["actual"], p["stressed"]) if p["actual"] else None,
                 "phoneme": p["phone"].lower(), "sound_like": p["actual"].lower() if p["actual"] else "-",
                 "span": _span(p["span"], fs, offset_sec)}
            if is_v:
                q["stress_mark"] = 1 if p["stressed"] else 0
            if p["flagged"]:
                if p["actual"] is None:
                    q["errorType"] = "음운탈락"
                elif p["actual"] != p["phone"]:
                    q["errorType"] = "모음오류" if is_v else "자음오류"
                    if p.get("pair_cand") and p["actual"] == p.get("pair_alt"):   # 한국인 대표 오류(f→p 등)
                        q["errorRule"] = p["pair_cand"]
                        q["errorHint"] = "%s → %s" % (q["alpha_ipa"], q["actual_ipa"])
                        q["actual_src"] = "pair" if p.get("actual_src") == "pair" else "asr"
                elif v2 and p.get("actual_src") == "unclear":
                    q["errorType"] = "모음오류" if is_v else "자음오류"   # 점수가 낮다 — 무엇으로 냈는지는 모른다
                    q["actual_src"] = "unclear"
                    q["sound_like"] = "?"
                else:
                    q["uncertain"] = True                 # (v1) 점수는 낮지만 무엇으로 냈는지 못 잡았다
                if p["actual_src"] == "gop":
                    q["actual_src"] = "likely"            # (v1) 받아쓰기가 아니라 "가장 그럴듯한 소리" 추정
            elif p["actual"] != p["phone"] and (v2 or p["mismatch"]):
                q["uncertain"] = True                     # 들린 소리는 달랐지만 근거가 문턱 아래 — 지적하지 않는다
                if p.get("pair_cand"):
                    q["errorRule"] = p["pair_cand"]
            ph_out.append(q)
            if not v2:                                    # v2: 모음 삽입은 정밀도가 낮아(약 0.27) 보여 주지 않는다
                ins += [{"actual": t.lower(), "actual_ipa": to_ipa(t)} for t in p["inserted"]]
        got = [q["pronunciation"] for q in ph_out]
        score = int(round(sum(got) / len(got))) if got else None
        n_err = sum(1 for q in ph_out if q.get("errorType"))
        n_del = sum(1 for q in ph_out if q.get("errorType") == "음운탈락")
        if ph_out and n_del == len(ph_out):
            rt = READ_MISS
        elif score is not None and score < BAD_SCORE and n_err:
            rt = READ_BAD
        elif n_err:
            rt = READ_SUB
        elif ins:
            rt = READ_INS
        else:
            rt = READ_NORMAL
        spans = [q["span"] for q in ph_out if q["span"]]
        # 오독(readType 3)은 한국어 API 와 같이 0 점(→ null)으로 두고, 문장 발음 평균에도 0 으로 넣는다
        row = {"word": word, "readType": rt, "score": 0 if rt == READ_BAD else score, "phonemes": ph_out,
               "scores": {"overall": score, "pronunciation": score},
               "span": {"start": spans[0]["start"], "end": spans[-1]["end"]} if spans else None}
        if ins:
            row["inserted"] = ins
        words_out.append(row)

    got = [w["score"] for w in words_out if w["score"] is not None]
    pron = int(round(sum(got) / len(got))) if got else None
    s = raw["sentence"] or {}
    to100 = lambda k: int(round(s[k] * 10)) if k in s else None
    ko = raw.get("scores_ko")
    if ko:                      # 한국인 037 사람 점수 기준(mdd_en.sent_score) — 세 칸 모두 학습자 분포 백분위
        top = {"overall": ko.get("overall"), "pronunciation": ko.get("pronunciation"),
               "fluency": ko.get("fluency"), "rhythm": ko.get("rhythm")}
    else:                       # 예전: speechocean762 사람 점수(0~10) ×10, 발음 = 단어 점수 평균
        top = {"overall": to100("total"), "pronunciation": pron, "fluency": to100("fluency"),
               "rhythm": to100("prosodic")}
    spoken = [w for w in words_out if w["readType"] != READ_MISS]
    integrity = int(round(100.0 * len(spoken) / len(words_out))) if words_out else None
    sp = [p["span"] for p in raw["phones"] if p["span"]]
    dur = (max(b for _, b in sp) - min(a for a, _ in sp)) * fs if sp else 0.0
    gaps = []
    for w1, w2 in zip(words_out, words_out[1:]):
        if w1["span"] and w2["span"]:
            gaps.append((w2["span"]["start"] - w1["span"]["end"]) / 100.0)
    for w in words_out:                                   # 평균을 구한 뒤 0 → null (한국어 API 와 같은 규칙)
        w["score"] = nz(w["score"])
        w["scores"] = {"overall": w["score"], "pronunciation": w["score"]}
        for q in w["phonemes"]:
            q["pronunciation"] = nz(q["pronunciation"])
    return {
        "refText": ref_text, "recordId": record_id,
        "result": {
            "overall": nz(top["overall"]), "pronunciation": nz(top["pronunciation"]), "fluency": nz(top["fluency"]),
            "rhythm": nz(top["rhythm"]), "comprehensibility": None,
            "integrity": nz(integrity), "speed": int(round(len(spoken) / dur * 60)) if dur > 0.3 else None,
            "pause_count": int(sum(1 for g in gaps if g >= 0.3)),
            "words": words_out,
        },
        "_meta": {
            "engine": "beavertalk-pron-en",
            "scale": ("overall = 0.6·발음 + 0.2·유창 + 0.2·운율. %s. 단어·음소 점수는 음소 점수 기반이라 눈금이 다르다"
                      % (raw.get("scale_ko") or "한국인 학습자(AIHub 037) 사람 점수 기준")
                      if ko else
                      "overall·fluency·rhythm = 사람 점수(0~10, speechocean762) 예측 ×10. "
                      "pronunciation = 단어 점수 평균, 단어 점수 = 음소 점수 평균"),
            "sentence_scorer": "ko037" if ko else "so762",
            "pronunciation_phone": pron,          # 단어 점수 평균(예전 pronunciation) — 참고용
            "human_ko": raw.get("human_ko"),      # 백분위로 펴기 전: 조음·운율 1~5 예측, 유창 0~10
            "phoneme_score": "사람 음소 점수(0 틀림 · 1 억양 강함 · 2 맞음) 예측 ×50 — 0~100",
            "flag_rule": (("음소 점수 < %.1f(speechocean762 에서 맞게 발음한 자리의 %s 가 걸리는 문턱)" % (
                               flag_rule["T"], flag_rule.get("frr_target", "?")) if flag_rule else "") +
                          (" 또는 한국인 대표 오류 자리에서 들린 소리가 그 한국인식 소리이고 근거(LLR)가 규칙별 문턱을 넘을 때"
                           " errorType 을 붙인다(errorRule·errorHint). ð→d 는 판정하지 않는다" if v2 else
                           " 이면 errorType 을 붙인다")),
            "actual_src": ("asr = 받아쓰기(들린 소리) · pair = 한국인 대표 오류로 판정한 소리 · unclear = 점수는 낮지만"
                           " 무엇으로 냈는지 모름(actual 은 표준 기호, 화면에는 '부정확'으로만)" if v2 else
                           "actual 은 받아쓰기(들린 소리). actual_src=likely 는 받아쓰기는 표준대로였지만 점수가 낮아"
                           " 가장 그럴듯한 대체 소리를 추정해 넣은 것"),
            "policy": raw.get("policy", "v1"),
            "rhythm": "운율(prosody) 예측 — SpeechSuper 의 rhythm 자리",
            "comprehensibility": "영어에는 없다(null)",
            "integrity": "읽은 단어 비율(모든 소리를 빠뜨린 단어 = 안 읽음)",
            "speed": "분당 단어 수(말한 구간 기준)",
            "pause_count": "단어 사이 0.3초 이상 쉼 개수",
            "raw_human_0to10": dict((k, round(v, 2)) for k, v in s.items()),
            "readType": "우리 코드(0 정상·1 누락·2 대치·3 오독·4 삽입) — SpeechSuper 영어 코드와 다르다",
            "zero_as_null": "0 점은 null 로 보낸다(한국어 API 와 같은 규칙)",
        },
    }

# -*- coding: utf-8 -*-
"""speechocean762 읽기 — 영어 발음 점수(음소·단어·문장)를 사람 5명이 매긴 공개 코퍼스(CC BY 4.0).

화자는 전원 **중국어 모국어**(어린이 절반)라 한국인 학습자와 오류 양상이 다르다 — 점수 눈금 맞추기와
사람 점수와의 상관(PCC) 비교용이다. 한국인 오류 판정은 따로 확인한다.

    from mdd_en.speechocean import load_split
    rows = load_split(root, "train")     # root = $SPEECH_AI_DATA/english/speechocean762

음소 점수(phones-accuracy)는 5명 평균 0~2: 2 맞음 · 1 억양이 강함 · 0 틀림/빠뜨림.
"""

import json
import os
import re


def base_phone(p):
    """'AO0' → 'AO'. speechocean762 에는 강세 숫자가 없는 모음(IH·UH)도 섞여 있다."""
    return re.sub(r"\d", "", p)


def load_split(root, split):
    """→ [{utt, spk, age, wav, text, accuracy, completeness, fluency, prosodic, total, words}].

    words[] = {text, phones(강세 포함), phones_acc(0~2), accuracy, stress, total}
    """
    scores = json.load(open(os.path.join(root, "resource", "scores.json"), encoding="utf-8"))
    sd = os.path.join(root, split)

    def kv(name):
        p = os.path.join(sd, name)
        if not os.path.exists(p):
            return {}
        return dict(l.rstrip("\n").split(None, 1) for l in open(p, encoding="utf-8") if l.strip())

    wav, u2s, age = kv("wav.scp"), kv("utt2spk"), kv("spk2age")
    out = []
    for utt in sorted(wav):
        r = scores[utt]
        words = [{"text": w["text"], "phones": list(w["phones"]),
                  "phones_acc": [float(a) for a in w["phones-accuracy"]],
                  "accuracy": float(w["accuracy"]), "stress": float(w["stress"]), "total": float(w["total"])}
                 for w in r["words"]]
        spk = u2s.get(utt, "")
        out.append({"utt": utt, "spk": spk, "age": int(age[spk]) if spk in age else None,
                    "wav": os.path.join(root, wav[utt]), "text": r["text"],
                    "accuracy": float(r["accuracy"]), "completeness": float(r["completeness"]),
                    "fluency": float(r["fluency"]), "prosodic": float(r["prosodic"]),
                    "total": float(r["total"]), "words": words})
    return out

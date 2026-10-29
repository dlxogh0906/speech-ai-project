# -*- coding: utf-8 -*-
"""L2-ARCTIC v5.0 읽기 — 사람이 **실제로 들린 음소**를 단 영어 L2 코퍼스(한국인 화자 HJK·HKK·YDCK·YKWK).

⚠️ 라이선스 CC BY-NC 4.0 — **사내 평가 전용**. 학습·점수 보정에 쓰지 않고, 데이터를 밖으로 내보내지 않는다.
상업 이용은 TAMU 허가가 필요하다.

annotation/*.TextGrid 의 phones 층:
    "AH0"          표준대로 발음
    "DH,D,s"       대치 — 표준 DH 를 D 로
    "T,sil,d"      삭제(들린 소리 없음)
    "sil,AH,a"     삽입 — 표준에 없는 AH 가 들어감
들린 소리 표기에 `*`(왜곡)·`err` 가 붙기도 한다 → 표준과 다른 소리로 센다.
"""

import os
import re

KOREAN = ("HJK", "HKK", "YDCK", "YKWK")
SIL = {"", "sil", "sp", "spn", "SIL", "SP"}


def _norm(p):
    """강세 숫자·꼬리 '_' 를 떼고, AX(약모음) 는 AH 로. 'R*'(왜곡)·'ERR' 는 그대로 둬서 표준과 다른 소리로 센다."""
    p = re.sub(r"\d", "", (p or "").strip()).upper().rstrip("_")
    if p == "AX":
        return "AH"
    return None if p in {"SIL", "SP", "SPN", ""} else p


def _stressed(p):
    """모음의 강세 표시(1·2 → True, 0 → False). 자음·표시 없음 → None."""
    m = re.search(r"(\d)", p or "")
    return None if m is None else m.group(1) != "0"


def read_textgrid_tier(path, tier):
    """간단한 TextGrid 파서 → [(xmin, xmax, text)]."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    out, cur, i = [], None, 0
    while i < len(lines):
        l = lines[i].strip()
        if l.startswith("name ="):
            cur = l.split("=", 1)[1].strip().strip('"')
        elif cur == tier and l.startswith("xmin =") and i + 2 < len(lines) and "text =" in lines[i + 2]:
            xmin = float(l.split("=")[1])
            xmax = float(lines[i + 1].split("=")[1])
            text = lines[i + 2].split("=", 1)[1].strip().strip('"').strip()
            out.append((xmin, xmax, text))
            i += 3
            continue
        i += 1
    return out


def parse_phones(path):
    """→ ([{"canon": 표준(강세 뗌) 또는 None(삽입), "actual": 들린 소리 또는 None(삭제), "type": ok/s/d/a,
           "t": (xmin, xmax), "stressed", "w": 단어 번호}], [단어 …])"""
    words = [(a, b, t) for a, b, t in read_textgrid_tier(path, "words") if t not in SIL]
    out = []
    for xmin, xmax, text in read_textgrid_tier(path, "phones"):
        if text in SIL:
            continue
        if "," in text:
            parts = [x.strip() for x in text.split(",")]
            canon, actual, typ = (parts + ["", "", ""])[:3]
            typ = typ.lower()[:1] or "s"
            c, a = _norm(canon), _norm(actual)
            if typ == "a":
                c = None
            if typ == "d":
                a = None
            if c is None and a is None:
                continue
            out.append({"canon": c, "actual": a, "type": typ, "t": (xmin, xmax), "stressed": _stressed(canon)})
        else:
            c = _norm(text)
            if c:
                out.append({"canon": c, "actual": c, "type": "ok", "t": (xmin, xmax), "stressed": _stressed(text)})
    for p in out:                                         # 음소 가운데 시각이 들어가는 단어
        mid = sum(p["t"]) / 2
        p["w"] = next((k for k, (a, b, _) in enumerate(words) if a <= mid < b), None)
    return out, [t for _, _, t in words]


def load_speakers(root, speakers=KOREAN):
    """→ [{utt, spk, wav, text, phones, words}] — 사람 주석(annotation)이 있는 발화만."""
    rows = []
    for spk in speakers:
        ad = os.path.join(root, spk, "annotation")
        if not os.path.isdir(ad):
            continue
        for f in sorted(os.listdir(ad)):
            if not f.endswith(".TextGrid"):
                continue
            stem = f[:-9]
            tp = os.path.join(root, spk, "transcript", stem + ".txt")
            rows.append({"utt": "%s_%s" % (spk, stem), "spk": spk,
                         "wav": os.path.join(root, spk, "wav", stem + ".wav"),
                         "text": open(tp, encoding="utf-8").read().strip() if os.path.exists(tp) else "",
                         "phones": None, "words": None})
            rows[-1]["phones"], rows[-1]["words"] = parse_phones(os.path.join(ad, f))
    return rows


def annotation_words(r):
    """주석 음소 → 단어 목록 [{w, ph(ARPAbet+강세 숫자), act(사람이 들은 소리)}] — 표준 자리가 없는(삽입) 음소는 뺀다.
    강세: True → 1, False → 0, 자음 → 숫자 없음. `mdd_en.pseudo.candidates` 에 바로 넣을 수 있다."""
    import collections
    ws = collections.OrderedDict()
    for p in r["phones"]:
        if p["canon"] is None:
            continue
        st = p["stressed"]
        ws.setdefault(p["w"], []).append((p["canon"] + ("" if st is None else ("1" if st else "0")), p["actual"]))
    return [{"w": str(k), "ph": [x for x, _ in v], "act": [y for _, y in v]} for k, v in ws.items()]

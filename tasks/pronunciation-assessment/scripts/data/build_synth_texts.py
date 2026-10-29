# -*- coding: utf-8 -*-
"""한국인 오류 합성용 문장 목록 — 문장마다 단어별 표준발음(ARPAbet + 강세, `mdd_en.g2p`).

문장 출처(학습에 써도 되는 것만):
    AIHub 037 PRN 문장(학습 라벨의 sentence) · speechocean762 문장(CC BY 4.0)
L2-ARCTIC(CMU ARCTIC 문장)은 평가 전용이라 넣지 않는다 — 겹치는 문장은 빼고 개수를 남긴다.

산출: $MDD_OUTPUTS/english/synth_ko/texts.jsonl   한 줄 = {sid, src, text, words: [{w, ph}]}

    NLTK_DATA=... PYTHONPATH=<영어 덧씌우기> $PY tasks/pronunciation-assessment/scripts/data/build_synth_texts.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, glob, json, os, re

D037 = os.path.join("aihub", "037.교육용_한국인의_영어_음성_데이터", "01-1.정식개방데이터")


def key(t):
    return re.sub(r"[^a-z ]", "", t.lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root037", default=os.path.join(str(DATA_ROOT), D037))
    ap.add_argument("--so762", default=os.path.join(str(DATA_ROOT), "english", "speechocean762"))
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "english", "synth_ko", "texts.jsonl"))
    ap.add_argument("--max-words", dest="max_words", type=int, default=30)
    a = ap.parse_args()
    from mdd_en.g2p import EnglishG2P
    g = EnglishG2P()

    texts = []
    for f in sorted(glob.glob(os.path.join(a.root037, "Training", "02.라벨링데이터", "TL_PRN_en_NA_NA", "**", "*.json"),
                              recursive=True)):
        for c in json.load(open(f, encoding="utf-8-sig"))["itemScript"]["contents"]:
            texts.append(("aihub037", c["sentence"].strip()))
    so = json.load(open(os.path.join(a.so762, "resource", "scores.json"), encoding="utf-8"))
    texts += [("so762", r["text"].capitalize()) for r in so.values()]
    arctic = set()
    for p in glob.glob(os.path.join(a.l2arctic, "*", "transcript", "*.txt")):
        arctic.add(key(open(p, encoding="utf-8").read()))

    seen, rows, n_arctic = set(), [], 0
    for src, t in texts:
        k = key(t)
        if not k or k in seen:
            continue
        seen.add(k)
        if k in arctic:
            n_arctic += 1
            continue
        words = [{"w": w["word"], "ph": w["variants"][0]} for w in g.words(t)]
        if not words or len(words) > a.max_words:
            continue
        rows.append({"sid": "s%05d" % len(rows), "src": src, "text": t, "words": words})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[r["src"]] = by.get(r["src"], 0) + 1
    print("[texts] %d문장 %s · L2-ARCTIC 과 겹쳐 뺀 문장 %d → %s" % (len(rows), by, n_arctic, a.out))


if __name__ == "__main__":
    main()

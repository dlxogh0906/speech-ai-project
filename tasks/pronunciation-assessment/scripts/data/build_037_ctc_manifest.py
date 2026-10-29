# -*- coding: utf-8 -*-
"""AIHub 037(교육용 한국인의 영어 음성) PRN → 영어 음소 CTC 학습 목록. 정답 = 라벨의 **phonemeText(들린 소리)**.

⚠️ phonemeText 는 AIHub 가 **자동 인식기로 만든** 라벨이다(사람 전사 아님). 사람 조음 점수 5.0 발화에서도
   음소의 16% 를 "틀림"으로 적는다 — 학습해 보고 사람 라벨(L2-ARCTIC 한국인)로 효과를 재는 것이 목적이다.

- IPA → ARPAbet 39 (`mdd_en.phones.aihub037_to_arpa`: 쪼개 적은 이중모음 a ɪ → AY, ə r → ER …)
- 문단 항목(문장 5~7개)은 문장별 라벨을 이어 붙인다. MAX_SEC 초 넘는 녹음·0.5초 미만은 뺀다
- 037 공식 검증 세트는 학습과 **화자가 같다**(882명 전원) → train+val 을 합쳐 화자 단위로 dev 를 새로 뗀다

산출: $MDD_OUTPUTS/english/aihub037_ctc/{train,dev}.jsonl · stats.json
    한 줄 = {id, wav, sec, text, phones(들린 소리), canon(037 표준발음, 못 바꾸면 null), spk, age, ability,
             art, pros, tag_err, src}

    $PY tasks/pronunciation-assessment/scripts/data/build_037_ctc_manifest.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, json, os, random

D037 = os.path.join("aihub", "037.교육용_한국인의_영어_음성_데이터", "01-1.정식개방데이터")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(str(DATA_ROOT), D037))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "english", "aihub037_ctc"))
    ap.add_argument("--dev-speakers", dest="dev_spk", type=int, default=40)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from mdd_en.phones import aihub037_to_arpa

    rows, drop = [], collections.Counter()
    for src, lab, wav in (("train", "Training/02.라벨링데이터/TL_PRN_en_NA_NA", "Training/01.원천데이터/TS_PRN_en_NA_NA"),
                          ("val", "Validation/02.라벨링데이터/VL_PRN_en_NA_NA", "Validation/01.원천데이터/VS_PRN_en_NA_NA")):
        for f in sorted(glob.glob(os.path.join(a.root, lab, "**", "*.json"), recursive=True)):
            j = json.load(open(f, encoding="utf-8-sig"))
            it = j["itemScript"]
            sec = float(j.get("recrdTime") or 0)
            w = os.path.join(a.root, wav, j["fileNm"] + ".wav")
            if not os.path.exists(w):
                drop["no_wav"] += 1; continue
            if sec < 0.5 or sec > a.max_sec:
                drop["length"] += 1; continue
            phones, canon, tags, text = [], [], [], []
            for c in it["contents"]:
                text.append(c["sentence"].strip())
                p = aihub037_to_arpa(c["phonemeText"].split())
                q = aihub037_to_arpa(c["transAnsw"].split())
                if p is None:
                    phones = None; break
                phones += p
                canon = None if (canon is None or q is None) else canon + q
                tags += c["tagging"].split()
            if not phones:
                drop["bad_label"] += 1; continue
            r = it["rating"]
            rows.append({"id": j["fileNm"], "wav": w, "sec": round(sec, 3), "text": " ".join(text),
                         "phones": " ".join(phones),
                         "canon": " ".join(canon) if canon else None, "spk": j["recorder"]["recorderId"],
                         "age": j["recorder"]["age"], "ability": j["recorder"]["ability"],
                         "art": r["articulationScore"], "pros": r["prosodyScore"],
                         "tag_err": round(sum(t != "C" for t in tags) / max(len(tags), 1), 4), "src": src})
    spks = sorted(set(r["spk"] for r in rows))
    random.Random(a.seed).shuffle(spks)
    dev = set(spks[: a.dev_spk])
    os.makedirs(a.out, exist_ok=True)
    n = collections.Counter()
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as ft, \
            open(os.path.join(a.out, "dev.jsonl"), "w", encoding="utf-8") as fd:
        for r in rows:
            sp = "dev" if r["spk"] in dev else "train"
            (fd if sp == "dev" else ft).write(json.dumps(r, ensure_ascii=False) + "\n")
            n[sp] += 1; n[sp + "_sec"] += r["sec"]
    stats = {"train": n["train"], "dev": n["dev"], "train_hours": round(n["train_sec"] / 3600, 1),
             "dev_hours": round(n["dev_sec"] / 3600, 1), "speakers": len(spks), "dev_speakers": len(dev),
             "dropped": dict(drop), "max_sec": a.max_sec,
             "label_source": "phonemeText (AIHub 자동 인식 라벨, 사람 전사 아님)"}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

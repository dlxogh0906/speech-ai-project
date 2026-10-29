# -*- coding: utf-8 -*-
"""AIHub 037 SPK(말하기 평가) → 영어 문장 점수의 **잠근 외부 검증셋** 목록.

SPK 는 질문에 자유롭게 답한 녹음이고, 사람 점수는 말하기 전반 1개(itemScript.rating, 평가자 3명 평균, 약 1~9)다.
낭독 발음 점수(PRN 의 조음·운율)와 같은 잣대는 아니므로 **학습에는 쓰지 않고**, 점수가 실력 순서를 따라가는지만 본다.
refText 는 그 녹음의 전사(contents[].sentence) — 말한 내용을 기준으로 발음을 잰다.

거르는 것: PRN 목록(aihub037_ctc)에 있는 화자 · 0.5초 미만 / 30초 초과(서버 상한) · 전사 없음.
쓰는 칸: fileNm · recrdTime · recorder.recorderId(화자 단위 집계용) · itemScript.{rating, testType, contents}.
나이·수준 등 다른 화자 정보는 읽지 않는다.

    $PY .../build_037_spk_eval.py            # → $MDD_OUTPUTS/english/aihub037_spk/test.jsonl
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, json, os

D037 = os.path.join("aihub", "037.교육용_한국인의_영어_음성_데이터", "01-1.정식개방데이터")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(str(DATA_ROOT), D037))
    ap.add_argument("--prn", default=os.path.join(str(MDD_OUTPUTS), "english", "aihub037_ctc"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "english", "aihub037_spk"))
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=30.0)
    a = ap.parse_args()

    prn_spk = set()
    for sp in ("train", "dev"):
        with open(os.path.join(a.prn, sp + ".jsonl"), encoding="utf-8") as f:
            prn_spk.update(json.loads(l)["spk"] for l in f)
    rows, drop = [], collections.Counter()
    for split, lab_pre, wav_pre in (("Training", "TL_SPK", "TS_SPK"), ("Validation", "VL_SPK", "VS_SPK")):
        for lab_dir in sorted(glob.glob(os.path.join(a.root, split, "02.라벨링데이터", lab_pre + "_*"))):
            wav_dir = os.path.join(a.root, split, "01.원천데이터", wav_pre + os.path.basename(lab_dir)[len(lab_pre):])
            for f in sorted(glob.glob(os.path.join(lab_dir, "**", "*.json"), recursive=True)):
                j = json.load(open(f, encoding="utf-8-sig"))
                spk = j["recorder"]["recorderId"]
                if spk in prn_spk:
                    drop["prn_speaker"] += 1; continue
                it = j["itemScript"]
                sec = float(j.get("recrdTime") or 0)
                if sec < 0.5 or sec > a.max_sec:
                    drop["length"] += 1; continue
                text = " ".join(c["sentence"].strip() for c in (it.get("contents") or []) if c.get("sentence"))
                if not text.strip():
                    drop["no_text"] += 1; continue
                w = os.path.join(wav_dir, j["fileNm"] + ".wav")
                if not os.path.exists(w):
                    drop["no_wav"] += 1; continue
                if it.get("rating") is None:
                    drop["no_rating"] += 1; continue
                rows.append({"id": j["fileNm"], "wav": w, "sec": round(sec, 3), "text": text, "spk": spk,
                             "rating": float(it["rating"]), "task": it.get("testType"), "src": split})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "test.jsonl"), "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"utts": len(rows), "speakers": len(set(r["spk"] for r in rows)), "dropped": dict(drop),
             "tasks": dict(collections.Counter(r["task"] for r in rows)), "max_sec": a.max_sec,
             "note": "SPK 말하기 평가 · PRN 화자 제외 · 사람 점수 = 말하기 전반 1개(평가자 3명 평균). 학습에 쓰지 않는다"}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

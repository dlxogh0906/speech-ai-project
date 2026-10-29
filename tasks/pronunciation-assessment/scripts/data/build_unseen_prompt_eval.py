# -*- coding: utf-8 -*-
"""**학습에 없던 문항**으로 MDD 평가 split 을 만든다 — 문장 일반화를 재려면 이것뿐이다.

배경: Stage 2(`manifest_full_tag_replay10`)는 AIHub 040~043 의 문항 226개를 **전부** 학습에 썼다.
그래서 지금까지 「안 본 문항」이라 부른 split(`manifest_foreign_clean_v2/test_unseen`)도 실제로는
학습에 있던 문항이다(`scripts/eval/split_leakage_check.py` 로 확인). 네 데이터셋의 문항 코드를
전부 모으면 229개이고, 그중 **RP002·RP003·RP004(문단 읽기) 3개가 학습에 없다.**
여기에 학습 화자까지 빼면 **화자·문장 모두 처음인** 평가셋이 된다.

주의 — 같은 조건 비교가 아니다. RP 는 문단이라 발화가 길고(다른 split 은 문장·단어), 난이도와
자모 분포가 다르다. 보고서에는 「문항+화자 모두 미관측」이라는 조건과 이 차이를 함께 적을 것.

라벨: TAG phones 티어 = 실제 발음(CTC 정답·MDD 기준) · LAB phonemic.phones = 표준음(canon).
     `build_tag_manifest.py` 와 같은 파싱이다.

    $PY .../build_unseen_prompt_eval.py --out $MDD_OUTPUTS/manifest_unseen_prompt_rp \\
        --exclude-speakers-from $MDD_OUTPUTS/manifest_full_tag_replay10:train
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, glob, io, json, os, re, collections

AIHUB = os.path.join(str(DATA_ROOT), "aihub")
WAV = os.path.join(str(DATA_ROOT), "mdd_gold_wav16k")
ITEM = re.compile(r"-([A-Z]{2}\d{3})-")


def is_jamo(c):
    return 0x3131 <= ord(c) <= 0x3163


def parse_tag(path):
    """TAG TextGrid 의 phones 티어 → 실제 발음 자모열 (build_tag_manifest.py 와 같은 규칙)."""
    txt = None
    for enc in ("utf-16", "utf-8"):
        try:
            txt = io.open(path, encoding=enc).read()
            break
        except Exception:
            continue
    if txt is None:
        return None, None
    m = re.search(r'name = "phones".*?(?=name = "|\Z)', txt, re.S)
    if not m:
        return None, None
    toks = []
    for lbl in re.findall(r'text = "([^"]*)"', m.group(0)):
        lbl = lbl.strip()
        if lbl in ("", ".", "-", "spn", "<unk>"):
            continue
        toks += [ch for ch in lbl if is_jamo(ch)]
    w = re.search(r'name = "words".*?(?=name = "|\Z)', txt, re.S)
    words = [t.strip() for t in re.findall(r'text = "([^"]*)"', w.group(0))] if w else []
    return toks, u" ".join([t for t in words if t])


def parse_lab(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    ph = (d.get("RecordingMetadata") or {}).get("phonemic") or {}
    if "phones" not in ph:
        return None
    toks = []
    for x in ph["phones"]:
        toks += [ch for ch in str(x.get("phone", "")).strip() if is_jamo(ch)]
    return toks


def wavpath(stem):
    return os.path.join(WAV, stem[:2], stem + ".wav")


def frames_of(p):
    return max(0, (os.path.getsize(p) - 44) // 2)          # 16k mono int16


def load_speakers(spec):
    mdir, split = spec.rsplit(":", 1)
    out = set()
    with open(os.path.join(mdir, "%s.tsv" % split), encoding="utf-8") as f:
        next(f)
        for line in f:
            out.add(os.path.basename(line.split("\t")[0])[:5])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", default=["RP002", "RP003", "RP004"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude-speakers-from", dest="excl", nargs="*", default=[],
                    help="<매니페스트>:<split> — 이 split 의 화자는 제외(학습 화자 제거)")
    ap.add_argument("--dict", dest="dict_path", default=None, help="복사해 둘 dict.ltr.txt (기본: manifest_full_tag_replay10)")
    ap.add_argument("--min-sec", dest="min_sec", type=float, default=0.5)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=20.0)
    a = ap.parse_args()

    excl = set()
    for spec in a.excl:
        excl |= load_speakers(spec)
    print("[제외] 학습 화자 %d명" % len(excl))

    want = set(a.items)
    tags, labs = {}, {}
    for root in glob.glob(os.path.join(AIHUB, "0[0-9][0-9]*", "01-1.정식개방데이터", "*", "02.라벨링데이터", "*_pronunciation_tag")):
        for p in glob.glob(os.path.join(root, "*.TextGrid")):
            b = os.path.basename(p)[:-9]
            m = ITEM.search(b)
            if m and m.group(1) in want:
                tags[b] = p
    for root in glob.glob(os.path.join(AIHUB, "0[0-9][0-9]*", "01-1.정식개방데이터", "*", "02.라벨링데이터", "*_pronunciation_lab")):
        for p in glob.glob(os.path.join(root, "*.json")):
            b = os.path.basename(p)[:-5]
            if b in tags:
                labs[b] = p
    print("[라벨] TAG %d · LAB %d" % (len(tags), len(labs)))

    rows, drop = [], collections.Counter()
    for stem in sorted(tags):
        if stem[:5] in excl:
            drop["학습화자"] += 1; continue
        wp = wavpath(stem)
        if not os.path.exists(wp):
            drop["wav없음"] += 1; continue
        fr = frames_of(wp)
        sec = fr / 16000.0
        if sec < a.min_sec or sec > a.max_sec:
            drop["길이"] += 1; continue
        ltr, words = parse_tag(tags[stem])
        canon = parse_lab(labs[stem]) if stem in labs else None
        if not ltr or not canon:
            drop["라벨파싱"] += 1; continue
        parts = stem.split("-")
        rows.append({"stem": stem, "wav": wp, "frames": fr, "ltr": ltr, "canon": canon,
                     "meta": {"stem": stem, "spk": parts[0], "item": ITEM.search(stem).group(1),
                              "nationality": parts[3] if len(parts) > 3 else None,
                              "language": parts[3] if len(parts) > 3 else None,
                              "prompt": words, "norm_prompt": re.sub(r"[^0-9A-Za-z가-힣]", "", words or ""),
                              "sec": round(sec, 2), "task": "RP"}})
    print("[선택] %d발화 · 제외 %s" % (len(rows), dict(drop)))
    if not rows:
        raise SystemExit("남은 발화가 없다 — 조건을 확인할 것")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "%s.tsv" % a.split), "w", encoding="utf-8") as f:
        f.write("/\n")
        for r in rows:
            f.write("%s\t%d\n" % (r["wav"].lstrip("/"), r["frames"]))
    for ext, key in (("ltr", "ltr"), ("canon", "canon")):
        with open(os.path.join(a.out, "%s.%s" % (a.split, ext)), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(u" ".join(r[key]) + "\n")
    with open(os.path.join(a.out, "%s.meta.jsonl" % a.split), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r["meta"], ensure_ascii=False) + "\n")
    src = a.dict_path or os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10", "dict.ltr.txt")
    if os.path.exists(src):
        open(os.path.join(a.out, "dict.ltr.txt"), "w", encoding="utf-8").write(open(src, encoding="utf-8").read())
    spk = {r["meta"]["spk"] for r in rows}
    stats = {"items": sorted(want), "utterances": len(rows), "speakers": len(spk),
             "excluded": dict(drop), "excluded_speaker_sets": a.excl,
             "hours": round(sum(r["frames"] for r in rows) / 16000.0 / 3600.0, 2),
             "by_nationality": dict(collections.Counter(r["meta"]["nationality"] for r in rows)),
             "note": "문항·화자 모두 Stage2 학습에 없음. RP=문단 읽기라 다른 split 과 길이·난이도가 다르다"}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print("[saved] %s · 화자 %d · %.2f시간" % (a.out, len(spk), stats["hours"]))


if __name__ == "__main__":
    main()

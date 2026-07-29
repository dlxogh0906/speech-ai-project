# -*- coding: utf-8 -*-
"""041(유럽)+043(아시아) TAG 통합 manifest — 자모. 화자분리 8:1:1."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import glob, os, json, re, hashlib, collections

A = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
E = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
SRC = [
    (f"{E}/TL_pronunciation_tag", f"{E}/TL_pronunciation_lab"),                                    # 041 유럽
    (f"{A}/Training/02.라벨링데이터/TL_pronunciation_tag", f"{A}/Training/02.라벨링데이터/TL_pronunciation_lab"),   # 043 아시아 TL
    (f"{A}/Validation/02.라벨링데이터/VL_pronunciation_tag", f"{A}/Validation/02.라벨링데이터/VL_pronunciation_lab"), # 043 아시아 VL
    (f"{DATA_ROOT}/aihub/040_labels/extracted/tag", f"{DATA_ROOT}/aihub/040_labels/extracted/lab"),      # 040 영어
    (f"{DATA_ROOT}/aihub/042_labels/extracted/tag", f"{DATA_ROOT}/aihub/042_labels/extracted/lab"),      # 042 중일
]
WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
OUT = f"{MDD_OUTPUTS}/manifest_tag_all"

def is_jamo(c): return 0x3131 <= ord(c) <= 0x3163
def wavpath(st): return os.path.join(WAV, st[:2], st + ".wav")
def frames_of(wp): return max(0, (os.path.getsize(wp) - 44) // 2)

def parse_tag(tg):
    try: txt = open(tg, encoding="utf-16").read()
    except:
        try: txt = open(tg, encoding="utf-8").read()
        except: return None
    m = re.search(r'name = "phones".*?(?=name = "|\Z)', txt, re.S)
    if not m: return None
    toks = []
    for lbl in re.findall(r'text = "([^"]*)"', m.group(0)):
        lbl = lbl.strip()
        if lbl in ("", ".", "-", "spn", "<unk>"): continue
        toks += [ch for ch in lbl if is_jamo(ch)]
    return toks

def parse_lab(lab):
    try: d = json.load(open(lab, encoding="utf-8"))
    except: return None
    ph = d.get("RecordingMetadata", {}).get("phonemic", {})
    if not ph or "phones" not in ph: return None
    toks = []
    for x in ph["phones"]:
        toks += [ch for ch in str(x.get("phone", "")).strip() if is_jamo(ch)]
    return toks

def split_of(spk):
    h = int(hashlib.md5(spk.encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "valid" if h == 1 else "train"

def main():
    os.makedirs(OUT, exist_ok=True)
    recs = []; seen = set()
    for tag_dir, lab_dir in SRC:
        print(f"[glob] {tag_dir.split('/')[-3]}/{tag_dir.split('/')[-1]}", flush=True)
        tags = {os.path.basename(p).split(".")[0]: p for p in glob.glob(f"{tag_dir}/*.TextGrid")}
        labs = {os.path.basename(p).split(".")[0]: p for p in glob.glob(f"{lab_dir}/*.json")}
        print(f"  TAG {len(tags):,} | LAB {len(labs):,}", flush=True)
        n = 0
        for st in tags:
            n += 1
            if st in seen or st not in labs: continue
            wp = wavpath(st)
            if not os.path.exists(wp): continue
            fr = frames_of(wp)
            if fr < 8000 or fr > 240000: continue   # 0.5초 미만 제외(초단 클립 마스킹 크래시 방지)
            ltr = parse_tag(tags[st]); canon = parse_lab(labs[st])
            if not ltr or not canon: continue
            seen.add(st)
            recs.append((st, wp, fr, ltr, canon, st.split("-")[0]))
            if n % 100000 == 0: print(f"    ...{n:,} · 누적유효 {len(recs):,}", flush=True)
    print(f"[유효 총] {len(recs):,}", flush=True)

    buckets = {"train": [], "valid": [], "test": []}
    for r in recs: buckets[split_of(r[5])].append(r)
    for s in buckets:
        spk = len(set(r[5] for r in buckets[s]))
        print(f"  {s}: {len(buckets[s]):,} ({spk} spk)", flush=True)

    freq = collections.Counter()
    for r in buckets["train"]: freq.update(r[3])
    with open(f"{OUT}/dict.ltr.txt", "w", encoding="utf-8") as f:
        for sym, c in freq.most_common(): f.write(f"{sym} {c}\n")
    print(f"[dict] {len(freq)} 자모", flush=True)

    for s, rows in buckets.items():
        with open(f"{OUT}/{s}.tsv", "w") as ft, open(f"{OUT}/{s}.ltr", "w", encoding="utf-8") as fl, \
             open(f"{OUT}/{s}.canon", "w", encoding="utf-8") as fc:
            ft.write("/\n")
            for st, wp, fr, ltr, canon, sp in rows:
                ft.write(f"{wp.lstrip('/')}\t{fr}\n")
                fl.write(" ".join(ltr) + "\n")
                fc.write(" ".join(canon) + "\n")
    print(f"[완료] {OUT}", flush=True)

if __name__ == "__main__":
    main()

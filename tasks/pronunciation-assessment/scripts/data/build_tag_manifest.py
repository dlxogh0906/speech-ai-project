# -*- coding: utf-8 -*-
"""TAG(전사자) 기반 CTC 학습 manifest 구축 — 자모 vocab.
  train.ltr  = TAG phones 티어(전사자, 자모)   ← CTC 정답
  train.canon= LAB phonemic.phones(표준음, 자모) ← MDD 평가용
  화자분리 8:1:1. 041(유럽) 우선.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import glob, os, json, re, hashlib, collections, argparse

BASE = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
TAG_DIR = f"{BASE}/TL_pronunciation_tag"
LAB_DIR = f"{BASE}/TL_pronunciation_lab"
WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
OUT = f"{MDD_OUTPUTS}/manifest_tag_041"

def is_jamo(c): return 0x3131 <= ord(c) <= 0x3163
def wavpath(st): return os.path.join(WAV, st[:2], st + ".wav")
def frames_of(wp): return max(0, (os.path.getsize(wp) - 44) // 2)  # 16k mono int16

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
    ap = argparse.ArgumentParser(); ap.add_argument("--max_frames", type=int, default=240000); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("[1] TAG/LAB glob...", flush=True)
    tags = {os.path.basename(p).split(".")[0]: p for p in glob.glob(f"{TAG_DIR}/*.TextGrid")}
    labs = {os.path.basename(p).split(".")[0]: p for p in glob.glob(f"{LAB_DIR}/*.json")}
    print(f"  TAG {len(tags):,} | LAB {len(labs):,}", flush=True)

    recs = []
    n = skip_audio = skip_parse = skip_len = 0
    for st in tags:
        n += 1
        if st not in labs: continue
        wp = wavpath(st)
        if not os.path.exists(wp): skip_audio += 1; continue
        fr = frames_of(wp)
        if fr < 400 or fr > a.max_frames: skip_len += 1; continue
        ltr = parse_tag(tags[st]); canon = parse_lab(labs[st])
        if not ltr or not canon: skip_parse += 1; continue
        recs.append((st, wp, fr, ltr, canon, st.split("-")[0]))
        if n % 50000 == 0: print(f"  ...{n:,}/{len(tags):,} · 유효 {len(recs):,}", flush=True)
    print(f"[유효] {len(recs):,} | 스킵: 오디오없음 {skip_audio:,} · 길이 {skip_len:,} · 파싱 {skip_parse:,}", flush=True)

    # 화자분리 8:1:1
    buckets = {"train": [], "valid": [], "test": []}
    for r in recs: buckets[split_of(r[5])].append(r)
    spk = {s: collections.Counter() for s in buckets}
    for s in buckets:
        for r in buckets[s]: spk[s][r[5]] += 1
    print(f"[split] train {len(buckets['train']):,}({len(spk['train'])}spk) / "
          f"valid {len(buckets['valid']):,}({len(spk['valid'])}spk) / "
          f"test {len(buckets['test']):,}({len(spk['test'])}spk)", flush=True)

    # dict (train 기준 빈도)
    freq = collections.Counter()
    for r in buckets["train"]:
        freq.update(r[3])
    with open(f"{OUT}/dict.ltr.txt", "w", encoding="utf-8") as f:
        for sym, c in freq.most_common():
            f.write(f"{sym} {c}\n")
    print(f"[dict] {len(freq)} 자모 심볼", flush=True)

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

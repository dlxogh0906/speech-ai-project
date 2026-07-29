# -*- coding: utf-8 -*-
"""디스크에 있는 48kHz gold(발음문항) 오디오 → 16k 리샘플 → mdd_gold_wav16k 확장.
현재 16k gold(17.6만)에 없는, korean_speech 폴더의 gold RW/RS/RP wav를 추가."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, MDD_ANALYSIS

import os, glob, json, wave, collections
from math import gcd
from multiprocessing import Pool
import numpy as np
from scipy.signal import resample_poly

OUT = f"{DATA_ROOT}/mdd_gold_wav16k"
PRONLAB = f"{MDD_ANALYSIS}/040_041_042_043_both_pronlab/records_pronlab.jsonl"
SRC_DIRS = [f"{DATA_ROOT}/aihub/{d}_korean_speech" for d in ["eng", "european", "sjr", "asian"]]


def rs16(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a = (a - 128) / 128 if sw == 1 else a / {2: 32768.0, 4: 2147483648.0}[sw]
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != 16000:
        g = gcd(int(sr), 16000); a = resample_poly(a, 16000 // g, sr // g)
    return a


def worker(item):
    src, stem = item
    dst = os.path.join(OUT, stem[:2], stem + ".wav")
    if os.path.exists(dst) and os.path.getsize(dst) > 44:
        return 0
    try:
        a = rs16(src)
        if a.size < 400:
            return -1
        pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with wave.open(dst, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
        return 1
    except Exception:
        return -1


def main():
    print("[1] gold stem 로드(records_pronlab)...", flush=True)
    gold = set()
    for line in open(PRONLAB, encoding="utf-8"):
        f = json.loads(line).get("file", "")
        gold.add(f[:-5] if f.endswith(".json") else f)
    print(f"    gold {len(gold):,}", flush=True)

    print("[2] 이미 16k인 것 인덱스...", flush=True)
    have16 = set(os.path.basename(p)[:-4] for p in glob.glob(f"{OUT}/**/*.wav", recursive=True))
    print(f"    16k 보유 {len(have16):,}", flush=True)

    print("[3] 48k 소스 인덱스(gold만)...", flush=True)
    todo = []
    seen = set()
    for d in SRC_DIRS:
        for p in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
            stem = os.path.basename(p)[:-4]
            if stem in gold and stem not in have16 and stem not in seen:
                seen.add(stem); todo.append((p, stem))
    print(f"    추출대상(48k→16k) {len(todo):,}", flush=True)

    got = 0; done = 0
    with Pool(14) as pool:
        for r in pool.imap_unordered(worker, todo, chunksize=64):
            done += 1
            if r == 1:
                got += 1
            if done % 20000 == 0:
                print(f"    {done:,}/{len(todo):,} (신규 {got:,})", flush=True)
    total = len(glob.glob(f"{OUT}/**/*.wav", recursive=True))
    print(f"[done] 신규 추출 {got:,} | mdd_gold_wav16k 총 {total:,} / gold {len(gold):,} ({total/len(gold)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()

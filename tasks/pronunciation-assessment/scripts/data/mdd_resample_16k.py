# -*- coding: utf-8 -*-
"""외국인 track B wav를 16kHz mono로 리샘플 + 매니페스트 갱신 (XLS-R 학습용).

- manifest의 tsv가 가리키는 wav를 16kHz mono int16으로 변환 → OUT_WAV(언어코드별 샤딩)
- 멀티프로세싱, 멱등(이미 있으면 skip). 실패 파일은 tsv/ltr/canon 3종에서 함께 제외.
- 새 매니페스트: {split}.tsv(16k 경로/프레임) + {split}.ltr/.canon(원본 유지, 실패분만 제거)
  → 출력 디렉토리 OUT_MAN
"""

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

import os, sys, wave
from math import gcd
from multiprocessing import Pool

import numpy as np
from scipy.signal import resample_poly

MAN = f"{MDD_OUTPUTS}/manifest"
OUT_MAN = f"{MDD_OUTPUTS}/manifest_16k"
OUT_WAV = f"{DATA_ROOT}/mdd_wav16k"
WORKERS = 14


def load_resample(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128.0
    elif sw == 4:
        a = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"sampwidth {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        a = resample_poly(a, 16000 // g, sr // g)
    return a


def worker(item):
    src, stem = item
    dst = os.path.join(OUT_WAV, stem[:2], stem + ".wav")
    if os.path.exists(dst) and os.path.getsize(dst) > 44:
        try:
            with wave.open(dst, "rb") as w:
                return stem, w.getnframes()
        except Exception:
            pass
    try:
        a = load_resample(src)
        if a.size < 400:
            return stem, None
        pcm = np.clip(a, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with wave.open(dst, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        return stem, len(pcm)
    except Exception:
        return stem, None


def main():
    os.makedirs(OUT_MAN, exist_ok=True)
    os.makedirs(OUT_WAV, exist_ok=True)

    # 전 split의 (src, stem) 수집 (중복 없음: stem 유일)
    per_split = {}
    work = []
    for split in ["train", "valid", "test"]:
        rows = []
        with open(f"{MAN}/{split}.tsv") as ft:
            ft.readline()  # root
            for line in ft:
                p = "/" + line.split("\t")[0]
                stem = os.path.basename(p)[:-4]
                rows.append((p, stem))
                work.append((p, stem))
        per_split[split] = rows
    print(f"[work] 리샘플 대상 {len(work):,} (workers {WORKERS})", flush=True)

    # 리샘플 (병렬)
    frames = {}
    done = 0
    with Pool(WORKERS) as pool:
        for stem, nf in pool.imap_unordered(worker, work, chunksize=64):
            if nf:
                frames[stem] = nf
            done += 1
            if done % 20000 == 0:
                print(f"  {done:,}/{len(work):,}  성공 {len(frames):,}", flush=True)
    print(f"[resample] 성공 {len(frames):,} / 실패 {len(work)-len(frames):,}", flush=True)

    # 매니페스트 갱신 (실패 stem은 tsv/ltr/canon에서 함께 제외)
    for split, rows in per_split.items():
        ltr = open(f"{MAN}/{split}.ltr").read().splitlines()
        can = open(f"{MAN}/{split}.canon").read().splitlines()
        assert len(rows) == len(ltr) == len(can), f"{split} 정합 깨짐"
        kept = 0
        with open(f"{OUT_MAN}/{split}.tsv", "w") as ft, \
             open(f"{OUT_MAN}/{split}.ltr", "w") as fl, \
             open(f"{OUT_MAN}/{split}.canon", "w") as fc:
            ft.write("/\n")
            for (p, stem), lt, cn in zip(rows, ltr, can):
                if stem not in frames:
                    continue
                dst = os.path.join(OUT_WAV, stem[:2], stem + ".wav")
                ft.write(f"{dst.lstrip('/')}\t{frames[stem]}\n")
                fl.write(lt + "\n")
                fc.write(cn + "\n")
                kept += 1
        print(f"  [{split}] {kept:,} utts (drop {len(rows)-kept})", flush=True)

    # dict, test_speakers 복사
    import shutil
    for f in ["dict.ltr.txt", "test_speakers.txt"]:
        if os.path.exists(f"{MAN}/{f}"):
            shutil.copy(f"{MAN}/{f}", f"{OUT_MAN}/{f}")
    print(f"\n[DONE] 16k 매니페스트 -> {OUT_MAN} | wav -> {OUT_WAV}", flush=True)


if __name__ == "__main__":
    main()

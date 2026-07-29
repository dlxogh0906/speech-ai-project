# -*- coding: utf-8 -*-
"""재다운로드한 pron zip에서 없는 gold stem만 추출 + 16k 리샘플 → mdd_gold_wav16k."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse, glob, io, json, os, wave, zipfile
from math import gcd
import numpy as np
from scipy.signal import resample_poly

OUT = f"{DATA_ROOT}/mdd_gold_wav16k"
MISS = f"{DATA_ROOT}/aihub/fullgold_missing.json"


def rs(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a = (a - 128) / 128 if sw == 1 else a / {2: 32768.0, 4: 2147483648.0}[sw]
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != 16000:
        g = gcd(int(sr), 16000); a = resample_poly(a, 16000 // g, sr // g)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pron_root", required=True)
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    mj = json.load(open(MISS))
    need = set()
    for v in mj.values():          # 전체 missing 합집합 (언어→zip 라우팅 불일치 대비: 어느 zip에 있든 잡음)
        need.update(v)
    zips = sorted(glob.glob(os.path.join(a.pron_root, "**", "*pronunciation_sound*.zip"), recursive=True))
    print(f"[{a.dataset}] 대상 {len(need):,} | pron zip {len(zips)}개", flush=True)
    got = 0
    for zp in zips:
        try:
            z = zipfile.ZipFile(zp)
        except Exception as e:
            print(f"[skip] {zp}: {e}", flush=True); continue
        for nm in z.namelist():
            if not nm.endswith(".wav"):
                continue
            stem = os.path.basename(nm)[:-4]
            if stem not in need:
                continue
            dst = os.path.join(OUT, stem[:2], stem + ".wav")
            if os.path.exists(dst) and os.path.getsize(dst) > 44:
                continue
            try:
                x = rs(z.read(nm))
                if x.size < 400:
                    continue
                pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with wave.open(dst, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
                got += 1
            except Exception:
                pass
        z.close()
        print(f"[{a.dataset}] {os.path.basename(zp)} 처리 · 누적 {got:,}", flush=True)
    print(f"[{a.dataset}] 신규 추출 {got:,}", flush=True)


if __name__ == "__main__":
    main()

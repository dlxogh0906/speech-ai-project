# -*- coding: utf-8 -*-
"""발음(pronunciation) zip에서 held-out gold basename만 추출 + 16kHz 리샘플.
zip 멤버를 메모리에서 바로 리샘플(원본 저장 안 함). 멱등.
사용: python mdd_gold_extract_pron.py --pron_root <다운로드dir> --dataset 040"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, glob, io, json, os, wave
from math import gcd
import numpy as np
from scipy.signal import resample_poly

OUT_WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
SPEC_NEEDED = f"{MDD_OUTPUTS}/gold_eval/needed_by_dataset.json"
OUT_DIR = f"{MDD_OUTPUTS}/gold_eval"


def resample_bytes(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, {1:np.uint8,2:np.int16,4:np.int32}[sw]).astype(np.float32)
    a = (a-128)/128 if sw==1 else a/{2:32768.0,4:2147483648.0}[sw]
    if ch>1: a=a.reshape(-1,ch).mean(1)
    if sr!=16000:
        g=gcd(int(sr),16000); a=resample_poly(a,16000//g,sr//g)
    return a


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pron_root", required=True)
    ap.add_argument("--dataset", required=True)   # 040/042/043
    a=ap.parse_args()
    need=set(json.load(open(SPEC_NEEDED))[a.dataset])
    print(f"[{a.dataset}] 필요 gold basename {len(need):,}", flush=True)

    frames={}
    fpath=f"{OUT_DIR}/frames_{a.dataset}.json"
    if os.path.exists(fpath): frames=json.load(open(fpath))
    zips=sorted(glob.glob(os.path.join(a.pron_root,"**","*pronunciation_sound*.zip"),recursive=True))
    print(f"[{a.dataset}] pron zip {len(zips)}개", flush=True)
    got=0
    import zipfile
    for zp in zips:
        try: z=zipfile.ZipFile(zp)
        except Exception as e: print(f"[skip] {zp}: {e}",flush=True); continue
        members=[n for n in z.namelist() if n.endswith(".wav") and os.path.basename(n)[:-4] in need]
        for n in members:
            stem=os.path.basename(n)[:-4]
            dst=os.path.join(OUT_WAV, stem[:2], stem+".wav")
            if stem in frames and os.path.exists(dst): continue
            try:
                x=resample_bytes(z.read(n))
                if x.size<400: continue
                pcm=(np.clip(x,-1,1)*32767).astype(np.int16)
                os.makedirs(os.path.dirname(dst),exist_ok=True)
                with wave.open(dst,"wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
                frames[stem]=len(pcm); got+=1
            except Exception: pass
        z.close()
        print(f"[unzip] {os.path.basename(zp)}: 대상 {len(members):,}, 누적 {got:,}", flush=True)
    json.dump(frames, open(fpath,"w"))
    have=len([s for s in need if s in frames])
    print(f"[{a.dataset}] 완료: 확보 {have:,}/{len(need):,} -> {OUT_WAV}", flush=True)


if __name__ == "__main__":
    main()

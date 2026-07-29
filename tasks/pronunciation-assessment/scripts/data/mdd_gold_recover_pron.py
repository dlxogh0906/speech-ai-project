# -*- coding: utf-8 -*-
"""아직 미연결인 gold(041/043)를 발음 zip에서 추가 추출+16k. (재다운로드분 대상)
still_miss.json의 basename만 추출. 원본에 없으면 '결손' 보고."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, glob, io, json, os, wave, zipfile
from math import gcd
import numpy as np
from scipy.signal import resample_poly

OUT=f"{DATA_ROOT}/mdd_gold_wav16k"
STILL="/tmp/still_miss.json"
FRAMES_DIR=f"{MDD_OUTPUTS}/gold_eval"


def rs(data):
    with wave.open(io.BytesIO(data),"rb") as w:
        sr,ch,sw,n=w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes(); raw=w.readframes(n)
    a=np.frombuffer(raw,{1:np.uint8,2:np.int16,4:np.int32}[sw]).astype(np.float32)
    a=(a-128)/128 if sw==1 else a/{2:32768.0,4:2147483648.0}[sw]
    if ch>1: a=a.reshape(-1,ch).mean(1)
    if sr!=16000: g=gcd(int(sr),16000); a=resample_poly(a,16000//g,sr//g)
    return a


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pron_root", required=True)
    ap.add_argument("--dataset", required=True)
    a=ap.parse_args()
    need=set(json.load(open(STILL)).get(a.dataset,[]))
    print(f"[{a.dataset}] 복구 대상 {len(need):,}", flush=True)
    zips=sorted(glob.glob(os.path.join(a.pron_root,"**","*pronunciation_sound*.zip"),recursive=True))
    print(f"[{a.dataset}] pron zip {len(zips)}개", flush=True)
    fp=f"{FRAMES_DIR}/frames_{a.dataset}.json"
    frames=json.load(open(fp)) if os.path.exists(fp) else {}
    got=0; found_in_zip=set()
    for zp in zips:
        try: z=zipfile.ZipFile(zp)
        except Exception as e: print(f"[skip] {zp}: {e}"); continue
        for n in z.namelist():
            if not n.endswith(".wav"): continue
            stem=os.path.basename(n)[:-4]
            if stem not in need: continue
            found_in_zip.add(stem)
            dst=os.path.join(OUT,stem[:2],stem+".wav")
            if os.path.exists(dst) and os.path.getsize(dst)>44: continue
            try:
                x=rs(z.read(n))
                if x.size<400: continue
                pcm=(np.clip(x,-1,1)*32767).astype(np.int16)
                os.makedirs(os.path.dirname(dst),exist_ok=True)
                with wave.open(dst,"wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
                frames[stem]=len(pcm); got+=1
            except: pass
        z.close()
        print(f"[unzip] {os.path.basename(zp)} 처리", flush=True)
    json.dump(frames, open(fp,"w"))
    absent=need - found_in_zip
    print(f"[{a.dataset}] 신규추출 {got:,} | zip에 존재 {len(found_in_zip):,} | zip에도 없음(원본결손) {len(absent):,}", flush=True)
    json.dump(sorted(absent), open(f"{FRAMES_DIR}/absent_{a.dataset}.json","w"))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""held-out(LAR-test) 화자의 gold(발음과제) F1 평가 스펙 생성 + 041 gold 오디오 추출.

- gold_eval_spec.jsonl: 평가 대상 발화별 (stem, spk, langcode, target_phones, has_error, err positions/types)
- needed_by_dataset.json: 데이터셋(040/042/043/041)별 필요 basename 목록 (다운로드/추출용)
- 041(보유)분은 지금 16kHz 추출 → gold 평가셋 부분 완성
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, MDD_ANALYSIS, MDD_OUTPUTS

import os, json, glob, wave, collections
from math import gcd
from multiprocessing import Pool
import numpy as np
from scipy.signal import resample_poly

PRONLAB = f"{MDD_ANALYSIS}/040_041_042_043_both_pronlab/records_pronlab.jsonl"
LAR_TEST_TSV = f"{MDD_OUTPUTS}/manifest_16k/test.tsv"
OUT_DIR = f"{MDD_OUTPUTS}/gold_eval"
OUT_WAV = f"{DATA_ROOT}/mdd_gold_wav16k"                 # 추출·리샘플된 gold 오디오
EU_LANGS = {"ES","RU","DE","FR","PT","IT","TR","PL","HU","RO","CS","AZ","UZ","HY","BG","UA","KK","TM","KY","TJ"}


def dataset_of(lc):
    if lc == "EN": return "040"
    if lc in {"ZH","JA"}: return "042"
    if lc in EU_LANGS: return "041"
    return "043"


def load_resample_16k(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, {1:np.uint8,2:np.int16,4:np.int32}[sw]).astype(np.float32)
    a = (a-128)/128 if sw==1 else a/{2:32768.0,4:2147483648.0}[sw]
    if ch>1: a=a.reshape(-1,ch).mean(1)
    if sr!=16000:
        g=gcd(int(sr),16000); a=resample_poly(a,16000//g,sr//g)
    return a


def resample_worker(item):
    src, stem = item
    dst = os.path.join(OUT_WAV, stem[:2], stem+".wav")
    if os.path.exists(dst) and os.path.getsize(dst)>44:
        try:
            with wave.open(dst,"rb") as w: return stem, w.getnframes()
        except: pass
    try:
        a=load_resample_16k(src)
        if a.size<400: return stem, None
        pcm=(np.clip(a,-1,1)*32767).astype(np.int16)
        os.makedirs(os.path.dirname(dst),exist_ok=True)
        with wave.open(dst,"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
        return stem, len(pcm)
    except: return stem, None


def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(OUT_WAV, exist_ok=True)

    # 1) LAR test 화자키
    test_spk=set()
    for line in open(LAR_TEST_TSV):
        if line.startswith("/"): continue
        b=os.path.basename(line.split("\t")[0])[:-4]
        test_spk.add("-".join(b.split("-")[:4]))
    print(f"[held-out] LAR test 화자 {len(test_spk):,}", flush=True)

    # 2) gold 스펙 (held-out 화자 발화만)
    spec=[]; need=collections.defaultdict(list)
    for line in open(PRONLAB, encoding="utf-8"):
        d=json.loads(line); f=d.get("file","")
        stem=f[:-5] if f.endswith(".json") else f
        spk="-".join(f.split("-")[:4])
        if spk not in test_spk: continue
        lc=f.split("-")[3] if len(f.split("-"))>3 else "?"
        ds=dataset_of(lc)
        errs=d.get("errors") or []
        spec.append({"stem":stem,"spk":spk,"langcode":lc,"dataset":ds,
                     "L1":d.get("L1"),"target_phones":d.get("target_phones"),
                     "has_error":bool(d.get("has_error")),
                     "err_pos":[e.get("pos") for e in errs],
                     "err_types":[e.get("error_type") for e in errs]})
        need[ds].append(stem)
    with open(f"{OUT_DIR}/gold_eval_spec.jsonl","w",encoding="utf-8") as fo:
        for r in spec: fo.write(json.dumps(r,ensure_ascii=False)+"\n")
    json.dump({k:v for k,v in need.items()}, open(f"{OUT_DIR}/needed_by_dataset.json","w"))
    ne=sum(1 for r in spec if r["has_error"])
    print(f"[spec] gold 평가대상 {len(spec):,} (오류 {ne:,} / 정상 {len(spec)-ne:,})", flush=True)
    for ds in ["040","041","042","043"]:
        print(f"  {ds}: {len(need[ds]):,}", flush=True)

    # 3) 041(보유) gold 오디오 16k 추출
    idx041={}
    for r,_,fs in os.walk(f"{DATA_ROOT}/aihub/european_korean_speech"):
        for ff in fs:
            if ff.endswith(".wav"): idx041[ff[:-4]]=os.path.join(r,ff)
    todo=[(idx041[s],s) for s in need["041"] if s in idx041]
    print(f"\n[041] 추출 대상 {len(todo):,} / gold041 {len(need['041']):,} (누락 {len(need['041'])-len(todo)})", flush=True)
    frames={}; done=0
    with Pool(14) as pool:
        for stem,nf in pool.imap_unordered(resample_worker, todo, chunksize=64):
            if nf: frames[stem]=nf
            done+=1
            if done%10000==0: print(f"  {done:,}/{len(todo):,}", flush=True)
    json.dump(frames, open(f"{OUT_DIR}/frames_041.json","w"))
    print(f"[041] 16k 추출 완료 {len(frames):,} -> {OUT_WAV}", flush=True)


if __name__ == "__main__":
    main()

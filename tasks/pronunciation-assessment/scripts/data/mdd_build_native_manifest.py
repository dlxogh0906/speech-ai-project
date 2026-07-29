# -*- coding: utf-8 -*-
"""원어민 track A 매니페스트 (001) + 외국인 track B와 통합.

- 화자당 CAP발화만 (seed 고정 샘플링) → 원어민 과대표집 방지
- 전사(stt) → KoG2P 음소열 (records dict 심볼셋과 정합 확인됨)
- 16kHz mono 리샘플 (병렬)
- 화자단위 8:1:1 (원어민은 001 화자폴더 = speaker)
- 출력: manifest_native/ (원어민 단독) + manifest_merged/ (B+A 통합, kind 태그 tsv3열)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, G2P_TOOLS, MDD_OUTPUTS, ensure_kog2p

import os, sys, glob, json, wave, random, collections
from math import gcd
from multiprocessing import Pool
import numpy as np
ensure_kog2p()
from scipy.signal import resample_poly
from g2p import readRules, graph2prono, ver_info

TRANS = f"{DATA_ROOT}/aihub/001_labels/extracted_matched"      # 매칭 전사 JSON
AUDIO_ROOT = f"{DATA_ROOT}/aihub/001.자유대화(일반남여)"
RULEBOOK = f"{G2P_TOOLS}/KoG2P/rulebook.txt"
OUT_WAV = f"{DATA_ROOT}/mdd_wav16k_native"
OUT_NAT = f"{MDD_OUTPUTS}/manifest_native"
OUT_MRG = f"{MDD_OUTPUTS}/manifest_merged"
MAN_B = f"{MDD_OUTPUTS}/manifest_16k"          # 외국인(16k)
CAP = 100
SEED = 42
WORKERS = 14

_RIN, _ROUT = readRules(ver_info[0], RULEBOOK)


def kog2p(text):
    try:
        p = graph2prono(text, _RIN, _ROUT)
        return p.split() if p else None
    except Exception:
        return None


def resample_worker(item):
    src, stem = item
    dst = os.path.join(OUT_WAV, stem[:8], stem + ".wav")
    if os.path.exists(dst) and os.path.getsize(dst) > 44:
        try:
            with wave.open(dst, "rb") as w:
                return stem, w.getnframes()
        except Exception:
            pass
    try:
        with wave.open(src, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128.0
        elif sw == 4:
            a = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
        else:
            return stem, None
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        if sr != 16000:
            g = gcd(int(sr), 16000)
            a = resample_poly(a, 16000 // g, sr // g)
        if a.size < 400:
            return stem, None
        pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with wave.open(dst, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        return stem, len(pcm)
    except Exception:
        return stem, None


def main():
    for d in (OUT_WAV, OUT_NAT, OUT_MRG):
        os.makedirs(d, exist_ok=True)

    # 1) 오디오 basename → path
    wav_of = {}
    for r, _, fs in os.walk(AUDIO_ROOT):
        for f in fs:
            if f.endswith(".wav"):
                wav_of[f[:-4]] = os.path.join(r, f)
    print(f"[audio] {len(wav_of):,} wav", flush=True)

    # 2) 오디오 매칭되는 전사 목록만 (아직 g2p·stt읽기 안 함) → 화자당 CAP 먼저
    trans_files = [f for f in os.listdir(TRANS) if f.endswith(".json")]
    by_spk = collections.defaultdict(list)
    n_noaudio = 0
    for jf in trans_files:
        stem = jf[:-5]
        if stem not in wav_of:
            n_noaudio += 1; continue
        spk = "_".join(stem.split("_")[:-1])          # 화자폴더(=화자)
        by_spk[spk].append(stem)
    rng = random.Random(SEED)
    picked = []
    for spk in sorted(by_spk):
        lst = by_spk[spk]; rng.shuffle(lst)
        picked.extend((s, spk) for s in lst[:CAP])
    print(f"[cap] 화자 {len(by_spk):,}, 화자당≤{CAP} → {len(picked):,} (오디오없음 {n_noaudio})", flush=True)

    # 3) 선택분만 KoG2P (cap 이후라 ~6만만 처리)
    items = []  # (stem, path, spk, phones)
    n_notrans = n_nog2p = 0
    for i, (stem, spk) in enumerate(picked):
        try:
            stt = json.load(open(os.path.join(TRANS, stem + ".json"), encoding="utf-8"))["발화정보"]["stt"]
        except Exception:
            n_notrans += 1; continue
        ph = kog2p(stt)
        if not ph:
            n_nog2p += 1; continue
        items.append((stem, wav_of[stem], spk, ph))
        if i % 20000 == 0:
            print(f"  g2p {i:,}/{len(picked):,}", flush=True)
    capped = items
    print(f"[g2p] 음소화 성공 {len(capped):,} (전사실패 {n_notrans}, g2p실패 {n_nog2p})", flush=True)

    # 4) 16k 리샘플 (병렬)
    frames = {}
    done = 0
    with Pool(WORKERS) as pool:
        for stem, nf in pool.imap_unordered(resample_worker, [(it[1], it[0]) for it in capped], chunksize=64):
            if nf:
                frames[stem] = nf
            done += 1
            if done % 20000 == 0:
                print(f"  resample {done:,}/{len(capped):,}", flush=True)
    capped = [it for it in capped if it[0] in frames]
    print(f"[resample] 성공 {len(capped):,}", flush=True)

    # 5) 화자단위 8:1:1 (원어민은 국적 단일이라 화자만 셔플 분할)
    spks = sorted({it[2] for it in capped})
    rng2 = np.random.default_rng(SEED); rng2.shuffle(spks)
    n = len(spks); n_tr = int(round(n * 0.8)); n_va = int(round(n * 0.1))
    split_of = {}
    for idx, s in enumerate(spks):
        split_of[s] = "train" if idx < n_tr else ("valid" if idx < n_tr + n_va else "test")

    # 6) native 매니페스트 (원어민은 실현형=정답 → ltr=canon 동일)
    buckets = collections.defaultdict(list)
    for it in capped:
        buckets[split_of[it[2]]].append(it)
    for split in ["train", "valid", "test"]:
        rows = buckets[split]
        with open(f"{OUT_NAT}/{split}.tsv", "w") as ft, open(f"{OUT_NAT}/{split}.ltr", "w") as fl, \
             open(f"{OUT_NAT}/{split}.canon", "w") as fc:
            ft.write("/\n")
            for stem, _, spk, ph in rows:
                dst = os.path.join(OUT_WAV, stem[:8], stem + ".wav")
                ft.write(f"{dst.lstrip('/')}\t{frames[stem]}\n")
                fl.write(" ".join(ph) + "\n")
                fc.write(" ".join(ph) + "\n")
        print(f"  [native {split}] {len(rows):,} utts, {len({r[2] for r in rows})} spk", flush=True)

    # 7) 통합 매니페스트 (B + A), tsv에 kind 3열(foreign/native) 추가
    import shutil
    for split in ["train", "valid", "test"]:
        with open(f"{OUT_MRG}/{split}.tsv", "w") as ft, open(f"{OUT_MRG}/{split}.ltr", "w") as fl, \
             open(f"{OUT_MRG}/{split}.canon", "w") as fc:
            ft.write("/\n")
            # 외국인 B
            bl = open(f"{MAN_B}/{split}.ltr").read().splitlines()
            bc = open(f"{MAN_B}/{split}.canon").read().splitlines()
            bt = open(f"{MAN_B}/{split}.tsv").read().splitlines()[1:]
            for t, l, c in zip(bt, bl, bc):
                ft.write(f"{t}\tforeign\n"); fl.write(l + "\n"); fc.write(c + "\n")
            # 원어민 A
            for stem, _, spk, ph in buckets[split]:
                dst = os.path.join(OUT_WAV, stem[:8], stem + ".wav")
                ft.write(f"{dst.lstrip('/')}\t{frames[stem]}\tnative\n")
                fl.write(" ".join(ph) + "\n"); fc.write(" ".join(ph) + "\n")
        nb, na = len(bt), len(buckets[split])
        print(f"  [merged {split}] foreign {nb:,} + native {na:,} = {nb+na:,}", flush=True)
    shutil.copy(f"{MAN_B}/dict.ltr.txt", f"{OUT_MRG}/dict.ltr.txt")
    shutil.copy(f"{MAN_B}/dict.ltr.txt", f"{OUT_NAT}/dict.ltr.txt")

    json.dump({"cap": CAP, "native_utt": len(capped), "native_spk": len(spks),
               "splits": {s: len(buckets[s]) for s in ["train", "valid", "test"]}},
              open(f"{OUT_NAT}/stats.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n[DONE] native -> {OUT_NAT} | merged -> {OUT_MRG}", flush=True)


if __name__ == "__main__":
    main()

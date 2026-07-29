# -*- coding: utf-8 -*-
"""LAR 도메인 적응용 혼합 manifest 빌더.

1) record에서 '전사 == 프롬프트'(한 글자도 안 틀리고 읽은) LAR 발화만 선별
   → 이 발화들은 G2P(프롬프트) 라벨이 실제 발음과 일치 (약라벨의 안전한 부분집합)
2) 48kHz wav → 16kHz 리샘플 ($SPEECH_AI_DATA/mdd_lar_wav16k)
3) manifest_tag_all(train 143만, 강라벨) + LAR(업샘플, 전체의 ~7%) → manifest_tag_all_lar
   valid/test/dict 는 tag_all 그대로 복사 (게이트① 비교 가능성 유지)
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
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_ANALYSIS, MDD_OUTPUTS, ensure_kog2p

import sys, os, json, wave, shutil, math
from multiprocessing import Pool
import numpy as np

sys.argv = [sys.argv[0]]
ensure_kog2p()
from mdd import pipeline as MD
from g2p import readRules, ver_info
MD.S["ri"], MD.S["ro"] = readRules(ver_info[0], MD.RULEBOOK)

RECORDS = f"{MDD_ANALYSIS}/040_041_042_043_both/records.jsonl"
DIRS = [f"{DATA_ROOT}/aihub/european_korean_speech/selected_speech_fix",
        f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech",
        f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech_manifest"]
SRC_M = f"{MDD_OUTPUTS}/manifest_tag_all"
OUT_M = f"{MDD_OUTPUTS}/manifest_tag_all_lar"
WAV16 = f"{DATA_ROOT}/mdd_lar_wav16k"
TARGET_FRAC = 0.07          # 업샘플 후 LAR 비율 목표
H = lambda c: 0xAC00 <= ord(c) <= 0xD7A3


def clean(s):
    return "".join(c for c in s if H(c) or c == " ").strip()


def hangul_only(s):
    return all(H(c) or c in " .?!,~" for c in s.strip())


def resample_one(args):
    src, dst = args
    try:
        from scipy.signal import resample_poly
        with wave.open(src) as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if sr != 16000:
            g = math.gcd(16000, sr)
            a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
        n = len(a)
        if not (8000 <= n <= 320000):     # 0.5s~20s
            return None
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with wave.open(dst, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        return n
    except Exception:
        return None


def main():
    aidx = {}
    for d in DIRS:
        for f in os.listdir(d):
            if "LAR" in f and f.endswith(".wav"):
                aidx.setdefault(f[:-4], os.path.join(d, f))

    dict_jamo = set(l.split()[0] for l in open(f"{SRC_M}/dict.ltr.txt", encoding="utf-8"))
    sel, n_lar_total, n_clean = {}, 0, 0
    for line in open(RECORDS, encoding="utf-8"):
        r = json.loads(line)
        if r["task"] != "LAR" or r.get("quality_flag") != "ok":
            continue
        stem = r["file"][:-5]
        if stem not in aidx or stem in sel:
            continue
        n_lar_total += 1
        pr, tr = clean(r.get("prompt", "")), clean(r.get("orthographic_clean", ""))
        if not (pr and hangul_only(r.get("prompt", "")) and hangul_only(r.get("orthographic_clean", ""))):
            continue
        if pr.replace(" ", "") != tr.replace(" ", ""):
            continue                       # 무오류(정확히 읽은) 발화만
        n_clean += 1
        try:
            canon, _ = MD.canonical(pr)
        except Exception:
            continue
        if not canon or not set(canon) <= dict_jamo:
            continue
        sel[stem] = {"wav": aidx[stem], "ltr": canon}
    print(f"[선별] 오디오 있는 LAR {n_lar_total:,} → 무오류(전사=프롬프트) {n_clean:,} → 라벨 생성 성공 {len(sel):,}", flush=True)

    work = [(v["wav"], f"{WAV16}/{k[:2]}/{k}.wav") for k, v in sel.items()]
    with Pool(16) as p:
        frames = p.map(resample_one, work, chunksize=64)
    rows = []
    for (k, v), n in zip(sel.items(), frames):
        if n is not None:
            rows.append((f"data/tta/mdd_lar_wav16k/{k[:2]}/{k}.wav", n, v["ltr"]))
    print(f"[리샘플] 성공 {len(rows):,} / 제외 {len(work)-len(rows):,}", flush=True)

    # 업샘플 배수 결정
    src_train = open(f"{SRC_M}/train.tsv").read().splitlines()
    n_src = len(src_train) - 1
    up = max(1, round(TARGET_FRAC * n_src / max(1, len(rows))))
    frac = up * len(rows) / (n_src + up * len(rows))
    print(f"[혼합] RW/RS {n_src:,} + LAR {len(rows):,}×{up} = LAR 비율 {frac:.1%}", flush=True)

    os.makedirs(OUT_M, exist_ok=True)
    src_ltr = open(f"{SRC_M}/train.ltr", encoding="utf-8").read().splitlines()
    src_canon = open(f"{SRC_M}/train.canon", encoding="utf-8").read().splitlines()
    with open(f"{OUT_M}/train.tsv", "w") as ft, \
         open(f"{OUT_M}/train.ltr", "w", encoding="utf-8") as fl, \
         open(f"{OUT_M}/train.canon", "w", encoding="utf-8") as fc:
        ft.write(src_train[0] + "\n")
        for i in range(n_src):
            ft.write(src_train[1 + i] + "\n"); fl.write(src_ltr[i] + "\n"); fc.write(src_canon[i] + "\n")
        for _ in range(up):
            for rel, n, ltr in rows:
                ft.write(f"{rel}\t{n}\n")
                fl.write(" ".join(ltr) + "\n"); fc.write(" ".join(ltr) + "\n")
    for split in ("valid", "test"):
        for ext in ("tsv", "ltr", "canon"):
            shutil.copy(f"{SRC_M}/{split}.{ext}", f"{OUT_M}/{split}.{ext}")
    shutil.copy(f"{SRC_M}/dict.ltr.txt", f"{OUT_M}/dict.ltr.txt")
    print(f"[완료] {OUT_M} · train {n_src + up*len(rows):,}행", flush=True)


if __name__ == "__main__":
    main()

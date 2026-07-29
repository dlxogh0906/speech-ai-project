# -*- coding: utf-8 -*-
"""외국인 track B 매니페스트 생성 (MDD / XLS-R+CTC).

- records.jsonl 268,533건 → wav 경로연결 (041 추출본 + 040/042 selected + 043 zip에서 추출)
- 화자(파일명 앞4토큰 = UserID-gender-birth-langcode) 단위 8:1:1, L1 층화, seed 42
- fairseq CTC 형식: {split}.tsv (오디오) + {split}.ltr (실현형 음소열=actual_phones)
  + {split}.canon (정답 음소열=target_phones, MDD 비교용) + dict.ltr.txt
- test 화자 목록·통계 저장. (음소열 = 학습 타깃, 팩 dict 도착 시 심볼셋 정합확인 예정)
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
from shared.paths import DATA_ROOT, MDD_ANALYSIS, MDD_OUTPUTS

import collections
import glob
import json
import os
import wave
import zipfile

import numpy as np

RECORDS = f"{MDD_ANALYSIS}/040_041_042_043_both/records.jsonl"
OUT = f"{MDD_OUTPUTS}/manifest"
WAV_DIRS = [
    f"{DATA_ROOT}/aihub/european_korean_speech",                      # 041 추출본(+fix)
    f"{DATA_ROOT}/aihub/eng_korean_speech/selected_speech",           # 040
    f"{DATA_ROOT}/aihub/sjr_korean_speech/selected_speech",           # 042(+ZA)
]
ASIA043 = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
ASIA043_OUT = f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech_manifest"
SEED = 42
RATIO = (0.8, 0.1, 0.1)


def build_index():
    idx = {}
    for d in WAV_DIRS:
        for r, _, fs in os.walk(d):
            for f in fs:
                if f.endswith(".wav"):
                    idx.setdefault(f[:-4], os.path.join(r, f))
    print(f"[index] 폴더 추출본 wav {len(idx):,}", flush=True)
    return idx


def extract_043(need_stems):
    """043 zip에서 records가 참조하는 wav만 추출 (멱등)."""
    os.makedirs(ASIA043_OUT, exist_ok=True)
    have = {f[:-4] for f in os.listdir(ASIA043_OUT) if f.endswith(".wav")}
    todo = need_stems - have
    if not todo:
        print(f"[043] 이미 추출됨 {len(have):,}", flush=True)
        return
    print(f"[043] zip에서 {len(todo):,}건 추출 시작", flush=True)
    zips = sorted(glob.glob(os.path.join(ASIA043, "**", "*Speech_sound*.zip"), recursive=True))
    got = 0
    for zp in zips:
        try:
            z = zipfile.ZipFile(zp)
        except Exception as e:
            print(f"[043 skip] {os.path.basename(zp)}: {e}", flush=True)
            continue
        members = [n for n in z.namelist()
                   if n.endswith(".wav") and os.path.basename(n)[:-4] in todo]
        for n in members:
            dst = os.path.join(ASIA043_OUT, os.path.basename(n))
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue
            with z.open(n) as src, open(dst, "wb") as out:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            got += 1
        z.close()
        print(f"[043 unzip] {os.path.basename(zp)}: +{len([m for m in members])} (누적신규 {got:,})", flush=True)


def nframes(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes(), w.getframerate()
    except Exception:
        return 0, 0


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) records 로드
    recs = []
    for line in open(RECORDS, encoding="utf-8"):
        d = json.loads(line)
        f = d.get("file", "")
        recs.append({
            "stem": f[:-5] if f.endswith(".json") else f,
            "spk": "-".join(f.split("-")[:4]),
            "L1": d.get("L1", "?"),
            "lang": d.get("language", "?"),
            "actual": d.get("actual_phones", []),
            "target": d.get("target_phones", []),
        })
    print(f"[records] {len(recs):,}", flush=True)

    # 2) 폴더 인덱스 + 043 추출
    idx = build_index()
    need_043 = {r["stem"] for r in recs if r["stem"] not in idx}
    extract_043(need_043)
    for f in os.listdir(ASIA043_OUT):
        if f.endswith(".wav"):
            idx.setdefault(f[:-4], os.path.join(ASIA043_OUT, f))
    print(f"[index] 043 추가 후 총 wav {len(idx):,}", flush=True)

    # 3) 경로연결 + 프레임
    resolved, miss = [], 0
    rates = collections.Counter()
    for r in recs:
        p = idx.get(r["stem"])
        if not p:
            miss += 1
            continue
        r["path"] = p
        resolved.append(r)
    print(f"[resolve] 연결 {len(resolved):,} / 미연결 {miss}", flush=True)

    print("[frames] wav 헤더 읽는 중 ...", flush=True)
    ok = []
    for i, r in enumerate(resolved):
        nf, sr = nframes(r["path"])
        if nf <= 0:
            continue
        r["frames"] = nf
        rates[sr] += 1
        ok.append(r)
        if i % 40000 == 0:
            print(f"  {i:,}/{len(resolved):,}", flush=True)
    print(f"[frames] 유효 {len(ok):,} | 샘플레이트 분포 {dict(rates)}", flush=True)

    # 4) 화자 L1 층화 8:1:1 (seed 고정)
    spk_L1 = {}
    for r in ok:
        spk_L1[r["spk"]] = r["L1"]
    by_L1 = collections.defaultdict(list)
    for spk, l1 in spk_L1.items():
        by_L1[l1].append(spk)
    rng = np.random.default_rng(SEED)
    split_of = {}
    for l1 in sorted(by_L1):
        spks = sorted(by_L1[l1])
        rng.shuffle(spks)
        n = len(spks)
        n_tr = int(round(n * RATIO[0]))
        n_va = int(round(n * RATIO[1]))
        # 화자 3명 이상일 때만 valid/test 확보, 아니면 전부 train
        if n < 3:
            for s in spks:
                split_of[s] = "train"
            continue
        for i, s in enumerate(spks):
            split_of[s] = "train" if i < n_tr else ("valid" if i < n_tr + n_va else "test")
    print(f"[split] 화자 {len(split_of):,} (L1 {len(by_L1)}개 층화)", flush=True)

    # 5) fairseq 매니페스트 작성
    buckets = {"train": [], "valid": [], "test": []}
    for r in ok:
        buckets[split_of[r["spk"]]].append(r)

    dict_cnt = collections.Counter()
    for split, rows in buckets.items():
        with open(f"{OUT}/{split}.tsv", "w", encoding="utf-8") as ft, \
             open(f"{OUT}/{split}.ltr", "w", encoding="utf-8") as fl, \
             open(f"{OUT}/{split}.canon", "w", encoding="utf-8") as fc:
            ft.write("/\n")  # root
            for r in rows:
                ft.write(f"{r['path'].lstrip('/')}\t{r['frames']}\n")
                fl.write(" ".join(r["actual"]) + "\n")
                fc.write(" ".join(r["target"]) + "\n")
                if split == "train":
                    dict_cnt.update(r["actual"])
                    dict_cnt.update(r["target"])
        print(f"  [{split}] {len(rows):,} utts, {len({r['spk'] for r in rows})} spk", flush=True)

    with open(f"{OUT}/dict.ltr.txt", "w", encoding="utf-8") as fd:
        for sym, c in dict_cnt.most_common():
            fd.write(f"{sym} {c}\n")
    print(f"[dict] 음소 심볼 {len(dict_cnt)}종 -> dict.ltr.txt", flush=True)

    # test 화자 목록 + 통계
    with open(f"{OUT}/test_speakers.txt", "w", encoding="utf-8") as f:
        for s in sorted(s for s, sp in split_of.items() if sp == "test"):
            f.write(s + "\n")
    stats = {
        "total_utt": len(ok), "n_speakers": len(split_of), "n_L1": len(by_L1),
        "n_phones": len(dict_cnt), "sample_rates": dict(rates),
        "splits": {sp: {"utt": len(rows), "spk": len({r["spk"] for r in rows})}
                   for sp, rows in buckets.items()},
        "seed": SEED, "ratio": RATIO,
    }
    json.dump(stats, open(f"{OUT}/stats.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n[DONE] -> {OUT}")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

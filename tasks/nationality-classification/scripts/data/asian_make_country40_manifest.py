# -*- coding: utf-8 -*-
"""043(아시아어) → 선별 14클래스 매니페스트 + 화자단위 선별 오디오 추출.

- 대상: 마지막 best에서 저성능/평가불가였던 클래스만 (1·2순위 리스트).
  이미 양호한 클래스(인니·베트남·몽골·태국·미얀마·중국·튀르키예·프랑스·이탈리아)는 의도적으로 제외.
- Speech(문장과제) 오디오만 존재(발음 오디오는 미다운로드).
- zip에서 선별 화자의 wav만 추출(전체 190GB 중 ~수십GB) → selected_speech/ 평면 저장.
- 매니페스트는 문장(orthographic ≥4단어)만 포함, 화자ID 'AS_' prefix (041의 EU_와 동일 관례).
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
from shared.paths import DATA_ROOT, NAT_DATA

import csv
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

BASE = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
OUT_AUDIO = f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech"
OUT_CSV = f"{NAT_DATA}/asian_country40_manifest.csv"
MIN_WORDS = 4

# 선별 14클래스 (1·2순위): 043 nationality → 한국어 클래스명
NAT2K = {
    "Iran, Islamic Republic Of": "이란", "Pakistan": "파키스탄", "India": "인도",
    "Kyrgyzstan": "키르기스스탄", "Tajikistan": "타지키스탄", "Bangladesh": "방글라데시",
    "Singapore": "싱가포르", "Philippines": "필리핀",
    "Uzbekistan": "우즈베키스탄", "Kazakhstan": "카자흐스탄", "Nepal": "네팔",
    "Malaysia": "말레이시아", "Cambodia": "캄보디아", "Germany": "독일",
}

LABEL_DIRS = [os.path.join(BASE, "Training/02.라벨링데이터/TL_Speech_lab"),
              os.path.join(BASE, "Validation/02.라벨링데이터/VL_Speech_lab")]
ZIPS = [os.path.join(BASE, "Training/01.원천데이터/TS_Speech_sound_1.zip"),
        os.path.join(BASE, "Training/01.원천데이터/TS_Speech_sound_2.zip"),
        os.path.join(BASE, "Training/01.원천데이터/TS_Speech_sound_3.zip"),
        os.path.join(BASE, "Validation/01.원천데이터/VS_Speech_sound.zip")]

CSV_FIELDS = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]


def main():
    os.makedirs(OUT_AUDIO, exist_ok=True)

    # --- 1) Speech 라벨 전수: 선별 국적 화자의 발화 메타 수집 ---
    meta = {}                       # wav_basename(stem) -> row dict
    uid_nat = {}
    kept_u, kept_s = Counter(), defaultdict(set)
    drop_short = 0
    for ld in LABEL_DIRS:
        for r, _, fs in os.walk(ld):
            for f in fs:
                if not f.endswith(".json"):
                    continue
                uid = f.split("-")[0]
                jp = os.path.join(r, f)
                if uid not in uid_nat:
                    try:
                        uid_nat[uid] = json.load(open(jp, encoding="utf-8"))["SpeakerMetadata"]["nationality"]
                    except Exception:
                        uid_nat[uid] = "?"
                k = NAT2K.get(uid_nat[uid])
                if not k:
                    continue
                try:
                    d = json.load(open(jp, encoding="utf-8"))
                    rm = d["RecordingMetadata"]
                    words = len((rm.get("orthographic") or "").split())
                    dur = float(rm.get("RecordedTime", 0) or 0)
                except Exception:
                    continue
                if words < MIN_WORDS:
                    drop_short += 1
                    continue
                stem = f[:-5]
                meta[stem] = {"json_path": jp, "speaker_id": "AS_" + uid,
                              "label_name": k, "duration_sec": dur}
                kept_u[k] += 1
                kept_s[k].add(uid)
    print(f"[label] 선별화자 문장(≥{MIN_WORDS}단어) {len(meta):,}발화 "
          f"(짧은발화 제외 {drop_short:,}) / 화자 {len(set().union(*kept_s.values()))}명", flush=True)

    # --- 2) zip에서 해당 wav만 선별 추출 ---
    n_ext, n_skip = 0, 0
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as z:
            members = [n for n in z.namelist()
                       if os.path.basename(n)[:-4] in meta and n.endswith(".wav")]
            for n in members:
                dst = os.path.join(OUT_AUDIO, os.path.basename(n))
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    n_skip += 1
                else:
                    with z.open(n) as src, open(dst, "wb") as out:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                    n_ext += 1
        print(f"[unzip] {os.path.basename(zp)}: 대상 {len(members):,} (누적 추출 {n_ext:,}, 기존 {n_skip:,})", flush=True)

    # --- 3) 매니페스트 (오디오가 실제 추출된 것만) ---
    classes = sorted(set(NAT2K.values()))
    c2i = {c: i for i, c in enumerate(classes)}
    rows, no_audio = [], 0
    for stem, m in sorted(meta.items()):
        ap = os.path.join(OUT_AUDIO, stem + ".wav")
        if not os.path.exists(ap):
            no_audio += 1
            continue
        rows.append({"audio_path": ap, "json_path": m["json_path"], "speaker_id": m["speaker_id"],
                     "label_name": m["label_name"], "label_id": c2i[m["label_name"]],
                     "duration_sec": m["duration_sec"]})
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[manifest] {len(rows):,} rows (라벨은 있는데 오디오 없음 {no_audio}) -> {OUT_CSV}")
    fin_u, fin_s = Counter(), defaultdict(set)
    for r in rows:
        fin_u[r["label_name"]] += 1
        fin_s[r["label_name"]].add(r["speaker_id"])
    print(f"{'클래스':10}{'발화':>8}{'화자':>6}")
    for k, c in fin_u.most_common():
        print(f"{k:10}{c:>8,}{len(fin_s[k]):>6}")
    missing = [k for k in classes if k not in fin_u]
    if missing:
        print(f"(오디오/문장 없어 빠진 클래스: {missing})")


if __name__ == "__main__":
    main()

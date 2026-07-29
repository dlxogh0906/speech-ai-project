# -*- coding: utf-8 -*-
"""043 발음과제 zip에서 선별 클래스 화자의 문장(prompt ≥4단어) wav만 추출.

존재하는 zip만 처리(재다운로드 후 재실행하면 이어서 추출됨). 이미 추출된 파일은 skip.
출력: selected_pron/ (문장과제 selected_speech/ 와 분리)
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
from shared.paths import DATA_ROOT

import json
import os
import zipfile
from collections import Counter

BASE = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
OUT_AUDIO = f"{DATA_ROOT}/aihub/asian_korean_speech/selected_pron"
MIN_WORDS = 4

NAT2K = {
    "Iran, Islamic Republic Of": "이란", "Pakistan": "파키스탄", "India": "인도",
    "Kyrgyzstan": "키르기스스탄", "Tajikistan": "타지키스탄", "Bangladesh": "방글라데시",
    "Singapore": "싱가포르", "Philippines": "필리핀",
    "Uzbekistan": "우즈베키스탄", "Kazakhstan": "카자흐스탄", "Nepal": "네팔",
    "Malaysia": "말레이시아", "Cambodia": "캄보디아", "Germany": "독일",
}
LABEL_DIRS = [os.path.join(BASE, "Training/02.라벨링데이터/TL_pronunciation_lab"),
              os.path.join(BASE, "Validation/02.라벨링데이터/VL_pronunciation_lab")]
ZIPS = [os.path.join(BASE, "Training/01.원천데이터/TS_pronunciation_sound_1.zip"),
        os.path.join(BASE, "Training/01.원천데이터/TS_pronunciation_sound_2.zip"),
        os.path.join(BASE, "Validation/01.원천데이터/VS_pronunciation_sound.zip")]


def main():
    os.makedirs(OUT_AUDIO, exist_ok=True)
    # 대상 stem 수집
    uid_nat = {}
    stems = {}
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
                    if len((d["RecordingMetadata"].get("prompt") or "").split()) < MIN_WORDS:
                        continue
                except Exception:
                    continue
                stems[f[:-5]] = k
    print(f"[label] 대상 stem {len(stems):,}개", flush=True)

    got = Counter()
    n_have = sum(1 for s in stems if os.path.exists(os.path.join(OUT_AUDIO, s + ".wav")))
    print(f"[resume] 기존 추출 {n_have:,}", flush=True)
    for zp in ZIPS:
        if not os.path.exists(zp):
            print(f"[skip] 없음: {os.path.basename(zp)}", flush=True)
            continue
        try:
            z = zipfile.ZipFile(zp)
        except Exception as e:
            print(f"[skip] 손상: {os.path.basename(zp)} ({e})", flush=True)
            continue
        members = [n for n in z.namelist()
                   if n.endswith(".wav") and os.path.basename(n)[:-4] in stems]
        n_new = 0
        for n in members:
            dst = os.path.join(OUT_AUDIO, os.path.basename(n))
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue
            with z.open(n) as src, open(dst, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            n_new += 1
            got[stems[os.path.basename(n)[:-4]]] += 1
        z.close()
        print(f"[unzip] {os.path.basename(zp)}: 대상 {len(members):,}, 신규 {n_new:,}", flush=True)

    have = [s for s in stems if os.path.exists(os.path.join(OUT_AUDIO, s + ".wav"))]
    print(f"\n[status] 확보 {len(have):,} / 대상 {len(stems):,} "
          f"(미확보 {len(stems)-len(have):,} — 잘린/미다운 zip 재확보 후 재실행)")
    cls = Counter(stems[s] for s in have)
    for k, c in cls.most_common():
        print(f"  {k:10} {c:>6,}")


if __name__ == "__main__":
    main()

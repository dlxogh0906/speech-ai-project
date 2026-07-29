# -*- coding: utf-8 -*-
"""001(원어민) 라벨 zip에서, 보유 오디오(600화자)에 매칭되는 전사 JSON만 선별 추출.

- zip 파일명은 CP949(무 UTF-8 플래그) → cp437 decode 후 cp949 encode로 복원.
- 보유 001 wav basename stem과 일치하는 .json만 평면 저장.
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

import os, glob, zipfile, json

AUDIO_ROOT = f"{DATA_ROOT}/aihub/001.자유대화(일반남여)"
LABEL_ZIPS = f"{DATA_ROOT}/aihub/001_labels"
OUT = f"{DATA_ROOT}/aihub/001_labels/extracted_matched"


def fix_name(n):
    """CP949 mojibake 복원."""
    try:
        return n.encode("cp437").decode("cp949")
    except Exception:
        return n


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) 보유 001 오디오 basename stem
    have = {}
    for r, _, fs in os.walk(AUDIO_ROOT):
        for f in fs:
            if f.endswith(".wav"):
                have[f[:-4]] = os.path.join(r, f)
    print(f"[audio] 보유 001 wav {len(have):,}", flush=True)

    # 2) 라벨 zip에서 매칭 json만 추출 (cp949 복원)
    got = 0
    already = {f[:-5] for f in os.listdir(OUT) if f.endswith(".json")}
    for zp in sorted(glob.glob(os.path.join(LABEL_ZIPS, "**", "*.zip"), recursive=True)):
        z = zipfile.ZipFile(zp)
        n_hit = 0
        for info in z.infolist():
            if info.is_dir():
                continue
            real = fix_name(info.filename)
            if not real.endswith(".json"):
                continue
            stem = os.path.basename(real)[:-5]
            if stem not in have or stem in already:
                continue
            data = z.read(info)
            with open(os.path.join(OUT, stem + ".json"), "wb") as f:
                f.write(data)
            got += 1
            n_hit += 1
        z.close()
        print(f"[zip] {os.path.basename(zp)}: 매칭추출 {n_hit:,} (누적 {got:,})", flush=True)

    # 3) 검증: 전사(stt) 필드 확인
    files = [f for f in os.listdir(OUT) if f.endswith(".json")]
    print(f"\n[done] 매칭 전사 JSON {len(files):,} / 보유오디오 {len(have):,}")
    if files:
        d = json.load(open(os.path.join(OUT, files[0]), encoding="utf-8"))
        stt = d.get("발화정보", {}).get("stt")
        print(f"  샘플 stt: {stt!r}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""043 최종 매니페스트: 선별 14클래스 × prompt ≥4단어 (041 sent_mask와 동일 기준).

- 문장과제(Speech) + 발음과제(pronunciation) 둘 다 포함, 기준은 prompt 단어수로 통일
  (041 마스크가 prompt 기준이었음 — 발음과제 라벨엔 orthographic이 없어서).
- 오디오는 selected_speech/(문장과제) + selected_pron/(발음과제)에서 찾고,
  없으면 남아있는 zip에서 선별 추출로 보충 (멱등 — 재실행 안전).
- 출력: $NATIONALITY_DATA/asian_country40_manifest.csv (덮어씀)
        + $NATIONALITY_DATA/asian_country40/train.csv (feature 추출용 복사)
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
import shutil
import zipfile
from collections import Counter, defaultdict

BASE = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
AUD = {"speech": f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech",
       "pron": f"{DATA_ROOT}/aihub/asian_korean_speech/selected_pron"}
OUT_CSV = f"{NAT_DATA}/asian_country40_manifest.csv"
OUT_DIR = f"{NAT_DATA}/asian_country40"
MIN_WORDS = 4

NAT2K = {
    "Iran, Islamic Republic Of": "이란", "Pakistan": "파키스탄", "India": "인도",
    "Kyrgyzstan": "키르기스스탄", "Tajikistan": "타지키스탄", "Bangladesh": "방글라데시",
    "Singapore": "싱가포르", "Philippines": "필리핀",
    "Uzbekistan": "우즈베키스탄", "Kazakhstan": "카자흐스탄", "Nepal": "네팔",
    "Malaysia": "말레이시아", "Cambodia": "캄보디아", "Germany": "독일",
}
LABELS = [("Training/02.라벨링데이터/TL_Speech_lab", "speech"),
          ("Validation/02.라벨링데이터/VL_Speech_lab", "speech"),
          ("Training/02.라벨링데이터/TL_pronunciation_lab", "pron"),
          ("Validation/02.라벨링데이터/VL_pronunciation_lab", "pron")]
ZIPS = {"speech": ["Training/01.원천데이터/TS_Speech_sound_1.zip",
                   "Training/01.원천데이터/TS_Speech_sound_2.zip",
                   "Training/01.원천데이터/TS_Speech_sound_3.zip",
                   "Validation/01.원천데이터/VS_Speech_sound.zip"],
        "pron": ["Training/01.원천데이터/TS_pronunciation_sound_1.zip",
                 "Training/01.원천데이터/TS_pronunciation_sound_2.zip",
                 "Validation/01.원천데이터/VS_pronunciation_sound.zip"]}
CSV_FIELDS = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]


def main():
    for d in AUD.values():
        os.makedirs(d, exist_ok=True)

    # --- 1) 라벨 전수: prompt ≥4단어 대상 수집 ---
    uid_nat = {}
    meta = {}                        # stem -> dict(task, json_path, spk, label, dur)
    for sub, task in LABELS:
        root = os.path.join(BASE, sub)
        for r, _, fs in os.walk(root):
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
                    if len((rm.get("prompt") or "").split()) < MIN_WORDS:
                        continue
                    dur = float(rm.get("RecordedTime", 0) or 0)
                except Exception:
                    continue
                meta[f[:-5]] = {"task": task, "json_path": jp, "speaker_id": "AS_" + uid,
                                "label_name": k, "duration_sec": dur}
    n_task = Counter(m["task"] for m in meta.values())
    print(f"[label] prompt>={MIN_WORDS}단어 대상 {len(meta):,} (speech {n_task['speech']:,} / pron {n_task['pron']:,})", flush=True)

    # --- 2) 부족한 wav를 zip에서 보충 (있는 zip만, resume) ---
    def have(stem):
        p = os.path.join(AUD[meta[stem]["task"]], stem + ".wav")
        return os.path.exists(p) and os.path.getsize(p) > 0
    missing = {s for s in meta if not have(s)}
    print(f"[audio] 기보유 {len(meta)-len(missing):,} / 부족 {len(missing):,}", flush=True)
    for task, zlist in ZIPS.items():
        want = {s for s in missing if meta[s]["task"] == task}
        if not want:
            continue
        for zrel in zlist:
            zp = os.path.join(BASE, zrel)
            if not os.path.exists(zp):
                continue
            try:
                z = zipfile.ZipFile(zp)
            except Exception as e:
                print(f"[zip skip] {os.path.basename(zp)}: {e}", flush=True)
                continue
            members = [n for n in z.namelist()
                       if n.endswith(".wav") and os.path.basename(n)[:-4] in want]
            n_new = 0
            for n in members:
                stem = os.path.basename(n)[:-4]
                dst = os.path.join(AUD[task], stem + ".wav")
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    continue
                with z.open(n) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                n_new += 1
            z.close()
            if members:
                print(f"[unzip] {os.path.basename(zp)}: +{n_new:,}", flush=True)

    # --- 3) 매니페스트 (오디오 존재분만) ---
    classes = sorted(set(NAT2K.values()))
    c2i = {c: i for i, c in enumerate(classes)}
    rows, no_audio = [], Counter()
    for stem, m in sorted(meta.items()):
        ap = os.path.join(AUD[m["task"]], stem + ".wav")
        if not (os.path.exists(ap) and os.path.getsize(ap) > 0):
            no_audio[m["task"]] += 1
            continue
        rows.append({"audio_path": ap, "json_path": m["json_path"], "speaker_id": m["speaker_id"],
                     "label_name": m["label_name"], "label_id": c2i[m["label_name"]],
                     "duration_sec": m["duration_sec"]})
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copy(OUT_CSV, os.path.join(OUT_DIR, "train.csv"))

    print(f"\n[manifest] {len(rows):,} rows -> {OUT_CSV} (+ {OUT_DIR}/train.csv)")
    if sum(no_audio.values()):
        print(f"  오디오 미확보로 제외: {dict(no_audio)} — zip 재확보 후 재실행하면 채워짐")
    fu, fs, ft = Counter(), defaultdict(set), Counter()
    for r in rows:
        fu[r["label_name"]] += 1
        fs[r["label_name"]].add(r["speaker_id"])
        ft["speech" if "/selected_speech/" in r["audio_path"] else "pron"] += 1
    print(f"  과제 구성: {dict(ft)}")
    print(f"{'클래스':10}{'발화':>8}{'화자':>6}")
    for k, c in fu.most_common():
        print(f"{k:10}{c:>8,}{len(fs[k]):>6}")
    miss_cls = [k for k in classes if k not in fu]
    if miss_cls:
        print(f"(빠진 클래스: {miss_cls})")


if __name__ == "__main__":
    main()

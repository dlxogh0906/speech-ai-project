"""61-class 외국어 매니페스트 + 한국어 원어민 매니페스트 → 62-class 합치기.

출력: taeho/manifest_62class_ft/  (train.csv, valid.csv, test.csv, labels.txt, label2id.json)
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
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS

import csv
import json
from pathlib import Path

FOREIGN_DIR = Path(f"{NAT_DATA}/aihub_nationality_fine_811")
KOREAN_DIR  = Path(f"{DATA_ROOT}/taeho/korean_native_manifest_seed42")
OUT_DIR     = Path(f"{NAT_OUTPUTS}/manifest_62class_ft")
KOREAN_LABEL = "한국어_모국어"

FIELDS = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. labels.txt: 61개 외국어 + 한국어_모국어
foreign_labels = [l.strip() for l in (FOREIGN_DIR / "labels.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(foreign_labels) == 61, f"expected 61 foreign labels, got {len(foreign_labels)}"
all_labels = foreign_labels + [KOREAN_LABEL]  # 62-class, Korean = label_id 61
korean_id = len(foreign_labels)  # 61

(OUT_DIR / "labels.txt").write_text("\n".join(all_labels) + "\n", encoding="utf-8")
label2id = {l: i for i, l in enumerate(all_labels)}
(OUT_DIR / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"labels: {len(all_labels)} classes (Korean label_id={korean_id})")

# 2. 각 split 합치기
for split in ["train", "valid", "test"]:
    foreign_csv = FOREIGN_DIR / f"{split}.csv"
    korean_csv  = KOREAN_DIR  / f"{split}.csv"
    out_csv     = OUT_DIR     / f"{split}.csv"

    rows = []

    # 외국어 rows
    for r in csv.DictReader(foreign_csv.open(encoding="utf-8")):
        rows.append({
            "audio_path":  r["audio_path"],
            "json_path":   r.get("json_path", ""),
            "speaker_id":  r["speaker_id"],
            "label_name":  r["label_name"],
            "label_id":    r["label_id"],
            "duration_sec": r.get("duration_sec", ""),
        })

    # 한국어 rows (label_id 재설정)
    for r in csv.DictReader(korean_csv.open(encoding="utf-8")):
        rows.append({
            "audio_path":  r["audio_path"],
            "json_path":   r.get("json_path", ""),
            "speaker_id":  r["speaker_id"],
            "label_name":  KOREAN_LABEL,
            "label_id":    str(korean_id),
            "duration_sec": r.get("duration_sec", ""),
        })

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_foreign = sum(1 for r in rows if r["label_name"] != KOREAN_LABEL)
    n_korean  = len(rows) - n_foreign
    print(f"{split}: {len(rows):,} rows  (foreign {n_foreign:,} + korean {n_korean:,})")

print(f"\n완료 → {OUT_DIR}")

"""Build a 41-class COUNTRY manifest dir (labels.txt + train/valid/test.csv) for fine-tuning.

Foreign rows: remap fine_811's mother-tongue -> country via LANG2C/SPLIT (speaker residence).
Korean rows: appended from korean_native_manifest_seed42, labeled '한국어'.
Class order = the deployed demo head's 41-class order (for compatibility).
Splits inherit each source's speaker-disjoint split (no leakage).
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

import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
from country_classifier.make_country40_probe import SPLIT, LANG2C

FINE = Path(f"{NAT_DATA}/aihub_nationality_fine_811")
KOR = Path(f"{DATA_ROOT}/taeho/korean_native_manifest_seed42")
OUT = Path(f"{NAT_DATA}/aihub_country41_811"); OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]

# canonical 41-class order from the deployed head
labels = list(torch.load(f"{NAT_OUTPUTS}/0629_country41_12s/head.pt",
                         map_location="cpu", weights_only=False)["labels"])
lab2id = {c: i for i, c in enumerate(labels)}
Path(OUT / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
spk_mt = {}; spk_cc = defaultdict(Counter)
for r in rows:
    spk_mt[r["speaker_id"]] = r["mother_tongue"]
    spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}


def c40(s):
    mt = spk_mt.get(s)
    return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)


for split in ["train", "valid", "test"]:
    out_rows = []
    dist = Counter()
    # foreign
    for r in csv.DictReader(open(FINE / f"{split}.csv", encoding="utf-8")):
        country = c40(r["speaker_id"])
        if country not in lab2id:
            continue
        out_rows.append({"audio_path": r["audio_path"], "json_path": r.get("json_path", ""),
                         "speaker_id": r["speaker_id"], "label_name": country,
                         "label_id": lab2id[country], "duration_sec": r.get("duration_sec", "")})
        dist[country] += 1
    # korean
    kpath = KOR / f"{split}.csv"
    if kpath.exists():
        for r in csv.DictReader(open(kpath, encoding="utf-8")):
            out_rows.append({"audio_path": r["audio_path"], "json_path": r.get("json_path", ""),
                             "speaker_id": r["speaker_id"], "label_name": "한국어",
                             "label_id": lab2id["한국어"], "duration_sec": r.get("duration_sec", "")})
            dist["한국어"] += 1
    with open(OUT / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out_rows)
    print(f"[{split}] {len(out_rows):,} rows | {len(dist)} classes | top: "
          + ", ".join(f"{k}={v}" for k, v in dist.most_common(5)))

print(f"-> {OUT} (labels.txt + train/valid/test.csv, {len(labels)} classes)")

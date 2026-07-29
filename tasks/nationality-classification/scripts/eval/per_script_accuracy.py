"""Which reading script does the model classify most accurately? (GPU-free)

Uses cached test features + the trained head (no encoder run) for predictions, then maps
each test utterance to its ReadingLabelText (only parses test JSONs -> fast). Reports
per-script top-1/top-5 over enough utterances+nationalities, to recommend deployment
sentences empirically.
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
from shared.paths import NAT_DATA, NAT_RUNS


import csv
import glob
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from country_classifier import baseline_common as bc

FEAT = Path(f"{NAT_RUNS}/features_fine_811")
MAN = Path(f"{NAT_DATA}/aihub_nationality_fine_811")


def main():
    id2label = json.loads((FEAT / "id2label.json").read_text())
    labels = [id2label[str(i)] for i in range(len(id2label))]
    d = np.load(FEAT / "test.npz", allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(np.int64)
    rows = list(csv.DictReader(open(MAN / "test.csv", encoding="utf-8")))
    base = [Path(r["audio_path"]).name for r in rows]
    assert len(base) == len(y)

    # predictions from cached head (no encoder)
    ck = torch.load(f"{NAT_RUNS}/probe_fine_811/head.pt", map_location="cpu")
    head = nn.Sequential(nn.Dropout(0.0), nn.Linear(ck["mu"].shape[1], len(labels)))
    head.load_state_dict(ck["head"]); head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy((X - ck["mu"]) / ck["sd"])).numpy()
    top1 = logits.argmax(1)
    top5 = np.argpartition(-logits, 4, axis=1)[:, :5]
    hit1 = top1 == y
    hit5 = np.array([yi in r for yi, r in zip(y, top5)])

    # map test basenames -> reading script text (parse only test JSONs)
    stem2idx = {Path(b).stem: i for i, b in enumerate(base)}
    test_stems = set(stem2idx)
    reading = {}
    n_parsed = 0
    for z in glob.glob(f"{bc.DEFAULT_AUDIO_ROOT}/*/라벨링데이터/*.zip"):
        try:
            zf = zipfile.ZipFile(z)
        except Exception:
            continue
        for info in zf.infolist():
            if not info.filename.endswith(".json"):
                continue
            stem = Path(info.filename).stem
            if stem not in test_stems:
                continue
            try:
                jd = json.loads(zf.read(info).decode("utf-8-sig"))
            except Exception:
                continue
            rd = (jd.get("transcription", {}).get("ReadingLabelText") or "").strip()
            if rd:
                reading[stem2idx[stem]] = rd
            n_parsed += 1
        zf.close()
    print(f"parsed {n_parsed} test JSONs; reading-type utts: {len(reading)}", flush=True)

    # overall reading vs answer accuracy
    read_idx = np.array(sorted(reading))
    is_read = np.zeros(len(y), bool); is_read[read_idx] = True
    print(f"\nreading utts: top1={hit1[is_read].mean():.3f} top5={hit5[is_read].mean():.3f}")
    print(f"answer  utts: top1={hit1[~is_read].mean():.3f} top5={hit5[~is_read].mean():.3f}")

    # per-script aggregation
    by = defaultdict(list)
    for i, txt in reading.items():
        by[txt].append(i)
    scripts = []
    for txt, idxs in by.items():
        idxs = np.array(idxs)
        nat = set(y[idxs].tolist())
        if len(idxs) >= 20 and len(nat) >= 5:
            # macro: per-nationality top5 averaged (controls for which nationalities present)
            macro5 = np.mean([hit5[idxs][y[idxs] == c].mean() for c in nat])
            scripts.append({"text": txt, "n": int(len(idxs)), "nat": len(nat),
                            "top1": float(hit1[idxs].mean()), "top5": float(hit5[idxs].mean()),
                            "macro5": float(macro5)})
    scripts.sort(key=lambda s: -s["macro5"])
    print(f"\n=== reading scripts (>=20 utts, >=5 nationalities): {len(scripts)} ===")
    print(f"{'macro5':>7}{'top5':>7}{'top1':>7}{'n':>6}{'nat':>5}  script")
    for s in scripts[:15]:
        print(f"{s['macro5']:>7.3f}{s['top5']:>7.3f}{s['top1']:>7.3f}{s['n']:>6}{s['nat']:>5}  {s['text'][:55]}")
    Path(f"{NAT_RUNS}/per_script_accuracy.json").write_text(
        json.dumps(scripts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {len(scripts)} scripts -> $NATIONALITY_RUNS/per_script_accuracy.json")


if __name__ == "__main__":
    main()

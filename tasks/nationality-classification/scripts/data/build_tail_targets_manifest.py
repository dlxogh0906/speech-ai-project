#!/usr/bin/env python
"""Step 1: build a TRAIN-ONLY tail-class manifest aligned to the target_counts split.

Why not a plain `df[df.label_name.isin(targets)]` on a train.csv?  Because the only
CSVs with audio_path (manifest_62class_ft/*) use the ORIGINAL fine_811 split, whose
train speakers differ from the target_counts resplit used by the decoupling probe.
Augmenting an original-split-train speaker who is a target_counts VALID/TEST speaker
would leak across the speaker-disjoint boundary.

So we take ALL foreign utterances (union of manifest_62class_ft train+valid+test) and
keep a row only if BOTH:
  - label_name is a target tail class, AND
  - speaker_id is assigned to "train" by
    feature_resplit_61_merged6_seed42_target_counts/speaker_assignment.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_DATA, NAT_OUTPUTS

import argparse
import csv
import json
import os
import random

DEFAULT_TARGETS = ["인도네시아어", "광동어", "말레이어", "스페인어", "타갈로그어"]
SRC_CSVS = [
    f"{NAT_OUTPUTS}/manifest_62class_ft/train.csv",
    f"{NAT_OUTPUTS}/manifest_62class_ft/valid.csv",
    f"{NAT_OUTPUTS}/manifest_62class_ft/test.csv",
]
SPK_ASSIGN = f"{NAT_OUTPUTS}/feature_resplit_61_merged6_seed42_target_counts/speaker_assignment.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="comma-separated label_name list")
    ap.add_argument("--out_csv", default=f"{NAT_DATA}/train_tail_targets.csv")
    ap.add_argument("--spk_assign", default=SPK_ASSIGN)
    ap.add_argument("--src_csvs", nargs="+", default=SRC_CSVS)
    args = ap.parse_args()

    targets = set(args.targets.split(","))
    spk_split = json.load(open(args.spk_assign, encoding="utf-8"))  # speaker -> split

    fieldnames = None
    seen = set()
    rows = []
    per_utts = {}
    per_spk = {lab: set() for lab in targets}
    for src in args.src_csvs:
        with open(src, newline="") as f:
            rd = csv.DictReader(f)
            fieldnames = rd.fieldnames
            for r in rd:
                lab = r["label_name"]
                if lab not in targets:
                    continue
                spk = r["speaker_id"]
                if spk_split.get(spk) != "train":   # target_counts TRAIN only
                    continue
                key = r["audio_path"]
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
                per_utts[lab] = per_utts.get(lab, 0) + 1
                per_spk[lab].add(spk)

    # shuffle so a per-class augmentation cap draws from ALL speakers, not just
    # whichever ones sort first (the goal is speaker diversity).
    random.seed(42)
    random.shuffle(rows)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[done] {len(rows)} TRAIN rows over {len(per_utts)} classes -> {args.out_csv}")
    print(f"{'label':<12} {'utts':>7} {'train_spk':>10}")
    for lab in sorted(per_utts, key=lambda k: -per_utts[k]):
        print(f"{lab:<12} {per_utts[lab]:>7} {len(per_spk[lab]):>10}")
    missing = targets - set(per_utts)
    if missing:
        print("[warn] no rows for:", ", ".join(missing))


if __name__ == "__main__":
    main()

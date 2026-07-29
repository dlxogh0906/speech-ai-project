"""Fine-grained (motherTongue, 61-class) speaker-disjoint train/valid/test split.

Uses data/fine_labels.csv (basename, speaker_id, mother_tongue) + the basename audio
index. Speakers are assigned whole to one split, stratified per fine label, balancing
utterance counts toward the target ratios (default 8:1:1). Writes CSV manifests that
AudioCountryDataset reads directly, plus labels.txt / label2id.json / split_stats.json.

    python scripts/make_fine_split.py --out_dir $NATIONALITY_DATA/aihub_nationality_fine_811
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_DATA


import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from country_classifier import baseline_common as bc
from country_classifier.make_speaker_split import assign_speakers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label_csv", default=f"{NAT_DATA}/fine_labels.csv")
    ap.add_argument("--out_dir", default=f"{NAT_DATA}/aihub_nationality_fine_811")
    ap.add_argument("--audio_root", default=bc.DEFAULT_AUDIO_ROOT)
    ap.add_argument("--ratios", type=float, nargs=3, default=[8, 1, 1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building audio basename index ...")
    audio_index = bc.build_audio_index(args.audio_root)
    print(f"  indexed {len(audio_index):,} wavs")

    # load fine labels
    rows = []
    label_count = Counter()
    with open(args.label_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bn = r["basename"]
            path = audio_index.get(bn)
            if path is None:
                continue
            rows.append({"audio_path": path, "speaker_id": r["speaker_id"],
                         "label_name": r["mother_tongue"], "num_frames": 0})
            label_count[r["mother_tongue"]] += 1
    print(f"matched {len(rows):,} utts to audio; {len(label_count)} classes")

    # label ids: most frequent first (stable, frequency-descending; tie-break by name)
    labels = [l for l, _ in sorted(label_count.items(), key=lambda kv: (-kv[1], kv[0]))]
    label2id = {l: i for i, l in enumerate(labels)}

    assignment = assign_speakers(rows, args.ratios, args.seed)

    writers, files = {}, {}
    for split in ["train", "valid", "test"]:
        fh = (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="")
        files[split] = fh
        w = csv.DictWriter(fh, fieldnames=bc.CSV_FIELDS)
        w.writeheader()
        writers[split] = w

    su = Counter(); sl = defaultdict(Counter); ss = defaultdict(set)
    for r in rows:
        split = assignment[r["speaker_id"]]
        writers[split].writerow({
            "audio_path": r["audio_path"], "json_path": "",
            "speaker_id": r["speaker_id"], "label_name": r["label_name"],
            "label_id": label2id[r["label_name"]], "duration_sec": "",
        })
        su[split] += 1; sl[split][r["label_name"]] += 1; ss[split].add(r["speaker_id"])
    for fh in files.values():
        fh.close()

    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    overlaps = {f"{a}_vs_{b}": len(ss[a] & ss[b]) for a, b in pairs}
    assert all(v == 0 for v in overlaps.values()), f"speaker leakage! {overlaps}"

    # how many classes actually appear in every split (learnable+evaluable)
    in_all = sum(1 for l in labels if all(sl[s].get(l, 0) > 0 for s in ["train", "valid", "test"]))
    missing_test = [l for l in labels if sl["test"].get(l, 0) == 0]

    total = sum(su.values())
    (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (out_dir / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    stats = {
        "num_classes": len(labels), "ratios": args.ratios, "seed": args.seed,
        "total_utts": total,
        "splits": {s: {"utts": su[s], "frac": round(su[s] / total, 4),
                       "speakers": len(ss[s]), "classes_present": len(sl[s])}
                   for s in ["train", "valid", "test"]},
        "speaker_overlap": overlaps,
        "classes_in_all_splits": in_all,
        "classes_absent_from_test": missing_test,
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    print(f"\n=== fine split ({len(labels)} classes) -> {out_dir} ===")
    for s in ["train", "valid", "test"]:
        d = stats["splits"][s]
        print(f"  {s:5s} utts={d['utts']:>9,} ({100*d['frac']:.1f}%) spk={d['speakers']:>5,} "
              f"classes_present={d['classes_present']}/{len(labels)}")
    print(f"  speaker_overlap={overlaps}")
    print(f"  classes present in ALL splits: {in_all}/{len(labels)}")
    print(f"  classes absent from test ({len(missing_test)}): {missing_test}")


if __name__ == "__main__":
    main()

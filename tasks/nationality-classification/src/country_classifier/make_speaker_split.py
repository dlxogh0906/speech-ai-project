"""Create a speaker-disjoint, label-stratified train/valid/test split (default 8:1:1).

Each speaker (and all of their utterances) is assigned to exactly one split, so there
is no speaker leakage across train/valid/test. Allocation is done per label to keep the
nationality distribution stratified, greedily balancing *utterance* counts toward the
target ratios.

Source = the union of the existing `train_full_valid_test` manifest's three splits,
which together partition the whole extracted corpus (1,042,269 utts / 1,910 speakers).
Output = CSV manifests (train.csv/valid.csv/test.csv) that AudioCountryDataset reads
directly, plus split_stats.json. Audio paths are resolved via the basename index.

    python scripts/make_speaker_split.py \
        --out_dir $NATIONALITY_DATA/aihub_korean_nationality_speaker_811 \
        --ratios 8 1 1 --seed 42
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_DATA


import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from country_classifier import baseline_common as bc


def build_master(source_set, audio_root, audio_index):
    """Collect unique utterances: list of dicts {audio_path, speaker_id, label, frames}."""
    md = f"{NAT_DATA}/{source_set}"
    seen = set()
    rows = []
    for split in ["train", "valid", "test"]:
        try:
            _, entries = bc.read_fairseq_split(md, split, audio_root=audio_root,
                                               audio_index=audio_index)
        except FileNotFoundError:
            continue
        for e in entries:
            bn = Path(e["audio_path"]).name
            if bn in seen:
                continue
            seen.add(bn)
            rows.append({
                "audio_path": e["audio_path"],
                "speaker_id": e["speaker_id"],
                "label_name": e["label_name"],
                "num_frames": e.get("num_frames", 0),
            })
    return rows


def assign_speakers(rows, ratios, seed):
    """Return {speaker_id: split}. Per-label greedy balancing on utterance counts."""
    frac = [r / sum(ratios) for r in ratios]
    names = ["train", "valid", "test"]
    by_label_spk = defaultdict(lambda: defaultdict(int))  # label -> spk -> utt count
    for r in rows:
        by_label_spk[r["label_name"]][r["speaker_id"]] += 1

    rng = random.Random(seed)
    assignment = {}
    for label, spk_counts in by_label_spk.items():
        total = sum(spk_counts.values())
        targets = [f * total for f in frac]
        current = [0, 0, 0]
        # Largest speakers first so big speakers don't overshoot a small split late.
        speakers = sorted(spk_counts.items(), key=lambda kv: kv[1], reverse=True)
        rng.shuffle(speakers)  # tie-break / avoid deterministic-by-size artifacts
        speakers.sort(key=lambda kv: kv[1], reverse=True)
        for spk, cnt in speakers:
            # assign to the split with the largest remaining deficit (target - current)
            deficits = [targets[i] - current[i] for i in range(3)]
            j = max(range(3), key=lambda i: deficits[i])
            assignment[spk] = names[j]
            current[j] += cnt
    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_set", default="aihub_korean_nationality_train_full_valid_test")
    ap.add_argument("--out_dir", default=f"{NAT_DATA}/aihub_korean_nationality_speaker_811")
    ap.add_argument("--audio_root", default=bc.DEFAULT_AUDIO_ROOT)
    ap.add_argument("--ratios", type=float, nargs=3, default=[8, 1, 1],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_rate", type=int, default=16000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building audio basename index ...")
    audio_index = bc.build_audio_index(args.audio_root)
    print(f"  indexed {len(audio_index):,} wavs")

    rows = build_master(args.source_set, args.audio_root, audio_index)
    print(f"master: {len(rows):,} utts")

    labels = bc.load_labels(f"{NAT_DATA}/{args.source_set}")
    label2id = {l: i for i, l in enumerate(labels)}

    assignment = assign_speakers(rows, args.ratios, args.seed)

    # write CSVs + collect stats
    writers = {}
    files = {}
    for split in ["train", "valid", "test"]:
        f = (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="")
        files[split] = f
        w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS)
        w.writeheader()
        writers[split] = w

    split_utts = Counter()
    split_label = defaultdict(Counter)
    split_spk = defaultdict(set)
    for r in rows:
        split = assignment[r["speaker_id"]]
        frames = r["num_frames"]
        writers[split].writerow({
            "audio_path": r["audio_path"],
            "json_path": "",
            "speaker_id": r["speaker_id"],
            "label_name": r["label_name"],
            "label_id": label2id[r["label_name"]],
            "duration_sec": (frames / args.sample_rate) if frames else "",
        })
        split_utts[split] += 1
        split_label[split][r["label_name"]] += 1
        split_spk[split].add(r["speaker_id"])
    for f in files.values():
        f.close()

    # speaker-disjointness assertion
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    overlaps = {f"{a}_vs_{b}": len(split_spk[a] & split_spk[b]) for a, b in pairs}
    assert all(v == 0 for v in overlaps.values()), f"speaker leakage! {overlaps}"

    total = sum(split_utts.values())
    stats = {
        "source_set": args.source_set,
        "ratios": args.ratios,
        "seed": args.seed,
        "total_utts": total,
        "total_speakers": sum(len(s) for s in split_spk.values()),
        "splits": {
            sp: {
                "utts": split_utts[sp],
                "utt_frac": round(split_utts[sp] / total, 4),
                "speakers": len(split_spk[sp]),
                "label_dist": dict(sorted(split_label[sp].items())),
            } for sp in ["train", "valid", "test"]
        },
        "speaker_overlap": overlaps,
        "labels": labels,
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                              encoding="utf-8")

    print(f"\n=== speaker-disjoint split -> {out_dir} ===")
    for sp in ["train", "valid", "test"]:
        s = stats["splits"][sp]
        print(f"  {sp:5s} utts={s['utts']:>9,} ({100*s['utt_frac']:.1f}%) "
              f"spk={s['speakers']:>5,}  {s['label_dist']}")
    print(f"  speaker_overlap={overlaps}  (all 0 = no leakage)")
    # also drop a labels.txt for convenience
    (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

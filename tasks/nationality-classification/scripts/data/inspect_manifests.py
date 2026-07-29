"""Inspect the fairseq-style manifests under $NATIONALITY_DATA.

For every manifest set it reports, per split: sample counts, presence of
tsv/.lang/.spk, the TSV root (and remapped local root), how many sampled audio
files actually exist, the label distribution, and speaker overlap across splits.
Results are written as JSON.

Usage:
    python scripts/inspect_manifests.py \
        --data_root $NATIONALITY_DATA \
        --output $NATIONALITY_RUNS/manifest_inspection.json \
        --audio_root $SPEECH_AI_DATA/aihub/foreign_korean_speech/131.../01.데이터_new_20220719
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_DATA, NAT_RUNS


import argparse
import json
from collections import Counter
from pathlib import Path

from country_classifier.baseline_common import (
    DEFAULT_AUDIO_ROOT,
    manifest_paths,
    read_fairseq_split,
    resolve_real_root,
)

SPLITS = ["train", "valid", "test"]


def inspect_split(manifest_dir, split, audio_root, sample, audio_index=None):
    paths = manifest_paths(manifest_dir, split)
    info = {
        "has_tsv": paths["tsv"].exists(),
        "has_lang": paths["lang"].exists(),
        "has_spk": paths["spk"].exists(),
    }
    if not info["has_tsv"]:
        info["present"] = False
        return info, set()
    info["present"] = True

    root, entries = read_fairseq_split(manifest_dir, split, audio_root=audio_root,
                                       audio_index=audio_index)
    info["num_samples"] = len(entries)
    info["tsv_root_local"] = str(root)
    info["tsv_root_exists"] = root.exists()

    # label distribution
    dist = Counter(e["label_name"] for e in entries)
    info["label_distribution"] = dict(sorted(dist.items()))

    # sampled audio existence (probe first `sample` + evenly spaced)
    if entries:
        step = max(1, len(entries) // sample)
        probe = entries[::step][:sample]
        present = sum(1 for e in probe if Path(e["audio_path"]).exists())
        info["audio_probe"] = {
            "checked": len(probe),
            "present": present,
            "example_path": probe[0]["audio_path"],
        }

    speakers = {e["speaker_id"] for e in entries if e["speaker_id"]}
    info["num_speakers"] = len(speakers)
    return info, speakers


def inspect_set(set_dir, audio_root, sample, audio_index=None):
    result = {"manifest_dir": str(set_dir), "splits": {}}
    spk_by_split = {}
    for split in SPLITS:
        try:
            info, speakers = inspect_split(set_dir, split, audio_root, sample,
                                           audio_index=audio_index)
        except Exception as exc:  # noqa: BLE001
            result["splits"][split] = {"present": False, "error": str(exc)}
            continue
        result["splits"][split] = info
        if info.get("present") and speakers:
            spk_by_split[split] = speakers

    overlaps = {}
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for a, b in pairs:
        if a in spk_by_split and b in spk_by_split:
            inter = spk_by_split[a] & spk_by_split[b]
            overlaps[f"{a}_vs_{b}"] = {
                "shared_speakers": len(inter),
                "examples": sorted(list(inter))[:5],
            }
    result["speaker_overlap"] = overlaps
    result["speaker_leakage"] = any(v["shared_speakers"] > 0 for v in overlaps.values())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=f"{NAT_DATA}")
    ap.add_argument("--output", default=f"{NAT_RUNS}/manifest_inspection.json")
    ap.add_argument("--audio_root", default=DEFAULT_AUDIO_ROOT,
                    help="local corpus base (set '' to use the manifest's own root)")
    ap.add_argument("--sample", type=int, default=20,
                    help="how many audio paths to probe per split")
    ap.add_argument("--no_audio_index", dest="audio_index", action="store_false")
    ap.set_defaults(audio_index=True)
    args = ap.parse_args()

    audio_index = None
    if args.audio_index and args.audio_root:
        from baseline_common import build_audio_index
        audio_index = build_audio_index(args.audio_root)
        print(f"(audio basename index: {len(audio_index):,} wavs)")

    data_root = Path(args.data_root)
    set_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if not set_dirs:
        raise SystemExit(f"no manifest sets under {data_root}")

    report = {
        "data_root": str(data_root),
        "audio_root": args.audio_root or "(manifest root)",
        "num_sets": len(set_dirs),
        "sets": {},
    }
    for set_dir in set_dirs:
        if not (set_dir / "train.tsv").exists():
            continue
        report["sets"][set_dir.name] = inspect_set(set_dir, args.audio_root, args.sample,
                                                   audio_index=audio_index)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # concise console summary
    print(f"\n=== manifest inspection -> {out} ===")
    print(f"audio_root: {report['audio_root']}\n")
    for name, s in report["sets"].items():
        print(f"[{name}]")
        for split in SPLITS:
            si = s["splits"].get(split, {})
            if not si.get("present"):
                continue
            probe = si.get("audio_probe", {})
            print(f"  {split:5s} n={si['num_samples']:>8d} spk={si.get('num_speakers',0):>5d} "
                  f"audio {probe.get('present','?')}/{probe.get('checked','?')} present  "
                  f"labels={si['label_distribution']}")
        leak = s["speaker_leakage"]
        print(f"  speaker_leakage={leak}  overlaps=" +
              ", ".join(f"{k}:{v['shared_speakers']}" for k, v in s["speaker_overlap"].items()))
        print()


if __name__ == "__main__":
    main()

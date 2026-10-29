"""Expand normal calibration tails and a distinct-prompt corpus evaluation.

Selection uses existing TAG labels and deterministic hashes, never scores.
Existing pilot UIDs are excluded. Earlier project-wide use is not ruled out.
This first expansion retains the established single-error/wholly-correct
subset; compound errors, deletions, and third-phone substitutions remain a
separate required validation, not silently re-labelled binary examples.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
import collections
import hashlib
import heapq
import json
import wave

from shared.paths import MDD_OUTPUTS, DATA_ROOT
from prepare_sibilant_rank_pilot import record, destination, balanced, priority, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    parser.add_argument("--exclude", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cal-source", choices=("valid", "test"), default="valid")
    parser.add_argument("--speaker-hash-split", action="store_true",
                        help="cal speakers hash%%3==0; evaluation speakers hash%%3!=0")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    exclusion = set()
    for file in args.exclude:
        for line in Path(file).read_text().splitlines():
            exclusion.add(json.loads(line)["uid"])
    vocab = json.loads((DATA_ROOT / "checkpoints/w2vbert_synth_tense_v2/step_3000/jamo_vocab.json").read_text())["vocab"]
    manifest = Path(args.manifest)
    protocol = {"normal_quota_per_expected_vowel": 300, "error_quota_per_direction_vowel": 100,
                "per_item_pool_limit": 600, "selection": "deterministic UID priority, item-balanced within cells, union without duplicating recordings",
                "splits": {"cal_source": args.cal_source, "test_source": "test",
                           "item_partition": "existing rank cal/test item buckets",
                           "speaker_hash_partition": args.speaker_hash_split},
                "fitting": "No fitting or model inference in preparation. Freeze next score/policy protocol before extraction/evaluation.",
                "limitations": ["Existing corpus labels, not new independent human validation", "Not certified unused in project history", "Single sibilant-error/fully-correct subset; complex errors still required", "Base model prior exposure remains"],
                "excluded_prior_uids": len(exclusion),
                "exclusion_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in args.exclude}}
    (out / "selection_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    results, reports = {}, {}
    for src, split in ((args.cal_source, "cal"), ("test", "test")):
        pools = collections.defaultdict(list)
        excluded, available = collections.Counter(), collections.Counter()
        seen = set()
        with (manifest / (src + ".tsv")).open() as ft, (manifest / (src + ".ltr")).open() as fa, \
                (manifest / (src + ".canon")).open() as fc, (manifest / (src + ".meta.jsonl")).open() as fm:
            audio_root = next(ft).strip()
            for index, (t, a, c, m) in enumerate(zip(ft, fa, fc, fm)):
                meta = json.loads(m)
                if destination(meta["item"]) != split:
                    continue
                if args.speaker_hash_split:
                    speaker_bucket = int(hashlib.sha256(("margin-expansion-v2:" + meta["spk"]).encode()).hexdigest()[:8], 16) % 3
                    if (speaker_bucket == 0) != (split == "cal"):
                        excluded["speaker_partition"] += 1
                        continue
                if meta["stem"] in exclusion or meta["stem"] in seen:
                    excluded["prior_or_duplicate_uid"] += 1
                    continue
                seen.add(meta["stem"])
                rel, n = t.rstrip().split("\t")
                row, why = record(meta, a.split(), c.split(), int(n), Path(audio_root) / rel, src, index, vocab)
                if row is None:
                    excluded[why] += 1
                    continue
                if not row["sites"]:
                    continue
                cells = {(s["canon"], s["actual"], row["canonical"][s["q"] + 1])
                         for s in row["sites"] if row["role"] == "normal" or s["canon"] != s["actual"]}
                for cell in cells:
                    available[str(cell)] += 1
                    heap = pools[(cell, row["item"])]
                    entry = (-priority(row["uid"]), row["uid"], row)
                    if len(heap) < 600:
                        heapq.heappush(heap, entry)
                    elif entry[0] > heap[0][0]:
                        heapq.heapreplace(heap, entry)
        selected = {}
        for cell in sorted({k[0] for k in pools}):
            rows = [e[2] for (key, _), heap in pools.items() if key == cell for e in heap]
            quota = 300 if cell[0] == cell[1] else 100
            for row in balanced(rows, quota):
                selected[row["uid"]] = row
        results[split] = sorted(selected.values(), key=lambda r: priority(r["uid"]))
        reports[split] = {"available_cells": dict(available), "excluded": dict(excluded), "selected": stats(results[split])}
        print(split, json.dumps(reports[split]["selected"], ensure_ascii=False), flush=True)
    checks = {}
    for key in ("uid", "speaker", "item", "norm_prompt"):
        overlap = {r[key] for r in results["cal"]} & {r[key] for r in results["test"]}
        checks[key] = len(overlap)
        if overlap:
            raise ValueError("cal/test overlap: " + key)
    for split, rows in results.items():
        for row in rows:
            with wave.open(row["wav"], "rb") as f:
                if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) != (16000, 1, 2, row["frames"]):
                    raise ValueError("audio metadata mismatch: " + row["uid"])
        with (out / (split + ".jsonl")).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"splits": reports, "overlap_checks": checks}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

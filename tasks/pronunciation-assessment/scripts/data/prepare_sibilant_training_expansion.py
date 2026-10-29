"""Expand conditional TAG training coverage without model-score selection.

Retains the established single-sibilant-error/fully-correct contract. All known
comparison speakers, prompts, items and recordings are excluded. This does
not make the previously trained base encoder independent of its old corpus.
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
import gzip
import hashlib
import heapq
import json
import shutil
import time
import wave

from shared.paths import DATA_ROOT, MDD_OUTPUTS
from prepare_sibilant_rank_pilot import record, destination, balanced, priority, stats, VOWELS


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    parser.add_argument("--cache", default=str(MDD_OUTPUTS / "cache/foreign_tag_lab_parsed.jsonl.gz"))
    parser.add_argument("--exclude", nargs="+", required=True, help="existing cal/test/confirmation manifests")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out, manifest = Path(args.out), Path(args.manifest)
    out.mkdir(parents=True, exist_ok=False)
    excludes = {key: set() for key in ("uid", "speaker", "item", "norm_prompt")}
    for path in args.exclude:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            for key in excludes:
                excludes[key].add(row[key])
    vocab_file = DATA_ROOT / "checkpoints/w2vbert_synth_tense_v2/step_3000/jamo_vocab.json"
    protocol = {"normal_per_expected_vowel_position": 150, "error_per_direction_vowel_position": 100,
                "per_item_cell_pool": 100, "selection": "deterministic UID priority, item-balanced within cell, union unique recordings",
                "heldout_exclusion_counts": {k: len(v) for k, v in excludes.items()},
                "source_sha256": {str(p): sha(p) for p in [manifest / "train.tsv", Path(args.cache), vocab_file] + [Path(p) for p in args.exclude]},
                "fit": "No model inference or scores, no test-based sample selection",
                "limits": ["Existing TAG actual / LAB expected, not fresh human validation",
                           "Same original item partition; conditional coverage may still be structurally limited",
                           "Base encoder prior exposure remains; complex errors/deletions still need separate evaluation"]}
    (out / "selection_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    for path in (Path(__file__), Path(__file__).with_name("prepare_sibilant_rank_pilot.py")):
        shutil.copy2(path, out / path.name)
    vocab = json.loads(vocab_file.read_text())["vocab"]
    membership = {}
    with (manifest / "train.tsv").open() as stream:
        audio_root = next(stream).strip()
        for index, line in enumerate(stream):
            rel, frames = line.rstrip().split("\t")
            if "/mdd_gold_wav16k/" not in "/" + rel or not 8000 <= int(frames) <= 192000:
                continue
            uid = Path(rel).stem
            parts = uid.split("-")
            if len(parts) < 6 or destination(parts[5]) != "train" or uid in excludes["uid"]:
                continue
            membership.setdefault(uid, (int(frames), index, str(Path(audio_root) / rel)))
    print("indexed unique training recordings", len(membership), flush=True)
    seen, pools = set(), collections.defaultdict(list)
    rejected, available = collections.Counter(), collections.Counter()
    started = time.monotonic()
    with gzip.open(args.cache, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            meta = json.loads(line)
            uid = meta["stem"]
            if uid not in membership or uid in seen:
                continue
            seen.add(uid)
            if any(meta[field] in excludes[key] for key, field in (("speaker", "spk"), ("item", "item"), ("norm_prompt", "norm_prompt"))):
                rejected["heldout_speaker_item_or_prompt"] += 1
                continue
            frames, index, wav = membership[uid]
            row, why = record(meta, meta["ltr"], meta["canon"], frames, wav, "train", index, vocab)
            if row is None:
                rejected[why] += 1
                continue
            cells = set()
            for site in row["sites"]:
                if row["role"] == "error" and site["canon"] == site["actual"]:
                    continue
                q = site["q"]
                position = "initial" if q == 0 else "post_vowel" if row["canonical"][q - 1] in VOWELS else "post_consonant"
                cells.add((site["canon"], site["actual"], row["canonical"][q + 1], position))
            for cell in cells:
                available[str(cell)] += 1
                pool = pools[(cell, row["item"])]
                entry = (-priority(uid), uid, row)
                if len(pool) < 100:
                    heapq.heappush(pool, entry)
                elif entry[0] > pool[0][0]:
                    heapq.heapreplace(pool, entry)
            if number % 10000 == 0:
                print("source rows", number, "eligible seen", len(seen), "cells", len(available),
                      "seconds", round(time.monotonic() - started, 1), flush=True)
    selected = {}
    for cell in sorted({key[0] for key in pools}):
        candidates = [entry[2] for (key, _), pool in pools.items() if key == cell for entry in pool]
        for row in balanced(candidates, 150 if cell[0] == cell[1] else 100):
            selected[row["uid"]] = row
    rows = sorted(selected.values(), key=lambda r: priority(r["uid"]))
    checks = {key: len({r[key] for r in rows} & values) for key, values in excludes.items()}
    if any(checks.values()):
        raise AssertionError("heldout overlap")
    for row in rows:
        with wave.open(row["wav"], "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) != (16000, 1, 2, row["frames"]):
                raise ValueError("source audio metadata mismatch")
    with (out / "train.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    cells = collections.Counter()
    for row in rows:
        for site in row["sites"]:
            q = site["q"]
            position = "initial" if q == 0 else "post_vowel" if row["canonical"][q - 1] in VOWELS else "post_consonant"
            cells[str((site["canon"], site["actual"], row["canonical"][q + 1], position))] += 1
    summary = {"selected": stats(rows), "selected_cells": dict(cells), "available_cells": dict(available),
               "excluded": dict(rejected), "overlap_checks": checks, "train_sha256": sha(out / "train.jsonl")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["selected"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

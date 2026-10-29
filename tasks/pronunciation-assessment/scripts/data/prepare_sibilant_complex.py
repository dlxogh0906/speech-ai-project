"""Prepare full TAG sibilant outcomes without single-error/binary filtering.

Reuses established item/speaker partitions and all-optimal edit mappings.
Case-balanced exploratory cohorts, never a natural-prevalence or unused test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]

import argparse
import collections
import gzip
import hashlib
import heapq
import json
import shutil
import time
import wave
from shared.paths import MDD_OUTPUTS, DATA_ROOT
from eval_sibilant_directional import all_optimal_mapping, sha
from prepare_sibilant_rank_pilot import destination, balanced, priority
from mdd.sibilant_labels import label_sites, VOWELS


def read(path):
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            yield row.get("row", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    parser.add_argument("--cache", default=str(MDD_OUTPUTS / "cache/foreign_tag_lab_parsed.jsonl.gz"))
    parser.add_argument("--heldout", nargs="+", required=True)
    parser.add_argument("--prior-train", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out, manifest = Path(args.out), Path(args.manifest)
    out.mkdir(parents=True, exist_ok=False)
    heldout = [r for p in args.heldout for r in read(p)]
    old_train = [r for p in args.prior_train for r in read(p)]
    protected = {k: {r[k] for r in heldout} for k in ("uid", "speaker", "item", "norm_prompt")}
    trained_speakers = {r["speaker"] for r in old_train}
    known_uids = {r["uid"] for r in old_train + heldout}
    vocab_path = DATA_ROOT / "checkpoints/w2vbert_synth_tense_v2/step_3000/jamo_vocab.json"
    vocab = json.loads(vocab_path.read_text())["vocab"]
    files = [Path(args.cache), vocab_path] + [Path(x) for x in args.heldout + args.prior_train]
    files += [manifest / ("train.tsv")] + [manifest / ("test." + suffix) for suffix in ("tsv", "ltr", "canon", "meta.jsonl")]
    protocol = {"scope": __doc__, "args": vars(args), "source_sha256": {str(x): sha(x) for x in files},
                "selection": "UID deterministic priority, up to100 candidates per item/cell, item-balanced100train/50caltest per cell; union unique recordings",
                "cell": "expected/label_kind/utterance_not_wholly_correct/highvowel_i_u_eu_or_other",
                "partition": "existing ssrank item buckets; officialtestcal speakers margin-expansion-v2 hash%3==0,testothers; exclude prior addedtraining speakers from evaluation; training excludes alloldheldouts andnewselectedevalspeakers",
                "labels": "Human TAG actual/LAB expected; all optimal edit mappings. Third phones, deletions, ambiguous mappings and changed nextphones retained. No model scores or inputquality gating.",
                "limits": ["case-balanced existing corpus exploration, not prevalence or independent listeners", "historical baseencoder exposure not removed", "annotation-invalid/OOV andduration exclusions explicit; not service input rejects", "no model training in preparation"]}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    for file in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_labels.py"):
        shutil.copy2(file, out / file.name)
    available, skips, results = {}, {}, {}

    def process(meta, actual, canonical, frames, wav, source, index, pools, counts, rejected):
        if not 8000 <= frames <= 192000:
            rejected["duration"] += 1
            return
        if not actual or not canonical or any(t not in vocab for t in actual + canonical) or any(meta.get("dropped", {}).values()):
            rejected["annotation_missing_dropped_or_oov"] += 1
            return
        if not any(c in ("ㅅ", "ㅆ") and canonical[q + 1] in VOWELS for q, c in enumerate(canonical[:-1])):
            rejected["no_target_onset"] += 1
            return
        mapping = [{i} for i in range(len(canonical))] if canonical == actual else all_optimal_mapping(canonical, actual)
        sites = label_sites(canonical, actual, mapping)
        row = {"uid": meta["stem"], "speaker": meta["spk"], "item": meta["item"], "prompt": meta["prompt"],
               "norm_prompt": meta["norm_prompt"], "wav": str(wav), "frames": frames, "actual": actual,
               "canonical": canonical, "sites": sites, "utterance_all_correct": actual == canonical,
               "source_split": source, "manifest_index": index, "source_manifest": str(manifest),
               "language": meta.get("language"), "label_provenance": "human_TAG_actual__LAB_phonemic_canonical",
               "previously_selected_uid": meta["stem"] in known_uids}
        counts["utterances"] += 1
        counts["not_wholly_correct"] += actual != canonical
        counts.update("site/" + s["canon"] + "/" + s["label_kind"] for s in sites)
        cells = {(s["canon"], s["label_kind"], actual != canonical,
                  s["following"] if s["following"] in ("ㅣ", "ㅜ", "ㅡ") else "other") for s in sites}
        for cell in cells:
            bucket = pools[(cell, row["item"])]
            entry = (-priority(row["uid"]), row["uid"], row)
            if len(bucket) < 100:
                heapq.heappush(bucket, entry)
            elif entry[0] > bucket[0][0]:
                heapq.heapreplace(bucket, entry)

    def select(split, pools, counts, rejected):
        selected = {}
        for cell in sorted({c for c, _ in pools}):
            candidates = [e[2] for (c, _), bucket in pools.items() if c == cell for e in bucket]
            for row in balanced(candidates, 100 if split == "train" else 50):
                selected[row["uid"]] = row
        results[split] = sorted(selected.values(), key=lambda r: priority(r["uid"]))
        available[split], skips[split] = dict(counts), dict(rejected)
        print(split, "inventory", dict(counts), "selected", len(selected), flush=True)

    for split in ("cal", "test"):
        pools, counts, rejected = collections.defaultdict(list), collections.Counter(), collections.Counter()
        with (manifest / "test.tsv").open() as ft, (manifest / "test.ltr").open() as fa, (manifest / "test.canon").open() as fc, (manifest / "test.meta.jsonl").open() as fm:
            audio_root = Path(next(ft).strip())
            for index, (tsv, aa, cc, mm) in enumerate(zip(ft, fa, fc, fm)):
                meta = json.loads(mm)
                if destination(meta["item"]) != split:
                    continue
                bucket = int(hashlib.sha256(("margin-expansion-v2:" + meta["spk"]).encode()).hexdigest()[:8], 16) % 3
                if (bucket == 0) != (split == "cal") or meta["spk"] in trained_speakers:
                    rejected["speaker_partition_or_prior_training_speaker"] += 1
                    continue
                rel, fr = tsv.rstrip().split("\t")
                process(meta, aa.split(), cc.split(), int(fr), audio_root / rel, "test", index, pools, counts, rejected)
        select(split, pools, counts, rejected)
    for split in ("cal", "test"):
        for row in results[split]:
            for k in protected:
                protected[k].add(row[k])
    membership = {}
    with (manifest / "train.tsv").open() as stream:
        audio_root = Path(next(stream).strip())
        for index, line in enumerate(stream):
            rel, fr = line.rstrip().split("\t")
            uid = Path(rel).stem
            parts = uid.split("-")
            if len(parts) < 6 or "/mdd_gold_wav16k/" not in "/" + rel or destination(parts[5]) != "train":
                continue
            if parts[0] in protected["speaker"] or parts[5] in protected["item"] or uid in protected["uid"]:
                continue
            membership.setdefault(uid, (int(fr), index, audio_root / rel))
    pools, counts, rejected, seen = collections.defaultdict(list), collections.Counter(), collections.Counter(), set()
    started = time.monotonic()
    with gzip.open(args.cache, "rt", encoding="utf-8") as stream:
        for line in stream:
            meta = json.loads(line)
            uid = meta["stem"]
            if uid not in membership or uid in seen:
                continue
            seen.add(uid)
            if meta["norm_prompt"] in protected["norm_prompt"]:
                rejected["heldout_prompt"] += 1
                continue
            fr, index, wav = membership[uid]
            process(meta, meta["ltr"], meta["canon"], fr, wav, "train", index, pools, counts, rejected)
            if len(seen) % 10000 == 0:
                print("train scanned", len(seen), "seconds", round(time.monotonic() - started, 1), flush=True)
    select("train", pools, counts, rejected)
    overlap = {}
    for a, b in (("train", "cal"), ("train", "test"), ("cal", "test")):
        overlap[a + "-" + b] = {k: len({r[k] for r in results[a]} & {r[k] for r in results[b]}) for k in protected}
        if any(overlap[a + "-" + b].values()):
            raise AssertionError("split overlap")
    selected_stats = {}
    for split, rows in results.items():
        for row in rows:
            with wave.open(row["wav"], "rb") as stream:
                if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) != (16000, 1, 2, row["frames"]):
                    raise AssertionError("WAV metadata changed")
        with (out / (split + ".jsonl")).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        selected_stats[split] = {"utterances": len(rows), "speakers": len({r["speaker"] for r in rows}),
                                 "items": len({r["item"] for r in rows}),
                                 "sites": dict(collections.Counter(s["canon"] + "/" + s["label_kind"] for r in rows for s in r["sites"])),
                                 "previously_selected_uid": sum(r["previously_selected_uid"] for r in rows),
                                 "not_wholly_correct": sum(not r["utterance_all_correct"] for r in rows)}
    report = {"available": available, "exclusions": skips, "selected": selected_stats, "overlap": overlap,
              "output_sha256": {s: sha(out / (s + ".jsonl")) for s in results}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

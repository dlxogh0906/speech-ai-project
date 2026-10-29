"""Additional frozen-policy check from previously unselected TAG recordings.

Retain established item/speaker partitions and full multi-error TAG outcomes.
Exclude every listed earlier UID plus auxiliary-training/calibration speakers.
This is existing-corpus, outcome-balanced evaluation, not new human validation.
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
import hashlib
import heapq
import json
import shutil
import wave
from eval_sibilant_directional import all_optimal_mapping, sha, write_line
from prepare_sibilant_rank_pilot import destination, balanced, priority
from mdd.sibilant_labels import label_sites, VOWELS


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            yield row.get("row", row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    prior = [r for path in p["exclude_uid_sources"] for r in records(path)]
    known = {r["uid"] for r in prior}
    protected = [r for path in p["exclude_speaker_sources"] for r in records(path)]
    speakers = {r["speaker"] for r in protected}
    fit_rows = {s: list(records(path)) for s, path in p["auxiliary_fit_sources"].items()}
    forbidden_items = {r["item"] for rows in fit_rows.values() for r in rows}
    vocab = json.loads(Path(p["vocab"]).read_text())["vocab"]
    manifest = Path(p["manifest"])
    pools, counts = collections.defaultdict(list), collections.Counter()
    with (manifest / "test.tsv").open() as ft, (manifest / "test.ltr").open() as fa, (manifest / "test.canon").open() as fc, (manifest / "test.meta.jsonl").open() as fm:
        audio_root = Path(next(ft).strip())
        for index, (tsv, aa, cc, mm) in enumerate(zip(ft, fa, fc, fm)):
            meta = json.loads(mm)
            if destination(meta["item"]) != "test":
                continue
            bucket = int(hashlib.sha256(("margin-expansion-v2:" + meta["spk"]).encode()).hexdigest()[:8], 16) % 3
            if bucket == 0 or meta["spk"] in speakers or meta["item"] in forbidden_items:
                counts["excluded_partition_or_fit_overlap"] += 1
                continue
            if meta["stem"] in known:
                counts["previously_selected_uid"] += 1
                continue
            rel, frames = tsv.rstrip().split("\t")
            frames, actual, canonical = int(frames), aa.split(), cc.split()
            if not 8000 <= frames <= 192000:
                counts["duration_excluded"] += 1
                continue
            if not actual or not canonical or any(t not in vocab for t in actual + canonical) or any(meta.get("dropped", {}).values()):
                counts["invalid_annotation_excluded"] += 1
                continue
            if not any(c in ("ㅅ", "ㅆ") and canonical[q + 1] in VOWELS for q, c in enumerate(canonical[:-1])):
                counts["no_target_onset"] += 1
                continue
            mapping = [{i} for i in range(len(canonical))] if actual == canonical else all_optimal_mapping(canonical, actual)
            sites = label_sites(canonical, actual, mapping)
            row = {"uid": meta["stem"], "speaker": meta["spk"], "item": meta["item"], "prompt": meta["prompt"],
                   "norm_prompt": meta["norm_prompt"], "wav": str(audio_root / rel), "frames": frames,
                   "actual": actual, "canonical": canonical, "sites": sites, "source_split": "test",
                   "source_manifest": str(manifest), "manifest_index": index, "language": meta.get("language"),
                   "utterance_all_correct": actual == canonical, "label_provenance": "existing_human_TAG_actual__LAB_phonemic_canonical"}
            counts["eligible_utterances"] += 1
            cells = {(s["canon"], s["label_kind"], actual != canonical, s["following"] if s["following"] in ("ㅣ", "ㅜ", "ㅡ") else "other") for s in sites}
            for cell in cells:
                bucket_rows = pools[(cell, row["item"])]
                entry = (-priority(row["uid"]), row["uid"], row)
                if len(bucket_rows) < 100:
                    heapq.heappush(bucket_rows, entry)
                elif entry[0] > bucket_rows[0][0]:
                    heapq.heapreplace(bucket_rows, entry)
    selected = {}
    for cell in sorted({c for c, _ in pools}):
        candidates = [x[2] for (c, _), bucket in pools.items() if c == cell for x in bucket]
        for row in balanced(candidates, p["per_cell"]):
            selected[row["uid"]] = row
    rows = sorted(selected.values(), key=lambda row: priority(row["uid"]))
    # UID aliases of a fitted recording are also forbidden; hashes are computed
    # only here and never select on model outputs or service admission.
    fitted_audio = set()
    for rr in fit_rows.values():
        for row in rr:
            fitted_audio.add(sha(row["wav"]))
    seen_hashes, kept = set(), []
    for row in rows:
        digest = sha(row["wav"])
        if digest in fitted_audio or digest in seen_hashes:
            counts["audio_alias_of_fit_or_selected"] += 1
            continue
        with wave.open(row["wav"], "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) != (16000, 1, 2, row["frames"]):
                raise AssertionError("WAV metadata mismatch")
        row["wav_sha256"] = digest
        seen_hashes.add(digest)
        kept.append(row)
    with (out / "additional.jsonl").open("x") as stream:
        for row in kept:
            write_line(stream, row)
    overlap = {s: {k: len({r[k] for r in kept} & {r[k] for r in rr}) for k in ("uid", "speaker", "item", "norm_prompt")} for s, rr in fit_rows.items()}
    if any(v for row in overlap.values() for v in row.values()):
        raise AssertionError("auxiliary fit split overlap")
    report = {"scope": p["scope"], "counts": dict(counts), "utterances": len(kept), "speakers": len({r["speaker"] for r in kept}),
              "items": len({r["item"] for r in kept}), "sites": dict(collections.Counter(s["canon"] + "/" + s["label_kind"] for r in kept for s in r["sites"])),
              "fit_overlap": overlap, "previously_listed_uid_overlap": len({r["uid"] for r in kept} & known),
              "output_sha256": sha(out / "additional.jsonl"), "policy_frozen_before_selection": p["policy_sha256"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

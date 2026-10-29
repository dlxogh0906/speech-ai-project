"""Inventory remaining recordings in a fixed cal/test partition before decoding.

Keep all valid target utterances, including multi-error, third-phone, deletion
and ambiguous outcomes. No model score or service admission selects the sample.
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
import json
import shutil
import wave
from prepare_sibilant_cv_additional import records
from prepare_sibilant_rank_pilot import destination, priority
from eval_sibilant_directional import all_optimal_mapping, sha, write_line
from mdd.sibilant_labels import label_sites, VOWELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    if p["partition"] not in ("cal", "test"):
        raise ValueError("only fixed heldout partitions allowed")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    known = {r["uid"] for path in p["exclude_uid_sources"] for r in records(path)}
    groups = {name: [r for path in paths for r in records(path)] for name, paths in p["disjoint_sources"].items()}
    protected = {key: {r[key] for rr in groups.values() for r in rr} for key in ("uid", "speaker", "item", "norm_prompt")}
    vocab = json.loads(Path(p["vocab"]).read_text())["vocab"]
    manifest, selected, counts = Path(p["manifest"]), [], collections.Counter()
    with (manifest / "test.tsv").open() as ft, (manifest / "test.ltr").open() as fa, (manifest / "test.canon").open() as fc, (manifest / "test.meta.jsonl").open() as fm:
        audio_root = Path(next(ft).strip())
        for index, (tsv, aa, cc, mm) in enumerate(zip(ft, fa, fc, fm)):
            meta = json.loads(mm)
            if destination(meta["item"]) != p["partition"]:
                continue
            bucket = int(hashlib.sha256(("margin-expansion-v2:" + meta["spk"]).encode()).hexdigest()[:8], 16) % 3
            if (bucket == 0) != (p["partition"] == "cal"):
                counts["other_speaker_partition"] += 1
                continue
            if (meta["stem"] in protected["uid"] or meta["spk"] in protected["speaker"]
                    or meta["item"] in protected["item"] or meta["norm_prompt"] in protected["norm_prompt"]):
                counts["protected_fit_or_other_partition"] += 1
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
            selected.append({"uid": meta["stem"], "speaker": meta["spk"], "item": meta["item"],
                             "prompt": meta["prompt"], "norm_prompt": meta["norm_prompt"], "wav": str(audio_root / rel),
                             "frames": frames, "actual": actual, "canonical": canonical,
                             "sites": label_sites(canonical, actual, mapping), "source_split": "test",
                             "experiment_partition": p["partition"], "source_manifest": str(manifest),
                             "manifest_index": index, "language": meta.get("language"),
                             "utterance_all_correct": actual == canonical,
                             "label_provenance": "existing_human_TAG_actual__LAB_phonemic_canonical"})
    protected_audio = {sha(r["wav"]) for rr in groups.values() for r in rr}
    seen, kept = set(), []
    for row in sorted(selected, key=lambda r: priority(r["uid"])):
        digest = sha(row["wav"])
        if digest in protected_audio or digest in seen:
            counts["protected_or_duplicate_audio_alias"] += 1
            continue
        with wave.open(row["wav"], "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) != (16000, 1, 2, row["frames"]):
                raise AssertionError("WAV metadata mismatch")
        row["wav_sha256"] = digest
        seen.add(digest)
        kept.append(row)
    with (out / "additional.jsonl").open("x") as stream:
        for row in kept:
            write_line(stream, row)
    overlap = {name: {k: len({r[k] for r in kept} & {r[k] for r in rr}) for k in protected} for name, rr in groups.items()}
    if any(v for values in overlap.values() for v in values.values()):
        raise AssertionError("partition overlap")
    report = {"scope": p["scope"], "partition": p["partition"], "counts": dict(counts), "utterances": len(kept),
              "speakers": len({r["speaker"] for r in kept}), "items": len({r["item"] for r in kept}),
              "sites": dict(collections.Counter(s["canon"] + "/" + s["label_kind"] for r in kept for s in r["sites"])),
              "protected_overlap": overlap, "previously_listed_uid_overlap": len({r["uid"] for r in kept} & known),
              "output_sha256": sha(out / "additional.jsonl"), "policy_frozen_before_selection": p["policy_sha256"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

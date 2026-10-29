"""Keep all protected-partition training errors with bounded same-context normals.

Selection uses original human annotation and deterministic UID priority only.
All declared evaluation recordings/speakers/items/prompts and exact PCM aliases
are excluded before fitting. Service admission/model outputs are never read.
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
import wave
from shared.paths import MDD_OUTPUTS, DATA_ROOT
from eval_sibilant_directional import all_optimal_mapping, sha, jsonable
from prepare_sibilant_rank_pilot import destination, priority
from prepare_sibilant_complex import read
from mdd.sibilant_labels import label_sites, VOWELS


def pcm_digest(path, expected_frames=None):
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError("expected original 16k mono PCM16")
        if expected_frames is not None and stream.getnframes() != expected_frames:
            raise ValueError("WAV frame count changed")
        return hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()


def context(row, site):
    return row["item"], site["q"], site["canon"], site["following"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    ap.add_argument("--cache", default=str(MDD_OUTPUTS / "cache/foreign_tag_lab_parsed.jsonl.gz"))
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--normal-per-context", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.normal_per_context < 1:
        raise ValueError("positive normal cap required")
    out, manifest = Path(a.out), Path(a.manifest)
    out.mkdir(parents=True, exist_ok=False)
    heldout = [r for path in a.heldout for r in read(path)]
    protected = {k: {r[k] for r in heldout} for k in ("uid", "speaker", "item", "norm_prompt")}
    heldout_audio = {r["wav"] for r in heldout}
    heldout_pcm = {pcm_digest(path) for path in heldout_audio}
    vocab_path = DATA_ROOT / "checkpoints/w2vbert_synth_tense_v2/step_3000/jamo_vocab.json"
    vocab = json.loads(vocab_path.read_text())["vocab"]
    source_files = [Path(a.cache), vocab_path, manifest / "train.tsv"] + [Path(p) for p in a.heldout]
    protocol = {"scope": __doc__, "args": vars(a), "source_sha256": {str(p): sha(p) for p in source_files},
                "selection": "all target errors including third/deletion/ambiguity; deterministic up-to-N normal recordings for every same item/position/expected/vowel context, separately wholly-correct and other-error utterances; union recordings",
                "labels": "existing TAG actual and LAB canonical, full outcomes; no fresh people or model-derived truth",
                "limits": "item domain remains restricted; historical base encoder exposure not removed; all prior tests are reused, never an untouched final test"}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(__file__, out / Path(__file__).name)
    membership, excluded = {}, collections.Counter()
    with (manifest / "train.tsv").open() as stream:
        audio_root = Path(next(stream).strip())
        for index, line in enumerate(stream):
            rel, frames = line.rstrip().split("\t")
            uid, parts = Path(rel).stem, Path(rel).stem.split("-")
            if len(parts) < 6 or "/mdd_gold_wav16k/" not in "/" + rel or destination(parts[5]) != "train":
                continue
            if uid in protected["uid"] or parts[0] in protected["speaker"] or parts[5] in protected["item"]:
                excluded["protected_uid_speaker_or_item"] += 1
                continue
            membership.setdefault(uid, (int(frames), index, audio_root / rel))
    errors, pools, seen, inventory = {}, collections.defaultdict(list), set(), collections.Counter()
    with gzip.open(a.cache, "rt", encoding="utf-8") as stream:
        for line in stream:
            meta = json.loads(line)
            uid = meta["stem"]
            if uid not in membership or uid in seen:
                continue
            seen.add(uid)
            frames, index, wav = membership[uid]
            if meta["norm_prompt"] in protected["norm_prompt"]:
                excluded["protected_prompt"] += 1
                continue
            actual, canonical = meta["ltr"], meta["canon"]
            if not 8000 <= frames <= 192000:
                excluded["duration"] += 1
                continue
            if not actual or not canonical or any(t not in vocab for t in actual + canonical) or any(meta.get("dropped", {}).values()):
                excluded["annotation_missing_dropped_or_oov"] += 1
                continue
            if not any(t in ("ㅅ", "ㅆ") and canonical[q + 1] in VOWELS for q, t in enumerate(canonical[:-1])):
                excluded["no_target_onset"] += 1
                continue
            mapping = [{q} for q in range(len(canonical))] if canonical == actual else all_optimal_mapping(canonical, actual)
            sites = label_sites(canonical, actual, mapping)
            row = {"uid": uid, "speaker": meta["spk"], "item": meta["item"], "prompt": meta["prompt"],
                   "norm_prompt": meta["norm_prompt"], "wav": str(wav), "frames": frames, "actual": actual,
                   "canonical": canonical, "sites": sites, "utterance_all_correct": actual == canonical,
                   "source_split": "train", "manifest_index": index, "source_manifest": str(manifest),
                   "language": meta.get("language"), "label_provenance": "human_TAG_actual__LAB_phonemic_canonical"}
            inventory["utterances"] += 1
            inventory.update(s["canon"] + "/" + s["label_kind"] for s in sites)
            if any(s["label_kind"] != "correct" for s in sites):
                errors[uid] = row
            for site in sites:
                if site["label_kind"] != "correct":
                    continue
                key = context(row, site) + (row["utterance_all_correct"],)
                entry = (-priority(uid), uid, row)
                if len(pools[key]) < a.normal_per_context:
                    heapq.heappush(pools[key], entry)
                elif entry[0] > pools[key][0][0]:
                    heapq.heapreplace(pools[key], entry)
            if inventory["utterances"] % 20000 == 0:
                print("target inventory", inventory["utterances"], "all error utterances", len(errors), flush=True)
    selected = dict(errors)
    for pool in pools.values():
        selected.update((entry[1], entry[2]) for entry in pool)
    rows, selected_pcm = [], collections.Counter()
    for row in sorted(selected.values(), key=lambda r: priority(r["uid"])):
        digest = pcm_digest(row["wav"], row["frames"])
        if digest in heldout_pcm:
            excluded["protected_exact_pcm_alias"] += 1
            continue
        rows.append(row)
        selected_pcm[digest] += 1
    overlap = {k: len({r[k] for r in rows} & protected[k]) for k in protected}
    if any(overlap.values()):
        raise AssertionError("training/heldout overlap")
    with (out / "train.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"inventory": dict(inventory), "exclusions": dict(excluded),
              "selected": {"utterances": len(rows), "speakers": len({r["speaker"] for r in rows}),
                           "items": len({r["item"] for r in rows}), "sites": dict(collections.Counter(s["canon"] + "/" + s["label_kind"] for r in rows for s in r["sites"])),
                           "unique_pcm": len(selected_pcm), "duplicate_pcm_copies": sum(n - 1 for n in selected_pcm.values())},
              "protected_rows": len(heldout), "protected_unique_uids": len(protected["uid"]),
              "protected_pcm": len(heldout_pcm), "protected_overlap": overlap,
              "output_sha256": sha(out / "train.jsonl"), "model_outputs_used": False, "admission_filter_used": False}
    (out / "report.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": sha(out / "report.json")}) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Capture runtime-only transition inputs and actual current formatter outputs.

Run after freezing a protocol. TAG supervision is attached only after all
runtime sibilant candidates have been extracted. This does not deploy anything.
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
import json
import shutil
import time

import numpy as np

import eval_sibilant_directional as service
from eval_sibilant_rank_pilot import sequence_log_probs
from mdd.sibilant_transition import PAIR, VOWELS, ACOUSTIC_NAMES, acoustic_tracks, observation_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--service-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke only; zero processes every frozen record")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    previous = json.loads(Path(args.service_metadata).read_text())
    for definition in protocol["data"].values():
        if service.sha(definition["path"]) != definition["sha256"]:
            raise ValueError("frozen data changed")
    # A captured baseline only has meaning with the same source and scorers.
    for name, digest in previous["source_sha256"].items():
        if service.sha(name) != digest:
            raise ValueError("baseline source changed: " + name)
    paths = {Path(p).name: p for p in previous["scoring_sha256"]}
    for name, digest in previous["scoring_sha256"].items():
        if service.sha(name) != digest:
            raise ValueError("scoring artifact changed: " + name)
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"],
                               norms=paths["norms.json"], calib=paths["calibration.json"],
                               flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    app = service.app_module(config, previous["base_ckpt"])
    original_forward = app.S["fwd"]
    captured = {}

    def capture(decoder, wavs):
        if len(wavs) != 1:
            raise ValueError("pilot captures singleton production inference")
        lp, lengths = original_forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S["fwd"] = capture
    sources = [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_transition.py",
               ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_acoustics.py"]
    snapshots = out / "source_snapshots"
    snapshots.mkdir()
    for f in sources:
        shutil.copy2(f, snapshots / f.name)
    metadata = {"protocol": protocol, "protocol_sha256": service.sha(args.protocol),
                "service_metadata": previous, "service_metadata_sha256": service.sha(args.service_metadata),
                "pilot_source_sha256": {str(f): service.sha(f) for f in sources},
                "acoustic_features": ACOUSTIC_NAMES, "limit": args.limit,
                "runtime_inputs": "VAD PCM, frozen logits and runtime canonical only; no TAG boundaries/actual labels",
                "hardware": "local GPU, no remote NPU identity or latency attestation"}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    dec = app.S["dec"]
    summaries = {}
    for split, definition in protocol["data"].items():
        rows = service.read_jsonl(definition["path"])
        if args.limit:
            rows = rows[:args.limit]
        arrays = {k: [] for k in ("posterior", "acoustic", "mask", "context")}
        labels, feature_rows, counts = [], [], collections.Counter()
        started = time.monotonic()
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for number, row in enumerate(rows, 1):
                captured.clear()
                x = service.load_audio(row)
                code = "too_short" if len(x) < 4800 else None
                if code is None:
                    code, _, _ = app.check_audio(x[:320000], row["prompt"])
                tick = time.monotonic()
                raw, body, canon = None, None, []
                features_by_q = {}
                if code is None:
                    _, _, raw = app.analyze(x[:320000], row["prompt"], "")
                    body = service.response(app, row, raw)
                    canon = raw["canon"]
                    wav, lp = captured["wav"], captured["lp"]
                    targets = [q for q, t in enumerate(canon) if t in PAIR and
                               q + 1 < len(canon) and canon[q + 1] in VOWELS]
                    tracks = acoustic_tracks(wav) if targets and len(wav) >= 640 else None
                    for q in targets:
                        if tracks is None:
                            counts["audio_under40ms"] += 1
                            continue
                        value = observation_window(wav, lp, canon, q, dec.d.index, dec.blank, tracks)
                        if value is None:
                            counts["paired_alignment_failure"] += 1
                            continue
                        ix = len(feature_rows)
                        for key in arrays:
                            arrays[key].append(value[key])
                        seq_scores = sequence_log_probs(lp, value["anchor"]["sequences"], dec.blank)
                        llr = seq_scores[1] - seq_scores[0]
                        if not np.isfinite(llr):
                            raise ValueError("nonfinite paired sequence score")
                        frames = sorted(set(sum(value["anchor"]["segments"], [])))
                        token_margin = float(lp[frames, dec.d.index("ㅆ")].max() -
                                             lp[frames, dec.d.index("ㅅ")].max())
                        info = {"index": ix, "uid": row["uid"], "q_runtime": q,
                                "canonical": canon[q], "following": value["following"],
                                "position": value["position"], "anchor_frame": value["anchor"]["frame"],
                                "hypothesis_frames": value["anchor"]["hypothesis_frames"],
                                "frame_seconds": value["frame_seconds"], "ctc_llr": llr,
                                "token_margin": token_margin, "baseline_pred": raw["sites"][q].get("pred"),
                                "vad_pcm_sha256": hashlib.sha256(wav.tobytes()).hexdigest()}
                        feature_rows.append(info)
                        features_by_q[q] = ix
                counts["rejected_" + code if code else "accepted"] += 1
                # Attach existing human labels AFTER input-only candidate extraction.
                mapping = service.all_optimal_mapping(row["canonical"], canon) if canon else []
                phones = service.phonemes(body) if body is not None else []
                for site in row["sites"]:
                    opts = mapping[site["q"]] if mapping else set()
                    target = next(iter(opts)) if len(opts) == 1 and None not in opts else None
                    state = ("input_rejected" if code else "runtime_mapping_ambiguous" if target is None else
                             "runtime_canonical_disagrees" if canon[target] != site["canon"] else
                             "features_unavailable" if target not in features_by_q else "available")
                    phone = phones[target] if target is not None else None
                    entry = dict(site)
                    entry.update(uid=row["uid"], speaker=row["speaker"], item=row["item"],
                                 norm_prompt=row["norm_prompt"], language=row.get("language"),
                                 expected=site["canon"], feature_index=features_by_q.get(target) if state == "available" else None,
                                 q_runtime=target, state=state, baseline_phone=phone,
                                 following=row["canonical"][site["q"] + 1],
                                 position="initial" if site["q"] == 0 else
                                 "post_vowel" if row["canonical"][site["q"] - 1] in VOWELS else "post_consonant")
                    labels.append(entry)
                service.write_line(stream, {"uid": row["uid"], "row": row, "rejected": code,
                                             "raw": raw, "body": body,
                                             "feature_indices": features_by_q,
                                             "elapsed_with_features_s": time.monotonic() - tick})
                if number % 50 == 0 or number == len(rows):
                    print("[%s] %d/%d features=%d %.1fs" %
                          (split, number, len(rows), len(feature_rows), time.monotonic() - started), flush=True)
        np.savez_compressed(out / (split + ".npz"), **{k: np.asarray(v, dtype=np.float32 if k != "mask" else bool)
                                                       for k, v in arrays.items()})
        for suffix, records in (("features", feature_rows), ("labels", labels)):
            with (out / (split + "." + suffix + ".jsonl")).open("w") as stream:
                for record in records:
                    service.write_line(stream, record)
        summaries[split] = {"utterances": len(rows), "runtime_features": len(feature_rows),
                            "labelled_sites": len(labels), "input_status": dict(counts),
                            "label_status": dict(collections.Counter(r["state"] for r in labels)),
                            "elapsed_s": time.monotonic() - started}
        (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

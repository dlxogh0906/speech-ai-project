"""Cache every runtime-eligible local pair clip, including restored expectations.

Eligibility uses original decoder and API state, never TAG. Attach TAG labels
after feature extraction only for evaluation. Retain rejected/unavailable sites.
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
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from eval_speech_rescue import observe
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_transition import PAIR, VOWELS, paired_anchor
from mdd.sibilant_margin import annotate_margin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    app, captured = app_from(p), {}
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])
    forward, dec = app.S["fwd"], app.S["dec"]

    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S["fwd"] = capture
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    for name, path in p["cohorts"].items():
        counts, arrays, features, labels = collections.Counter(), [], [], []
        records = service.read_jsonl(path)
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for number, record in enumerate(records, 1):
                row, prior = record["row"], record["existing"]
                raw, by_q = prior["raw"], {}
                reference = dict(prior)
                if raw is not None:
                    phones = service.phonemes(prior["body"])
                    eligible = [q for q, token in enumerate(raw["canon"][:-1])
                                if token in PAIR and raw["canon"][q + 1] in VOWELS
                                and raw["sites"][q].get("pred") in PAIR and not phones[q].get("errorType")]
                    if eligible:
                        x = service.load_audio(row)[:320000]
                        _, _, current = app.analyze(x, row["prompt"], "")
                        wav, lp = captured["wav"], captured["lp"]
                        sites, _ = annotate_margin(lp, current["canon"], current["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                        if render(app, row, current, sites) != prior["body"] or current["canon"] != raw["canon"]:
                            raise AssertionError("existing complete API response did not reproduce")
                        np.testing.assert_array_equal(x[retained_samples(x)], wav)
                        digest = hashlib.sha256(wav.tobytes()).hexdigest()
                        dt = len(wav) / (16000. * len(lp))
                        for q in eligible:
                            anchor = paired_anchor(lp, [dec.d.index(t) for t in current["canon"]], q,
                                                   [dec.d.index("ㅅ"), dec.d.index("ㅆ")], dec.blank)
                            if anchor is None:
                                counts["alignment_unavailable"] += 1
                                continue
                            left = max(0, int((anchor["frame"] - 15) * dt * 16000))
                            right = min(len(wav), int((anchor["frame"] + 16) * dt * 16000))
                            clip = wav[left:right]
                            value = fe([clip], sampling_rate=16000, return_tensors="np", padding=True)
                            length = int(value["attention_mask"][0].sum())
                            if not 3 <= length <= p["max_frames"]:
                                counts["unsupported_local_length"] += 1
                                continue
                            centers = left + (np.arange(length) + .5) * len(clip) / length
                            relative = np.clip((centers / 16000. - (anchor["frame"] + .5) * dt) / .31, -2, 2).astype(np.float32)
                            by_q[q] = len(arrays)
                            arrays.append((value["input_features"][0, :length].astype(np.float32), relative))
                            features.append({"uid": row["uid"], "q_runtime": q, "canonical": raw["canon"][q],
                                             "baseline_pred": raw["sites"][q].get("pred"), "left": left, "right": right,
                                             "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
                                             "vad_pcm_sha256": digest, "input_frames": length})
                        counts["eligible_utterances_reproduced"] += 1
                    else:
                        counts["cached_unchanged_accepted"] += 1
                    reference["body"] = render(app, row, raw, raw["sites"])
                else:
                    counts["input_rejected"] += 1
                viewed = observe(reference, row)
                mapping = service.all_optimal_mapping(row["canonical"], raw["canon"]) if raw else None
                for site in row["sites"]:
                    options = mapping[site["q"]] if mapping is not None else set()
                    rq = next(iter(options)) if len(options) == 1 and None not in options else None
                    labels.append(dict(site, expected=site["canon"], uid=row["uid"], speaker=row["speaker"], item=row["item"],
                                       language=row.get("language"), q_runtime=rq, feature_index=by_q.get(rq), existing=viewed[site["q"]]))
                service.write_line(stream, {"row": row, "original": record["original"], "existing": reference, "prior": prior})
                if number % 200 == 0:
                    print(name, number, dict(counts), "features", len(arrays), flush=True)
        data = {"x": np.zeros((len(arrays), p["max_frames"], 160), np.float32),
                "relative": np.zeros((len(arrays), p["max_frames"]), np.float32),
                "lengths": np.array([len(v[0]) for v in arrays], np.int64)}
        for i, (x, relative) in enumerate(arrays):
            data["x"][i, :len(x)] = x
            data["relative"][i, :len(x)] = relative
        np.savez(out / (name + ".npz"), **data)
        for suffix, values in (("features", features), ("labels", labels)):
            with (out / (name + "." + suffix + ".jsonl")).open("x") as stream:
                for value in values:
                    service.write_line(stream, value)
        report["cohorts"][name] = {"counts": dict(counts), "features": len(features), "all_sites": len(labels),
                                   "no_TAG_model_inputs": True, "input_sha256": service.sha(out / (name + ".npz")),
                                   "output_sha256": {suffix: service.sha(out / (name + suffix))
                                                     for suffix in (".npz", ".features.jsonl", ".labels.jsonl", ".responses.jsonl")}}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

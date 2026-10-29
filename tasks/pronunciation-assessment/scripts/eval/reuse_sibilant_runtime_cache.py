"""Reuse audited expanded local inputs and check against fresh runtime guards.

The full earlier confirmation/preserve caches are copied from the unchanged
extractor. Expanded inputs reuse old audio/fbank arrays; relative positions
follow their original clock. Labels are attached only after runtime selection.
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
import json
import shutil
import numpy as np
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import restored_sites
from eval_speech_rescue import observe
from mdd.sibilant_cache_reuse import pair_reference_equal, local_relative
from mdd.sibilant_transition import PAIR, VOWELS


def read(path):
    return service.read_jsonl(path)


def keyed_features(prefix):
    data = dict(np.load(str(prefix) + ".npz"))
    features = read(str(prefix) + ".features.jsonl")
    return {(f["uid"], f["q_runtime"]): (data, i, f) for i, f in enumerate(features)}


def check_input(x, relative, length, clip_hash, entry):
    data, i, feature = entry
    if clip_hash != feature["clip_sha256"] or length != int(data["lengths"][i]):
        raise AssertionError("cached clip identity changed")
    np.testing.assert_array_equal(x, data["x"][i, :length])
    np.testing.assert_array_equal(relative, data["relative"][i, :length])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, fresh, out = json.loads(Path(a.protocol).read_text()), Path(a.fresh), Path(a.out)
    for path, expected in p["sha256"].items():
        if service.sha(path) != expected:
            raise ValueError("frozen source changed: " + path)
    if json.loads((fresh / "finished.json").read_text())["sha256"] != service.sha(fresh / "report.json"):
        raise ValueError("fresh runtime checks incomplete")
    fresh_report = json.loads((fresh / "report.json").read_text())
    for name, value in fresh_report["cohorts"].items():
        for suffix, digest in value["output_sha256"].items():
            if service.sha(fresh / (name + suffix)) != digest:
                raise ValueError("fresh output changed")
    checks = {}
    for prefix in p["recent_input_checks"]:
        checks.update(keyed_features(prefix))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    for name, path in p["full_sources"].items():
        if name not in p["legacy"]:
            for suffix in (".npz", ".features.jsonl", ".labels.jsonl", ".responses.jsonl"):
                (out / (name + suffix)).symlink_to((fresh / (name + suffix)).resolve())
            report["cohorts"][name] = dict(fresh_report["cohorts"][name], source="unchanged full runtime extractor")
            continue
        legacy = p["legacy"][name]
        data = dict(np.load(legacy["input_npz"]))
        fs, audits = read(legacy["features"]), read(legacy["audit"])
        lookup = {(f["uid"], f["q_runtime"]): i for i, f in enumerate(fs)}
        old = {r["uid"]: r for r in read(legacy["responses"])}
        guards = keyed_features(fresh / name)
        counts, arrays, features, labels = collections.Counter(), [], [], []
        records = read(path)
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for record in records:
                row, prior = record["row"], record["existing"]
                raw, by_q, reference = prior["raw"], {}, dict(prior)
                if raw is not None:
                    phones = service.phonemes(prior["body"])
                    sites = restored_sites(raw, prior["body"], p["pair_thresholds"])
                    if render(app, row, raw, sites) != prior["body"]:
                        raise AssertionError("old complete API body did not reproduce")
                    counts["baseline_bodies_exact"] += 1
                    for q, token in enumerate(raw["canon"][:-1]):
                        if (token not in PAIR or raw["canon"][q + 1] not in VOWELS
                                or raw["sites"][q].get("pred") not in PAIR or phones[q].get("errorType")):
                            continue
                        key = row["uid"], q
                        if key not in lookup or not pair_reference_equal(old[row["uid"]]["raw"]["canon"], raw["canon"], q):
                            raise ValueError("uncached or reference-unsafe input: " + str(key))
                        i = lookup[key]
                        f, audit, length = fs[i], audits[i], int(data["lengths"][i])
                        if ((f["uid"], f["q_runtime"]) != (audit["uid"], audit["q_runtime"])
                                or f["vad_pcm_sha256"] != audit["vad_pcm_sha256"] or length != audit["input_frames"]):
                            raise AssertionError("original local-input audit mismatch")
                        x = data["x"][i, :length]
                        relative = local_relative(audit["start"], audit["end"], length, f["anchor_frame"],
                                                  f["frame_seconds"], p["relative_scale"])
                        if key in checks:
                            check_input(x, relative, length, audit["clip_sha256"], checks[key])
                            counts["recent_input_exact"] += 1
                        if key in guards:
                            check_input(x, relative, length, audit["clip_sha256"], guards[key])
                            counts["fresh_guard_exact"] += 1
                        by_q[q] = len(arrays)
                        arrays.append((x, relative))
                        features.append({"uid": row["uid"], "q_runtime": q, "canonical": token,
                                         "baseline_pred": raw["sites"][q].get("pred"),
                                         "left": audit["start"], "right": audit["end"], "input_frames": length,
                                         "vad_pcm_sha256": audit["vad_pcm_sha256"], "clip_sha256": audit["clip_sha256"]})
                        counts["legacy_inputs_reused"] += 1
                    reference["body"] = render(app, row, raw, raw["sites"])
                else:
                    counts["input_rejected"] += 1
                viewed = observe(reference, row)
                mapping = service.all_optimal_mapping(row["canonical"], raw["canon"]) if raw else None
                for site in row["sites"]:
                    options = mapping[site["q"]] if mapping is not None else set()
                    rq = next(iter(options)) if len(options) == 1 and None not in options else None
                    labels.append(dict(site, expected=site["canon"], uid=row["uid"], speaker=row["speaker"],
                                       item=row["item"], language=row.get("language"), q_runtime=rq,
                                       feature_index=by_q.get(rq), existing=viewed[site["q"]]))
                service.write_line(stream, {"row": row, "original": record["original"], "existing": reference, "prior": prior})
        if counts["fresh_guard_exact"] != len(guards):
            raise AssertionError("fresh guard coverage changed")
        output = {"x": np.zeros((len(arrays), p["max_frames"], 160), np.float32),
                  "relative": np.zeros((len(arrays), p["max_frames"]), np.float32),
                  "lengths": np.asarray([len(x) for x, _ in arrays], np.int64)}
        for i, (x, relative) in enumerate(arrays):
            output["x"][i, :len(x)] = x
            output["relative"][i, :len(x)] = relative
        np.savez(out / (name + ".npz"), **output)
        for suffix, values in (("features", features), ("labels", labels)):
            with (out / (name + "." + suffix + ".jsonl")).open("x") as stream:
                for value in values:
                    service.write_line(stream, value)
        report["cohorts"][name] = {"counts": dict(counts), "features": len(features), "all_sites": len(labels),
                                   "no_TAG_model_inputs": True}
        print(name, dict(counts), flush=True)
    for name in p["full_sources"]:
        report["cohorts"][name]["output_sha256"] = {suffix: service.sha(out / (name + suffix))
                                                   for suffix in (".npz", ".features.jsonl", ".labels.jsonl", ".responses.jsonl")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

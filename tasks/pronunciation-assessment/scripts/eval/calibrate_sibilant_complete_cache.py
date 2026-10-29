"""Freeze identical strict calibration on every protected CAL record.

Reuse only byte-identical runtime inputs and saved batch-one predictions from
the same model. Reproduce deterministic guards, score missing inputs singly,
then call the existing calibration algorithm. No test data or fitting is used.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import collections
import hashlib
import json
import shutil
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_context_rank import model, score_single
from train_sibilant_cv_support import calibrate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, expected in p["sha256"].items():
        if service.sha(path) != expected:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    prefix = p["complete_prefix"]
    data = dict(np.load(prefix + ".npz"))
    if set(data) != {"x", "lengths", "relative"}:
        raise ValueError("invalid inference fields")
    features = service.read_jsonl(prefix + ".features.jsonl")
    bank_features = {name: service.read_jsonl(value + ".features.jsonl") for name, value in p["banks"].items()}
    # No CAL labels are opened until all raw scores/guard checks are saved.
    report = {"scope": p["scope"], "inference_batch": 1, "test_scored": False, "arms": {}}
    for arm, spec in p["models"].items():
        artifact = torch.load(spec["artifact"], map_location="cpu")
        attestation = torch.load(spec["score_artifact"], map_location="cpu")
        for group in ("head", "last2"):
            if set(artifact[group]) != set(attestation[group]):
                raise AssertionError("cached score model parameter keys differ")
            for key, value in artifact[group].items():
                if not torch.equal(value, attestation[group][key]):
                    raise AssertionError("cached score model weights differ")
        del attestation
        cached = {name: dict(np.load(path)) for name, path in spec["scores"].items()}
        for name, scores in cached.items():
            if len(scores["available"]) != len(bank_features[name]):
                raise ValueError("cached prediction count differs")
        probabilities = np.zeros((len(features), 4), dtype=np.float32)
        available = np.zeros(len(features), dtype=bool)
        missing, reused_by_bank, guards, duplicates = [], collections.defaultdict(list), [], 0
        for i, feature in enumerate(features):
            sources = [s for s in feature["cache_sources"] if s["bank"] in cached]
            if not sources:
                missing.append(i)
                continue
            first = sources[0]
            for source in sources:
                f = bank_features[source["bank"]][source["index"]]
                if any(f[k] != feature[k] for k in ("uid", "q_runtime", "clip_sha256", "canonical", "baseline_pred")):
                    raise AssertionError("cached score identity differs")
                scores = cached[source["bank"]]
                old = cached[first["bank"]]
                np.testing.assert_array_equal(scores["probabilities"][source["index"]], old["probabilities"][first["index"]])
                if scores["available"][source["index"]] != old["available"][first["index"]]:
                    raise AssertionError("duplicate availability differs")
                duplicates += int(source != first)
            scores = cached[first["bank"]]
            probabilities[i] = scores["probabilities"][first["index"]]
            available[i] = scores["available"][first["index"]]
            reused_by_bank[first["bank"]].append(i)
        for name, indices in reused_by_bank.items():
            indices = sorted(indices, key=lambda i: hashlib.sha256((features[i]["uid"] + ":" + str(features[i]["q_runtime"])).encode()).hexdigest())
            guards.extend(indices[:p["guards_per_bank"]])
        encoder, head = model(p, artifact)
        guard_inputs = {key: value[guards] for key, value in data.items()}
        guard_ps, guard_av, guard_seconds = score_single(encoder, head, guard_inputs)
        np.testing.assert_array_equal(guard_ps, probabilities[guards])
        np.testing.assert_array_equal(guard_av, available[guards])
        seconds = 0.
        if missing:
            missing_inputs = {key: value[missing] for key, value in data.items()}
            ps, av, seconds = score_single(encoder, head, missing_inputs)
            probabilities[missing], available[missing] = ps, av
        np.savez(out / (arm + ".complete_cal.scores.npz"), probabilities=probabilities, available=available)
        labels = service.read_jsonl(prefix + ".labels.jsonl")
        thresholds, audit = calibrate(labels, features, probabilities, available)
        for key in thresholds:
            if key.endswith("/affricate"):
                thresholds[key] = None
        artifact["thresholds"] = thresholds
        torch.save(artifact, out / (arm + ".pt"))
        report["arms"][arm] = {"features": len(features), "sites_all": len(labels), "thresholds": thresholds,
                               "calibration": audit, "cached_predictions": len(features) - len(missing),
                               "new_predictions": len(missing), "duplicate_predictions_exact": duplicates,
                               "deterministic_guards_exact": len(guards), "guard_seconds": guard_seconds,
                               "new_score_seconds": seconds,
                               "artifact_sha256": service.sha(out / (arm + ".pt")),
                               "score_sha256": service.sha(out / (arm + ".complete_cal.scores.npz"))}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(arm, json.dumps(report["arms"][arm], ensure_ascii=False), flush=True)
        del encoder, head, artifact
        torch.cuda.empty_cache()
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

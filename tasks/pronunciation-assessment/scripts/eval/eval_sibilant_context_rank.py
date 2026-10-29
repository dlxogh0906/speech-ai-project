"""Matched batch-one calibration/evaluation of CE versus real-recording ranking.

Calibration reads CAL labels only. Evaluate requires both frozen calibration
artifacts, scores all declared test cohorts, and uses the existing API formatter.
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
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import restored_sites, verify_phones, detail
from train_sibilant_local_encoder import load_encoder, hidden, write
from train_sibilant_cv_support import calibrate
from mdd.sibilant_cv_support import SupportReadout, reliable_support
from mdd.sibilant_cv_policy import annotate


def score_single(encoder, head, data):
    if set(data) != {"x", "lengths", "relative"}:
        raise ValueError("inference accepts acoustic inputs and runtime position only")
    probabilities, availability = [], []
    device, tick = next(encoder.parameters()).device, time.monotonic()
    with torch.no_grad():
        for i in range(len(data["x"])):
            x = torch.as_tensor(data["x"][i:i + 1], device=device)
            lengths = torch.as_tensor(data["lengths"][i:i + 1], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(data["relative"][i:i + 1], device=device)
            logits, roles = head(hidden(encoder, x, mask), mask, relative)
            probabilities.append(logits.softmax(-1)[0].cpu().numpy())
            availability.append(bool(reliable_support(roles, mask)[0]))
            if (i + 1) % 1000 == 0:
                print("single requests scored", i + 1, "seconds", round(time.monotonic() - tick, 1), flush=True)
    torch.cuda.synchronize()
    return np.asarray(probabilities), np.asarray(availability), time.monotonic() - tick


def model(p, artifact):
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    head = SupportReadout(encoder.config.hidden_size, True).to(next(encoder.parameters()).device)
    head.load_state_dict(artifact["head"])
    return encoder, head


def inputs(dataset):
    data = dict(np.load(dataset["prefix"] + ".npz"))
    return data, service.read_jsonl(dataset["prefix"] + ".features.jsonl"), service.read_jsonl(dataset["prefix"] + ".labels.jsonl")


def calibration(p, out):
    report = {"scope": p["scope"], "inference_batch": 1, "test_scored": False, "arms": {}}
    for arm, path in p["artifacts"].items():
        artifact = torch.load(path, map_location="cpu")
        encoder, head = model(p, artifact)
        labels, features, probabilities, available = [], [], [], []
        cohorts = {}
        for name in p["calibration_cohorts"]:
            data, fs, ls = inputs(p["datasets"][name])
            ps, av, seconds = score_single(encoder, head, data)
            np.savez(out / (arm + "." + name + ".scores.npz"), probabilities=ps, available=av)
            offset = len(features)
            for row in ls:
                row = dict(row)
                if row.get("feature_index") is not None:
                    row["feature_index"] += offset
                labels.append(row)
            features.extend(fs)
            probabilities.extend(ps)
            available.extend(av)
            cohorts[name] = {"features": len(fs), "sites_all": len(ls), "seconds": seconds}
        thresholds, audit = calibrate(labels, features, np.asarray(probabilities), np.asarray(available))
        for key in thresholds:
            if key.endswith("/affricate"):
                thresholds[key] = None
        artifact["thresholds"] = thresholds
        torch.save(artifact, out / (arm + ".pt"))
        report["arms"][arm] = {"cohorts": cohorts, "thresholds": thresholds, "calibration": audit,
                               "artifact_sha256": service.sha(out / (arm + ".pt"))}
        print(arm, "calibration frozen", json.dumps(report["arms"][arm], ensure_ascii=False), flush=True)
        del encoder, head, artifact
        torch.cuda.empty_cache()
        write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


def evaluate(p, out, calibration_root):
    marker = json.loads((calibration_root / "finished.json").read_text())
    if marker["sha256"] != service.sha(calibration_root / "report.json"):
        raise ValueError("calibration incomplete/changed")
    fitted = json.loads((calibration_root / "report.json").read_text())
    for arm in p["artifacts"]:
        if service.sha(calibration_root / (arm + ".pt")) != fitted["arms"][arm]["artifact_sha256"]:
            raise ValueError("frozen artifact changed")
        artifact = torch.load(calibration_root / (arm + ".pt"), map_location="cpu")
        encoder, head = model(p, artifact)
        for name, dataset in p["datasets"].items():
            saved = calibration_root / (arm + "." + name + ".scores.npz")
            if saved.exists():
                value = np.load(saved)
                ps, av = value["probabilities"], value["available"]
            else:
                data = dict(np.load(dataset["prefix"] + ".npz"))
                ps, av, seconds = score_single(encoder, head, data)
                print(arm, name, "scored", len(ps), seconds, flush=True)
            np.savez(out / (arm + "." + name + ".scores.npz"), probabilities=ps, available=av)
        del encoder, head, artifact
        torch.cuda.empty_cache()
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "deployment": False, "inference_batch": 1, "arms": {}}
    for arm in p["artifacts"]:
        thresholds = fitted["arms"][arm]["thresholds"]
        results = {}
        for name, dataset in p["datasets"].items():
            prefix = dataset["prefix"]
            records = service.read_jsonl(prefix + ".responses.jsonl")
            features, labels = service.read_jsonl(prefix + ".features.jsonl"), service.read_jsonl(prefix + ".labels.jsonl")
            scores = np.load(out / (arm + "." + name + ".scores.npz"))
            by_uid, by_label = collections.defaultdict(dict), collections.defaultdict(list)
            for i, feature in enumerate(features):
                by_uid[feature["uid"]][feature["q_runtime"]] = {"probabilities": scores["probabilities"][i].tolist(), "available": bool(scores["available"][i])}
            for row in labels:
                by_label[row["uid"]].append(row)
            rows, counts = [], collections.Counter()
            with (out / (arm + "." + name + ".responses.jsonl")).open("x") as stream:
                for record in records:
                    row, base = record["row"], record["prior"]
                    candidate = dict(base)
                    if base["raw"] is not None and by_uid[row["uid"]]:
                        raw, phones = base["raw"], service.phonemes(base["body"])
                        sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                        if render(app, row, raw, sites) != base["body"]:
                            raise AssertionError("prior API did not reproduce")
                        sites, states = annotate(raw["canon"], sites, phones, by_uid[row["uid"]], thresholds, False)
                        counts.update(states.values())
                        if "confirmed" in states.values():
                            candidate["body"] = render(app, row, raw, sites)
                            verify_phones(phones, service.phonemes(candidate["body"]), sites)
                    views = {"original": observe(record["original"], row), "reference": observe(record["existing"], row),
                             "prior": observe(base, row), "candidate": observe(candidate, row)}
                    for label in by_label[row["uid"]]:
                        value = dict(label)
                        value.update({key: view[label["q"]] for key, view in views.items()})
                        rows.append(value)
                    service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
            with (out / (arm + "." + name + ".sites.jsonl")).open("x") as stream:
                for row in rows:
                    service.write_line(stream, row)
            results[name] = detail(rows, counts)
            print(arm, name, json.dumps(results[name]["versus_prior"], ensure_ascii=False), flush=True)
        report["arms"][arm] = {"thresholds": thresholds, "cohorts": results}
        write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    ap.add_argument("--calibration")
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(Path(a.protocol).read_text())
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    if a.stage == "calibrate":
        calibration(p, out)
    else:
        evaluate(p, out, Path(a.calibration))


if __name__ == "__main__":
    main()

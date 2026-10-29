"""Fit two predeclared linear temporal controls; freeze before response evaluation."""
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
import datetime
import json
import shutil

import numpy as np
import torch

import eval_sibilant_directional as service
from train_sibilant_hidden import load
from train_sibilant_transition import render, metrics
from eval_sibilant_margin_policy import paired_bootstrap
from audit_sibilant_service_coverage import audit as coverage_audit
from mdd.sibilant_hidden import zero_extra_thresholds
from mdd.sibilant_linear import features, predict_readout


def frozen_load(source, previous, protocol, split):
    if not (source / "finished.json").exists():
        raise ValueError("hidden extraction unfinished")
    if json.loads((source / "metadata.json").read_text())["limit"]:
        raise ValueError("smoke extraction cannot be used")
    for path, digest in protocol["hidden_sha256"][split].items():
        if service.sha(path) != digest:
            raise ValueError("frozen hidden observations changed")
    return load(source, previous, split)


def fit(protocol, out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    source = Path(protocol["hidden_features"])
    previous = json.loads((source / "metadata.json").read_text())["protocol"]
    data = {s: frozen_load(source, previous, protocol, s) for s in ("train", "cal")}
    if service.sha(protocol["projection_artifact"]) != protocol["projection_artifact_sha256"]:
        raise ValueError("previous training-only projection changed")
    projection = torch.load(protocol["projection_artifact"], map_location="cpu")["projection"]
    train = [r for r in data["train"][1] if r["feature_index"] is not None]
    indices = np.array([r["feature_index"] for r in train])
    targets = np.array([r["actual"] == "ㅆ" for r in train], np.int64)
    groups = collections.Counter((r["expected"], r["actual"]) for r in train)
    weights = np.array([len(train) / (4 * groups[(r["expected"], r["actual"])]) for r in train])
    if len(groups) != 4:
        raise ValueError("normal/error groups in both directions required")
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "projection.npz", **projection)
    report = {"frozen_utc": None, "test_read_during_fit": False, "arms": {}, "source_sha256": {}}
    for p in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_linear.py",
              ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_hidden.py"]:
        shutil.copy2(p, out / p.name)
        report["source_sha256"][str(p)] = service.sha(p)
    for arm in protocol["arms"]:
        x = {s: features(data[s][0], projection, temporal=arm == "h24_temporal_linear") for s in data}
        scaler = StandardScaler().fit(x["train"][indices])
        model = LogisticRegression(C=protocol["C"], solver="lbfgs", max_iter=1000, random_state=20261024)
        model.fit(scaler.transform(x["train"][indices]), targets, sample_weight=weights)
        if int(model.n_iter_[0]) >= 1000:
            raise RuntimeError("linear optimization failed to converge")
        artifact = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": model.coef_[0],
                    "intercept": model.intercept_[0]}
        probabilities = {s: predict_readout(x[s], artifact) for s in x}
        np.testing.assert_allclose(probabilities["cal"], model.predict_proba(scaler.transform(x["cal"]))[:, 1], rtol=0, atol=1e-12)
        thresholds, calibration = zero_extra_thresholds(data["cal"][1], data["cal"][2], probabilities["cal"])
        np.savez(out / (arm + ".npz"), **artifact)
        np.savez(out / (arm + ".fit_scores.npz"), **probabilities)
        report["arms"][arm] = {"thresholds": thresholds, "calibration": calibration,
                               "model_sha256": service.sha(out / (arm + ".npz")),
                               "iterations": int(model.n_iter_[0]), "coefficients": len(model.coef_[0]),
                               "train_accuracy_at_half": float(((probabilities["train"][indices] >= .5) == targets).mean())}
        print(arm, report["arms"][arm], flush=True)
    report["projection_sha256"] = service.sha(out / "projection.npz")
    report["frozen_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "fit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")


def evaluate(protocol, out):
    if (out / "report.json").exists():
        raise FileExistsError(out / "report.json")
    if json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("fitted protocol changed")
    fitted = json.loads((out / "fit.json").read_text())
    for path, digest in fitted["source_sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("fitted source changed")
    if service.sha(out / "projection.npz") != fitted["projection_sha256"]:
        raise ValueError("projection changed")
    projection = dict(np.load(out / "projection.npz"))
    source = Path(protocol["hidden_features"])
    previous = json.loads((source / "metadata.json").read_text())["protocol"]
    baseline = json.loads(Path(previous["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(baseline["source_sha256"], **baseline["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline changed")
    paths = {Path(p).name: p for p in baseline["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, baseline["base_ckpt"])
    report = {"scope": "Existing TAG, reused corpus, frozen service outputs; no deployment",
              "primary_arm": protocol["primary_arm"], "splits": {}}
    for split in ("cal", "test"):
        arrays, labels, feature_rows, responses = frozen_load(source, previous, protocol, split)
        report["splits"][split] = {}
        for arm in protocol["arms"]:
            if service.sha(out / (arm + ".npz")) != fitted["arms"][arm]["model_sha256"]:
                raise ValueError("fitted model changed")
            probabilities = predict_readout(features(arrays, projection, arm == "h24_temporal_linear"),
                                            dict(np.load(out / (arm + ".npz"))))
            if split == "cal":
                np.testing.assert_array_equal(probabilities, np.load(out / (arm + ".fit_scores.npz"))["cal"])
            np.save(out / (arm + "." + split + ".scores.npy"), probabilities)
            bodies, phones = render(app, responses, probabilities, fitted["arms"][arm]["thresholds"])
            rows = []
            for r in labels:
                row = dict(r)
                q = row["q_runtime"]
                row["candidate_phone"] = phones[r["uid"]][q] if q is not None and phones[r["uid"]] else None
                rows.append(row)
            for suffix, values in (("sites", rows), ("responses", bodies)):
                with (out / (arm + "." + split + "." + suffix + ".jsonl")).open("w") as stream:
                    for value in values:
                        service.write_line(stream, value)
            result = {"overall": metrics(rows), "service_coverage": coverage_audit(rows)}
            for field in ("following", "position", "item", "language"):
                result[field] = {str(key): metrics([r for r in rows if r[field] == key])
                                 for key in sorted({r[field] for r in rows}, key=str)}
            if split == "test":
                result["speaker_bootstrap"] = paired_bootstrap(rows, "speaker")
                result["item_bootstrap"] = paired_bootstrap(rows, "item")
            elif any(m["additional_fp"] for m in result["overall"].values()):
                raise AssertionError("zero additional calibration FP violated")
            report["splits"][split][arm] = result
            print(split, arm, result["overall"], flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    torch.set_num_threads(2)
    (fit if args.stage == "fit" else evaluate)(protocol, Path(args.out))


if __name__ == "__main__":
    main()

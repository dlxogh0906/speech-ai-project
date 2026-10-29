"""Frozen paired readouts for TAG-localization diagnosis, never a deployable score."""
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

import eval_sibilant_directional as service
from train_sibilant_transition import metrics, render
from eval_sibilant_margin_policy import paired_bootstrap
from audit_sibilant_service_coverage import audit as coverage_audit
from mdd.sibilant_hidden import zero_extra_thresholds
from mdd.sibilant_linear import predict_readout

ARM_COLUMN = {"runtime_matched_coverage": "runtime", "tag_localized_diagnostic": "localized"}


def read(protocol, source, split):
    if json.loads((source / "protocol.json").read_text()) != protocol:
        raise ValueError("extraction protocol changed")
    finished = json.loads((source / "finished.json").read_text())
    if service.sha(source / (split + ".npz")) != finished[split]["matrix_sha256"]:
        raise ValueError("extracted matrices changed")
    previous = json.loads(Path(protocol["linear_protocol"]).read_text())
    if service.sha(protocol["linear_protocol"]) != protocol["source_sha256"][protocol["linear_protocol"]]:
        raise ValueError("frozen prior protocol changed")
    metadata = json.loads((Path(previous["hidden_features"]) / "metadata.json").read_text())["protocol"]
    definition = metadata["cache_sources"][split]
    for path, digest in definition["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("original cached observations changed")
    cache = Path(definition["path"])
    arrays = dict(np.load(source / (split + ".npz")))
    labels = service.read_jsonl(cache / (split + ".labels.jsonl"))
    features = service.read_jsonl(cache / (split + ".features.jsonl"))
    responses = service.read_jsonl(cache / (split + ".responses.jsonl"))
    for row in labels:
        ix = row["feature_index"]
        row["diagnostic_available"] = ix is not None and bool(arrays["available"][ix])
        if not row["diagnostic_available"]:
            row["feature_index"] = None
    for record in responses:
        record["feature_indices"] = {q: ix for q, ix in record["feature_indices"].items() if arrays["available"][ix]}
    if len(features) != len(arrays["available"]):
        raise ValueError("diagnostic matrix count mismatch")
    return arrays, labels, features, responses, metadata


def fit(protocol, source, out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    data = {s: read(protocol, source, s) for s in ("train", "cal")}
    rows = [r for r in data["train"][1] if r["diagnostic_available"]]
    index = np.array([r["feature_index"] for r in rows])
    y = np.array([r["actual"] == "ㅆ" for r in rows], int)
    groups = collections.Counter((r["expected"], r["actual"]) for r in rows)
    if len(groups) != 4:
        raise ValueError("four training groups required")
    weights = np.array([len(rows) / (4 * groups[(r["expected"], r["actual"])]) for r in rows])
    for expected in ("ㅅ", "ㅆ"):
        if sum(r["diagnostic_available"] and r["expected"] == r["actual"] == expected for r in data["cal"][1]) < 100:
            raise ValueError("insufficient supported calibration normals")
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "TAG interval diagnostic; both arms share available sites, no test fitting",
              "test_read_during_fit": False, "training_groups": {str(k): v for k, v in groups.items()},
              "source_sha256": {}, "arms": {}}
    for p in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_linear.py",
              ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_hidden.py"]:
        report["source_sha256"][str(p)] = service.sha(p)
        shutil.copy2(p, out / p.name)
    for arm in protocol["arms"]:
        key = ARM_COLUMN[arm]
        x = {s: data[s][0][key] for s in data}
        scaler = StandardScaler().fit(x["train"][index])
        model = LogisticRegression(C=protocol["C"], solver="lbfgs", max_iter=1000, random_state=20261024)
        model.fit(scaler.transform(x["train"][index]), y, sample_weight=weights)
        if model.n_iter_[0] >= 1000:
            raise ValueError("linear optimizer did not converge")
        artifact = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": model.coef_[0], "intercept": model.intercept_[0]}
        scores = {s: predict_readout(x[s], artifact) for s in data}
        np.testing.assert_allclose(scores["cal"], model.predict_proba(scaler.transform(x["cal"]))[:, 1], rtol=0, atol=1e-12)
        thresholds, calibration = zero_extra_thresholds(data["cal"][1], data["cal"][2], scores["cal"])
        np.savez(out / (arm + ".npz"), **artifact)
        np.savez(out / (arm + ".fit_scores.npz"), **scores)
        report["arms"][arm] = {"thresholds": thresholds, "calibration": calibration,
                               "model_sha256": service.sha(out / (arm + ".npz")), "iterations": int(model.n_iter_[0]),
                               "train_accuracy_at_half": float(((scores["train"][index] >= .5) == y).mean())}
        print(arm, report["arms"][arm], flush=True)
    report["frozen_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "fit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")


def evaluate(protocol, source, out):
    from sklearn.metrics import roc_auc_score
    if (out / "report.json").exists():
        raise FileExistsError("evaluation exists")
    if json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("fitted protocol changed")
    fitted = json.loads((out / "fit.json").read_text())
    for path, digest in fitted["source_sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("fitted source changed")
    previous = json.loads(Path(protocol["linear_protocol"]).read_text())
    metadata = json.loads((Path(previous["hidden_features"]) / "metadata.json").read_text())["protocol"]
    baseline = json.loads(Path(metadata["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(baseline["source_sha256"], **baseline["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source/scorer changed")
    paths = {Path(p).name: p for p in baseline["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, baseline["base_ckpt"])
    report = {"scope": "TAG-localization diagnostic only, actual formatter comparison, reused corpus; no deployable claim", "splits": {}}
    for split in ("cal", "test"):
        arrays, labels, features, responses, _ = read(protocol, source, split)
        report["splits"][split] = {}
        for arm in protocol["arms"]:
            path = out / (arm + ".npz")
            if service.sha(path) != fitted["arms"][arm]["model_sha256"]:
                raise ValueError("saved model changed")
            scores = predict_readout(arrays[ARM_COLUMN[arm]], dict(np.load(path)))
            if split == "cal":
                np.testing.assert_array_equal(scores, np.load(out / (arm + ".fit_scores.npz"))["cal"])
            np.save(out / (arm + "." + split + ".scores.npy"), scores)
            bodies, phones = render(app, responses, scores, fitted["arms"][arm]["thresholds"])
            rows = []
            for label in labels:
                row = dict(label)
                q = row["q_runtime"]
                row["candidate_phone"] = phones[row["uid"]][q] if q is not None and phones[row["uid"]] else None
                rows.append(row)
            detail = {"overall": metrics(rows), "service_coverage": coverage_audit(rows), "diagnostic_discrimination": {}}
            for expected in ("ㅅ", "ㅆ"):
                eligible = [r for r in rows if r["diagnostic_available"] and r["expected"] == expected]
                target = np.array([r["actual"] != expected for r in eligible])
                probability = np.array([scores[r["feature_index"]] if expected == "ㅅ" else 1 - scores[r["feature_index"]] for r in eligible])
                detail["diagnostic_discrimination"][expected] = {"normals": int((~target).sum()), "errors": int(target.sum()),
                    "error_auc": float(roc_auc_score(target, probability)) if target.any() and not target.all() else None}
            for field in ("following", "position", "item", "language", "diagnostic_available"):
                detail[field] = {str(k): metrics([r for r in rows if r[field] == k]) for k in sorted({r[field] for r in rows}, key=str)}
            if split == "test":
                detail["speaker_bootstrap"] = paired_bootstrap(rows, "speaker")
                detail["item_bootstrap"] = paired_bootstrap(rows, "item")
            elif any(m["additional_fp"] for m in detail["overall"].values()):
                raise AssertionError("additional calibration FP")
            for suffix, values in [("responses", bodies), ("sites", rows)]:
                with (out / (arm + "." + split + "." + suffix + ".jsonl")).open("w") as stream:
                    for value in values:
                        service.write_line(stream, value)
            report["splits"][split][arm] = detail
            print(split, arm, detail["overall"], detail["diagnostic_discrimination"], flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    (fit if args.stage == "fit" else evaluate)(protocol, Path(args.features), Path(args.out))


if __name__ == "__main__":
    main()

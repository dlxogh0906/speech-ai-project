"""Fit frozen layer/temporal controls on train/cal, then separately evaluate saved heads."""
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
import time

import numpy as np
import torch

import eval_sibilant_directional as service
from eval_sibilant_margin_policy import paired_bootstrap
from train_sibilant_transition import metrics, render, mapped, flag
from mdd.sibilant_hidden import fit_projection, project_sequences, zero_extra_thresholds
from mdd.sibilant_transition_head import TransitionHead, predict
from mdd.sibilant_transition import PAIR


def load(source, protocol, split):
    definition = protocol["cache_sources"][split]
    for path, digest in definition["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen cached input changed: " + path)
    cache = Path(definition["path"])
    arrays = dict(np.load(cache / (split + ".npz")))
    features = service.read_jsonl(cache / (split + ".features.jsonl"))
    labels = service.read_jsonl(cache / (split + ".labels.jsonl"))
    responses = service.read_jsonl(cache / (split + ".responses.jsonl"))
    if not np.load(source / (split + ".filled.npy")).all():
        raise ValueError("incomplete hidden extraction")
    arrays.update({"h%d" % layer: np.load(source / (split + ".h%d.npy" % layer), mmap_mode="r")
                   for layer in protocol["layers"]})
    if any(len(values) != len(features) for values in arrays.values()):
        raise ValueError("feature array count mismatch")
    return arrays, labels, features, responses


def inputs(arrays, arm, projection):
    if arm == "ctc_sequence":
        seq = arrays["posterior"].astype(np.float32)
    else:
        seq = project_sequences(arrays[arm.split("_")[0]], arrays["mask"], projection,
                                pooled=arm.endswith("pooled"))
    return (torch.from_numpy(seq), torch.from_numpy(arrays["mask"].astype(bool)),
            torch.from_numpy(arrays["context"].astype(np.float32)))


def fit(source, out, meta):
    protocol = meta["protocol"]
    out.mkdir(parents=True, exist_ok=False)
    data = {s: load(source, protocol, s) for s in ("train", "cal")}
    train = [r for r in data["train"][1] if r["feature_index"] is not None and r["actual"] in PAIR]
    indices = torch.tensor([r["feature_index"] for r in train], dtype=torch.long)
    targets = torch.tensor([int(r["actual"] == "ㅆ") for r in train], dtype=torch.long)
    groups = collections.Counter((r["expected"], r["actual"]) for r in train)
    if len(groups) != 4:
        raise ValueError("both directions require normal and error training observations")
    weights = torch.tensor([len(train) / (len(groups) * groups[(r["expected"], r["actual"])]) for r in train])
    cfg = protocol["model"]
    projections = {"h%d" % layer: fit_projection(data["train"][0]["h%d" % layer][indices.numpy()],
                    data["train"][0]["mask"][indices.numpy()], protocol["projection"]["dimensions"], cfg["seed"])
                   for layer in protocol["layers"]}
    fit_report = {"protocol_sha256": meta["protocol_sha256"], "test_read_during_fit": False,
                  "training_groups": {str(k): v for k, v in groups.items()}, "train_available": len(train),
                  "arms": {}, "source_sha256": {}}
    for path in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_hidden.py",
                 ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_transition_head.py"]:
        shutil.copy2(path, out / path.name)
        fit_report["source_sha256"][str(path)] = service.sha(path)
    for arm in protocol["arms"]:
        started = time.monotonic()
        torch.manual_seed(cfg["seed"])
        rng = np.random.RandomState(cfg["seed"])
        projection = projections.get(arm.split("_")[0])
        tensors = {s: inputs(data[s][0], arm, projection) for s in data}
        model = TransitionHead(tensors["train"][0].shape[-1], tensors["train"][2].shape[-1], cfg["width"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        losses = []
        for epoch in range(cfg["epochs"]):
            model.train()
            order, total = rng.permutation(len(train)), 0.
            for start in range(0, len(order), cfg["batch_size"]):
                j = torch.tensor(order[start:start + cfg["batch_size"]], dtype=torch.long)
                logits = model(*(v[indices[j]] for v in tensors["train"]))
                loss = (torch.nn.functional.cross_entropy(logits, targets[j], reduction="none") * weights[j]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(j)
            losses.append(total / len(train))
        probabilities = {s: predict(model, tensors[s]) for s in data}
        thresholds, audit = zero_extra_thresholds(data["cal"][1], data["cal"][2], probabilities["cal"])
        artifact = {"state_dict": model.state_dict(), "input_dim": tensors["train"][0].shape[-1],
                    "context_dim": tensors["train"][2].shape[-1], "width": cfg["width"], "arm": arm,
                    "projection": projection, "thresholds": thresholds, "protocol_sha256": meta["protocol_sha256"]}
        torch.save(artifact, out / (arm + ".pt"))
        np.savez(out / (arm + ".fit_probabilities.npz"), **probabilities)
        fit_report["arms"][arm] = {"thresholds": thresholds, "calibration": audit, "train_loss": losses,
                                  "parameters": sum(p.numel() for p in model.parameters()),
                                  "fit_predict_seconds": time.monotonic() - started,
                                  "model_sha256": service.sha(out / (arm + ".pt"))}
        print("fit", arm, "loss", losses[-1], "thresholds", thresholds, flush=True)
    fit_report["frozen_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "fit.json").write_text(json.dumps(fit_report, ensure_ascii=False, indent=2) + "\n")


def evaluate(source, out, meta):
    if (out / "report.json").exists():
        raise FileExistsError(out / "report.json")
    protocol = meta["protocol"]
    fitted = json.loads((out / "fit.json").read_text())
    if fitted["protocol_sha256"] != meta["protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for path, digest in fitted["source_sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("fitted source changed: " + path)
    previous = json.loads(Path(protocol["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(previous["source_sha256"], **previous["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source/scorer changed: " + path)
    paths = {Path(p).name: p for p in previous["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, previous["base_ckpt"])
    report = {"scope": "existing TAG; reused corpus; frozen GPU API formatter; no production activation",
              "protocol_sha256": meta["protocol_sha256"], "fit_sha256": service.sha(out / "fit.json"),
              "primary_arm": protocol["primary_arm"], "splits": {}}
    for split in ("cal", "test"):
        arrays, labels, features, responses = load(source, protocol, split)
        report["splits"][split] = {}
        for arm in protocol["arms"]:
            if service.sha(out / (arm + ".pt")) != fitted["arms"][arm]["model_sha256"]:
                raise ValueError("saved model changed")
            artifact = torch.load(out / (arm + ".pt"), map_location="cpu")
            model = TransitionHead(artifact["input_dim"], artifact["context_dim"], artifact["width"])
            model.load_state_dict(artifact["state_dict"])
            probabilities = predict(model, inputs(arrays, arm, artifact["projection"]))
            if split == "cal":
                saved = np.load(out / (arm + ".fit_probabilities.npz"))["cal"]
                np.testing.assert_array_equal(probabilities, saved)
            np.save(out / (arm + "." + split + ".probabilities.npy"), probabilities)
            bodies, phones = render(app, responses, probabilities, artifact["thresholds"])
            rows = []
            for r in labels:
                row = dict(r)
                q = r["q_runtime"]
                row["candidate_phone"] = phones[r["uid"]][q] if q is not None and phones[r["uid"]] else None
                rows.append(row)
            for suffix, values in (("responses", bodies), ("sites", rows)):
                with (out / (arm + "." + split + "." + suffix + ".jsonl")).open("w") as stream:
                    for value in values:
                        service.write_line(stream, value)
            result = {"overall": metrics(rows), "rescued_original_predictions": {}}
            for expected in PAIR:
                rescued = [r for r in rows if r["expected"] == expected and r["actual"] == PAIR[expected]
                           and mapped(r) and flag(r["candidate_phone"])
                           and r["candidate_phone"]["actual_jamo"] == r["actual"]
                           and not (flag(r["baseline_phone"]) and r["baseline_phone"]["actual_jamo"] == r["actual"])]
                result["rescued_original_predictions"][expected] = dict(collections.Counter(
                    (r["baseline_phone"] or {}).get("actual_jamo", "missing") for r in rescued))
            for field in ("following", "position", "item", "language"):
                result[field] = {str(key): metrics([r for r in rows if r[field] == key])
                                 for key in sorted({r[field] for r in rows}, key=str)}
            if split == "test":
                result["speaker_bootstrap"] = paired_bootstrap(rows, "speaker")
                result["item_bootstrap"] = paired_bootstrap(rows, "item")
            elif any(m["additional_fp"] for m in result["overall"].values()):
                raise AssertionError("actual formatter violates zero-extra calibration FP")
            report["splits"][split][arm] = result
            print(split, arm, json.dumps(result["overall"], ensure_ascii=False), flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    args = parser.parse_args()
    source, out = Path(args.features), Path(args.out)
    if not (source / "finished.json").exists():
        raise ValueError("hidden extraction unfinished")
    meta = json.loads((source / "metadata.json").read_text())
    if meta["limit"]:
        raise ValueError("smoke subset cannot be used for training/evaluation")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (fit if args.stage == "fit" else evaluate)(source, out, meta)


if __name__ == "__main__":
    main()

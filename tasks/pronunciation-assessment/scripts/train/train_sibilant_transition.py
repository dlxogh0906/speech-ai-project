"""Train fixed temporal heads, calibrate on cal only, compare actual API bodies."""
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
import math
import shutil
import time

import numpy as np
import torch

import eval_sibilant_directional as service
from mdd.sibilant_transition import PAIR, annotate_probabilities, fit_acoustic_normalization
from mdd.sibilant_transition_head import TransitionHead, input_tensors, predict


def mapped(row):
    return row["state"] in ("available", "features_unavailable")


def flag(phone):
    return bool(phone and phone.get("errorType"))


def eligibility(row, features):
    ix = row.get("feature_index")
    return ix is not None and features[ix]["baseline_pred"] in PAIR


def calibrate(labels, features, probabilities, target_fpr=.01):
    """Fixed >= rule, strict ties, and an explicit disabled state.

    Budget is computed on canonical-matched accepted normals, never inflated
    with input rejections or mapping failures. Full denominators are also
    retained in evaluation; no claim is made about unknown mappings.
    """
    thresholds, audit = {}, {}
    for expected in PAIR:
        normals = [r for r in labels if r["expected"] == expected and r["actual"] == expected and mapped(r)]
        old_fp = sum(flag(r["baseline_phone"]) for r in normals)
        budget = math.floor(target_fpr * len(normals)) - old_fp
        scores = [float(probabilities[r["feature_index"]]) if expected == "ㅅ" else
                  1 - float(probabilities[r["feature_index"]]) for r in normals
                  if eligibility(r, features) and not flag(r["baseline_phone"])]
        threshold = None
        if normals and scores and budget >= 0:
            descending = sorted(scores, reverse=True)
            threshold = .5 if len(descending) <= budget else max(.5, float(np.nextafter(descending[budget], np.inf)))
            if threshold > 1:
                threshold = None
        new_fp = sum(s >= threshold for s in scores) if threshold is not None else 0
        if threshold is not None and old_fp + new_fp > math.floor(target_fpr * len(normals)):
            raise AssertionError("calibration budget exceeded")
        thresholds[expected] = threshold
        audit[expected] = {"normal_matched_accepted": len(normals), "original_fp": old_fp,
                           "additional_fp_budget": budget, "additional_fp": new_fp,
                           "threshold": threshold,
                           "disabled_reason": "baseline_over_budget_or_no_eligible_calibration" if threshold is None else None}
    return thresholds, audit


def metrics(rows):
    result = {}
    for token in PAIR:
        subset = [r for r in rows if r["expected"] == token]
        normals = [r for r in subset if r["actual"] == token]
        errors = [r for r in subset if r["actual"] == PAIR[token]]
        def true_positive(r, key):
            p = r.get(key)
            return mapped(r) and flag(p) and p.get("actual_jamo") == r["actual"]
        def false_positive(r, key):
            return mapped(r) and flag(r.get(key))
        btp, ctp = [sum(true_positive(r, key) for r in errors) for key in ("baseline_phone", "candidate_phone")]
        bfp, cfp = [sum(false_positive(r, key) for r in normals) for key in ("baseline_phone", "candidate_phone")]
        normal_mapped = sum(mapped(r) for r in normals)
        result[token + "→" + PAIR[token]] = {
            "errors_all": len(errors), "normals_all": len(normals),
            "errors_matched_accepted": sum(mapped(r) for r in errors), "normals_matched_accepted": normal_mapped,
            "baseline_tp": btp, "candidate_tp": ctp, "baseline_fp": bfp, "candidate_fp": cfp,
            "baseline_recall_all": btp / len(errors) if errors else None,
            "candidate_recall_all": ctp / len(errors) if errors else None,
            "baseline_fpr_matched_accepted": bfp / normal_mapped if normal_mapped else None,
            "candidate_fpr_matched_accepted": cfp / normal_mapped if normal_mapped else None,
            "additional_tp": sum(true_positive(r, "candidate_phone") and not true_positive(r, "baseline_phone") for r in errors),
            "lost_tp": sum(true_positive(r, "baseline_phone") and not true_positive(r, "candidate_phone") for r in errors),
            "additional_fp": sum(false_positive(r, "candidate_phone") and not false_positive(r, "baseline_phone") for r in normals),
            "status_counts": dict(collections.Counter(r["state"] for r in subset)),
            "normal_unmapped_or_rejected": len(normals) - normal_mapped}
    return result


def render(app, responses, probabilities, thresholds):
    import mdd.speechsuper as formatter
    output, phones_by_uid = [], {}
    for record in responses:
        raw = record["raw"]
        if raw is None:
            body = None
        else:
            p = {int(q): float(probabilities[ix]) for q, ix in record["feature_indices"].items()}
            sites = annotate_probabilities(raw["canon"], raw["sites"], p, thresholds)
            before = formatter.SZ_RULE
            try:
                formatter.SZ_RULE = True
                body = service.response(app, record["row"], raw, sites)
            finally:
                formatter.SZ_RULE = before
            old_ph, new_ph = service.phonemes(record["body"]), service.phonemes(body)
            for q, (old, new) in enumerate(zip(old_ph, new_ph)):
                if "sz" not in sites[q] and old != new:
                    raise AssertionError("unannotated phone changed")
                if old.get("pronunciation") != new.get("pronunciation"):
                    raise AssertionError("existing phone score changed")
                if old.get("errorType") and not new.get("errorType"):
                    raise AssertionError("original error removed")
        phones_by_uid[record["uid"]] = service.phonemes(body) if body else []
        output.append({"uid": record["uid"], "rejected": record["rejected"], "body": body})
    return output, phones_by_uid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.features), Path(args.out)
    if not (source / "finished.json").exists():
        raise ValueError("feature extraction is not complete")
    meta = json.loads((source / "metadata.json").read_text())
    if meta["limit"]:
        raise ValueError("cannot train frozen pilot on smoke subset")
    protocol = meta["protocol"]
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = {s: dict(np.load(source / (s + ".npz"))) for s in ("train", "cal", "test")}
    labels = {s: service.read_jsonl(source / (s + ".labels.jsonl")) for s in data}
    features = {s: service.read_jsonl(source / (s + ".features.jsonl")) for s in data}
    train = [r for r in labels["train"] if r["feature_index"] is not None and r["actual"] in PAIR]
    indices = torch.tensor([r["feature_index"] for r in train], dtype=torch.long)
    targets = torch.tensor([int(r["actual"] == "ㅆ") for r in train], dtype=torch.long)
    groups = collections.Counter((r["expected"], r["actual"]) for r in train)
    weights = torch.tensor([len(train) / (len(groups) * groups[(r["expected"], r["actual"])]) for r in train])
    if len(groups) != 4:
        raise ValueError("both directions need normal and error examples")
    normalization = fit_acoustic_normalization([data["train"]["acoustic"][r["feature_index"]] for r in train])
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    snapshots = out / "source_snapshots"
    snapshots.mkdir()
    for path in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_transition_head.py",
                 ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_transition.py"]:
        shutil.copy2(path, snapshots / path.name)
    fitted = {}
    for arm in protocol["arms"]:
        torch.manual_seed(protocol["model"]["seed"])
        rng = np.random.RandomState(protocol["model"]["seed"])
        inputs = {s: input_tensors(data[s], normalization, acoustic=arm == "ctc_acoustic_sequence") for s in data}
        model = TransitionHead(inputs["train"][0].shape[-1], inputs["train"][2].shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        losses, started = [], time.monotonic()
        for epoch in range(40):
            model.train()
            order = rng.permutation(len(train))
            loss_sum = 0.
            for start in range(0, len(order), 64):
                j = torch.tensor(order[start:start + 64], dtype=torch.long)
                ix = indices[j]
                logits = model(*(t[ix] for t in inputs["train"]))
                loss = (torch.nn.functional.cross_entropy(logits, targets[j], reduction="none") * weights[j]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(j)
            losses.append(loss_sum / len(train))
        probabilities = {s: predict(model, inputs[s]) for s in data}
        thresholds, calibration = calibrate(labels["cal"], features["cal"], probabilities["cal"])
        artifact = {"state_dict": model.state_dict(), "input_dim": inputs["train"][0].shape[-1],
                    "context_dim": inputs["train"][2].shape[-1], "arm": arm, "normalization": normalization,
                    "thresholds": thresholds, "protocol_sha256": meta["protocol_sha256"]}
        torch.save(artifact, out / (arm + ".pt"))
        np.savez(out / (arm + ".probabilities.npz"), **probabilities)
        fitted[arm] = {"probabilities": probabilities, "thresholds": thresholds, "calibration": calibration,
                       "train_loss": losses, "parameters": sum(p.numel() for p in model.parameters()),
                       "train_predict_seconds": time.monotonic() - started}
        print(arm, "groups", dict(groups), "loss", losses[-1], "thresholds", thresholds, flush=True)
    previous = meta["service_metadata"]
    paths = {Path(p).name: p for p in previous["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, previous["base_ckpt"])
    report = {"scope": "Existing TAG, reused exploratory corpus; exact cached baseline inputs and current formatter; no deployment",
              "protocol_sha256": meta["protocol_sha256"], "train_available": len(train),
              "training_groups": {str(k): v for k, v in groups.items()}, "arms": {}}
    for arm, value in fitted.items():
        arm_report = {k: v for k, v in value.items() if k != "probabilities"}
        for split in ("cal", "test"):
            responses = service.read_jsonl(source / (split + ".responses.jsonl"))
            output, phones = render(app, responses, value["probabilities"][split], value["thresholds"])
            with (out / (arm + "." + split + ".responses.jsonl")).open("w") as stream:
                for record in output:
                    service.write_line(stream, record)
            rows = []
            for r in labels[split]:
                row = dict(r)
                q = r["q_runtime"]
                row["candidate_phone"] = phones[r["uid"]][q] if q is not None and phones[r["uid"]] else None
                rows.append(row)
            with (out / (arm + "." + split + ".sites.jsonl")).open("w") as stream:
                for row in rows:
                    service.write_line(stream, row)
            arm_report[split] = {"overall": metrics(rows)}
            for field in ("following", "position", "item", "language"):
                arm_report[split][field] = {str(key): metrics([r for r in rows if r[field] == key])
                                          for key in sorted({r[field] for r in rows}, key=str)}
        report["arms"][arm] = arm_report
        print(arm, json.dumps(arm_report["test"]["overall"], ensure_ascii=False), flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

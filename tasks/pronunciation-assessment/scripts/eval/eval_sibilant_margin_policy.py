"""Freeze calibration policies, then evaluate expanded response-level outcomes.

The calibrate stage never opens test labels, features, or outputs. Evaluation
reads the written policies and verifies their calibration input hashes.
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
import datetime
import json
import shutil

import numpy as np

import eval_sibilant_directional as service
from train_sibilant_transition import metrics, mapped, flag
from mdd.sibilant_margin_policy import fit_zero_extra_policies, apply_margin_policy, score_ss
from mdd.sibilant_transition import PAIR


def load_split(source, split):
    responses = service.read_jsonl(source / (split + ".responses.jsonl"))
    by_uid = {r["uid"]: r for r in responses}
    if len(by_uid) != len(responses):
        raise ValueError("duplicate response UID")
    features = service.read_jsonl(source / (split + ".features.jsonl"))
    if [f["index"] for f in features] != list(range(len(features))):
        raise ValueError("noncontiguous feature indices")
    labels = service.read_jsonl(source / (split + ".labels.jsonl"))
    for row in labels:
        response = by_uid[row["uid"]]
        canon = response["raw"]["canon"] if response["raw"] else []
        q = row["q_runtime"]
        row["runtime_following"] = canon[q + 1] if q is not None and q + 1 < len(canon) else None
    return responses, features, labels


def hashes(source, split):
    return {suffix: service.sha(source / (split + "." + suffix + ".jsonl"))
            for suffix in ("responses", "features", "labels")}


def calibrate(source, out):
    summary = json.loads((source / "summary.json").read_text())
    if "cal" not in summary:
        raise ValueError("cal extraction incomplete")
    meta = json.loads((source / "metadata.json").read_text())
    protocol = meta["protocol"]
    if protocol["primary_arm"] != "vowel_zero_extra_calibration_fp":
        raise ValueError("wrong frozen experiment")
    _, features, labels = load_split(source, "cal")
    policies, audit = fit_zero_extra_policies(labels, features, min_normals=100)
    policies["original_service"] = {"mode": "global", "thresholds": {"ㅅ": None, "ㅆ": None}}
    policies["fixed_previous_margin"] = {"mode": "global", "thresholds": protocol["fixed_previous_thresholds"]}
    out.mkdir(parents=True, exist_ok=False)
    payload = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "protocol_sha256": meta["protocol_sha256"], "cal_input_sha256": hashes(source, "cal"),
               "test_read_during_calibration": False, "policies": policies, "audit": audit}
    (out / "policies.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for f in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_margin_policy.py"):
        shutil.copy2(f, out / ("calibration_" + f.name))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def paired_bootstrap(rows, cluster_key, seed=20261024, repetitions=1000):
    """Separate speaker/item cluster intervals; not joint two-way uncertainty."""
    output = {}
    rng = np.random.RandomState(seed)
    for token in PAIR:
        groups = collections.defaultdict(lambda: np.zeros(4, np.float64))
        for row in rows:
            if row["expected"] != token:
                continue
            g = groups[str(row[cluster_key])]
            old, new = row.get("baseline_phone"), row.get("candidate_phone")
            if row["actual"] == PAIR[token]:
                g[0] += 1
                g[1] += (int(mapped(row) and flag(new) and new.get("actual_jamo") == row["actual"]) -
                         int(mapped(row) and flag(old) and old.get("actual_jamo") == row["actual"]))
            elif row["actual"] == token and mapped(row):
                g[2] += 1
                g[3] += int(flag(new)) - int(flag(old))
        matrix = np.asarray(list(groups.values()))
        deltas = [[], []]
        if len(matrix):
            for _ in range(repetitions):
                counts = matrix[rng.randint(0, len(matrix), len(matrix))].sum(0)
                for j, den in enumerate((0, 2)):
                    if counts[den]:
                        deltas[j].append(counts[den + 1] / counts[den])
        output[token + "→" + PAIR[token]] = {
            "clusters": len(groups),
            "recall_delta_95": np.quantile(deltas[0], [.025, .975]).tolist() if deltas[0] else None,
            "fpr_delta_95": np.quantile(deltas[1], [.025, .975]).tolist() if deltas[1] else None}
    return output


def render_policy(app, responses, features, policy):
    import mdd.speechsuper as formatter
    scores = [score_ss(f["token_margin"]) for f in features]
    bodies, by_uid, counts = [], {}, collections.Counter()
    for record in responses:
        raw = record["raw"]
        if raw is None:
            body = None
            counts["input_rejected"] += 1
        else:
            values = {int(q): scores[ix] for q, ix in record["feature_indices"].items()}
            sites = apply_margin_policy(raw["canon"], raw["sites"], values, policy)
            before = formatter.SZ_RULE
            try:
                formatter.SZ_RULE = True
                body = service.response(app, record["row"], raw, sites)
            finally:
                formatter.SZ_RULE = before
            old_ph, new_ph = service.phonemes(record["body"]), service.phonemes(body)
            if len(old_ph) != len(new_ph) or len(old_ph) != len(sites):
                raise AssertionError("response target count changed")
            for q, (old, new) in enumerate(zip(old_ph, new_ph)):
                if "sz" not in sites[q] and old != new:
                    raise AssertionError("non-target phone changed")
                if old.get("pronunciation") != new.get("pronunciation"):
                    raise AssertionError("existing phone score changed")
                if old.get("errorType") and not new.get("errorType"):
                    raise AssertionError("original alert removed")
                counts["annotated_sites"] += "sz" in sites[q]
            counts["unchanged_entire_body"] += body == record["body"]
            counts["accepted"] += 1
        bodies.append({"uid": record["uid"], "rejected": record["rejected"], "body": body})
        by_uid[record["uid"]] = service.phonemes(body) if body else []
    return bodies, by_uid, dict(counts)


def evaluate(source, out):
    if not (source / "finished.json").exists():
        raise ValueError("full extraction incomplete")
    if (out / "report.json").exists():
        raise FileExistsError(out / "report.json")
    frozen = json.loads((out / "policies.json").read_text())
    meta = json.loads((source / "metadata.json").read_text())
    if frozen["cal_input_sha256"] != hashes(source, "cal") or frozen["protocol_sha256"] != meta["protocol_sha256"]:
        raise ValueError("frozen calibration inputs changed")
    previous = meta["service_metadata"]
    for path, digest in dict(previous["source_sha256"], **previous["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source/scorer changed: " + path)
    paths = {Path(p).name: p for p in previous["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, previous["base_ckpt"])
    report = {"protocol_sha256": meta["protocol_sha256"], "policies_sha256": service.sha(out / "policies.json"),
              "scope": "Existing TAG; source-matched GPU service formatter; reused corpus, no remote deployment",
              "primary_arm": meta["protocol"]["primary_arm"], "splits": {}}
    for split in ("cal", "test"):
        responses, features, labels = load_split(source, split)
        split_report = {"input_sha256": hashes(source, split), "arms": {}}
        for arm in meta["protocol"]["arms"]:
            bodies, phones, regression = render_policy(app, responses, features, frozen["policies"][arm])
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
            result = {"overall": metrics(rows), "regression": regression}
            for field in ("following", "runtime_following", "position", "item", "language"):
                result[field] = {str(key): metrics([r for r in rows if r[field] == key])
                                 for key in sorted({r[field] for r in rows}, key=str)}
            if split == "test":
                result["speaker_bootstrap"] = paired_bootstrap(rows, "speaker")
                result["item_bootstrap"] = paired_bootstrap(rows, "item")
            if split == "cal" and arm in ("global_zero_extra_calibration_fp", "vowel_zero_extra_calibration_fp"):
                if any(m["additional_fp"] for m in result["overall"].values()):
                    raise AssertionError("actual formatter violates zero-extra-FP calibration")
            split_report["arms"][arm] = result
            print(split, arm, json.dumps(result["overall"], ensure_ascii=False), flush=True)
        report["splits"][split] = split_report
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    (calibrate if args.stage == "calibrate" else evaluate)(Path(args.features), Path(args.out))


if __name__ == "__main__":
    main()

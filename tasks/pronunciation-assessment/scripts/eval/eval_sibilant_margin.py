"""Confirm the selected symmetric-margin policy without fitting thresholds."""
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
import json
import shutil

import numpy as np

import eval_sibilant_directional as service
from train_sibilant_transition import render, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.features), Path(args.out)
    if not (source / "finished.json").exists():
        raise ValueError("feature extraction incomplete")
    meta = json.loads((source / "metadata.json").read_text())
    selection = meta["protocol"]["confirmation_selection"]
    if selection["score"] != "token_margin" or not selection["no_new_fit"]:
        raise ValueError("wrong fixed confirmation protocol")
    out.mkdir(parents=True, exist_ok=False)
    thresholds = selection["thresholds"]
    previous = meta["service_metadata"]
    paths = {Path(p).name: p for p in previous["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, previous["base_ckpt"])
    report = {"selection": selection, "protocol_sha256": meta["protocol_sha256"], "splits": {}}
    for split in meta["protocol"]["data"]:
        features = service.read_jsonl(source / (split + ".features.jsonl"))
        labels = service.read_jsonl(source / (split + ".labels.jsonl"))
        responses = service.read_jsonl(source / (split + ".responses.jsonl"))
        probabilities = 1 / (1 + np.exp(-np.clip([r["token_margin"] for r in features], -700, 700)))
        output, phones = render(app, responses, probabilities, thresholds)
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for record in output:
                service.write_line(stream, record)
        rows = []
        for r in labels:
            row = dict(r)
            q = r["q_runtime"]
            row["candidate_phone"] = phones[r["uid"]][q] if q is not None and phones[r["uid"]] else None
            rows.append(row)
        with (out / (split + ".sites.jsonl")).open("w") as stream:
            for row in rows:
                service.write_line(stream, row)
        result = {"overall": metrics(rows)}
        for field in ("following", "position", "item", "language"):
            result[field] = {str(key): metrics([r for r in rows if r[field] == key])
                             for key in sorted({r[field] for r in rows}, key=str)}
        report["splits"][split] = result
        print(split, json.dumps(result["overall"], ensure_ascii=False), flush=True)
    for file in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_margin.py"):
        shutil.copy2(file, out / file.name)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "policy.json").write_text(json.dumps({"kind": "symmetric_ctc_margin", "thresholds": thresholds,
                                                "protocol_sha256": meta["protocol_sha256"], "enabled_in_service": False},
                                               ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

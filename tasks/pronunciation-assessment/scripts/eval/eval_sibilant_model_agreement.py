"""Fixed primary thresholds plus actual-class/support confirmation, saved inputs.

No fitting, threshold change, probability blending or word-specific filtering.
The primary model supplies every score/actual choice; the other head only vetoes.
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
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import restored_sites, verify_phones, detail
from mdd.sibilant_cv_policy import annotate
from mdd.sibilant_model_agreement import confirmed_availability


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
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "thresholds": p["thresholds"], "fitting": False,
              "deployment": False, "cohorts": {}}
    for name, dataset in p["datasets"].items():
        prefix = dataset["prefix"]
        full, other = np.load(dataset["primary_scores"]), np.load(dataset["confirmation_scores"])
        confirmed = confirmed_availability(full["probabilities"], full["available"], other["probabilities"], other["available"])
        available = confirmed if p.get("confirmation_required", True) else full["available"]
        np.savez(out / (name + ".scores.npz"), probabilities=full["probabilities"], available=available)
        features, labels = service.read_jsonl(prefix + ".features.jsonl"), service.read_jsonl(prefix + ".labels.jsonl")
        records = service.read_jsonl(prefix + ".responses.jsonl")
        if len(features) != len(available):
            raise ValueError("input and score order/count differs")
        by_uid, by_label = collections.defaultdict(dict), collections.defaultdict(list)
        for i, f in enumerate(features):
            by_uid[f["uid"]][f["q_runtime"]] = {"probabilities": full["probabilities"][i].tolist(), "available": bool(available[i])}
        for label in labels:
            by_label[label["uid"]].append(label)
        rows, counts = [], collections.Counter()
        counts.update({"all_feature_inputs": len(available), "primary_available": int(full["available"].sum()),
                       "confirmed_available": int(available.sum())})
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for record in records:
                row, base = record["row"], record[p.get("base_key", "prior")]
                candidate = dict(base)
                if base["raw"] is not None and by_uid[row["uid"]]:
                    raw, phones = base["raw"], service.phonemes(base["body"])
                    sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                    if render(app, row, raw, sites) != base["body"]:
                        raise AssertionError("prior response changed")
                    sites, states = annotate(raw["canon"], sites, phones, by_uid[row["uid"]], p["thresholds"], False)
                    counts.update(states.values())
                    if "confirmed" in states.values():
                        candidate["body"] = render(app, row, raw, sites)
                        verify_phones(phones, service.phonemes(candidate["body"]), sites)
                views = {"original": observe(record["original"], row), "reference": observe(record["existing"], row),
                         "prior": observe(record["prior"], row), "candidate": observe(candidate, row)}
                for label in by_label[row["uid"]]:
                    value = dict(label)
                    value.update({key: view[label["q"]] for key, view in views.items()})
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
        with (out / (name + ".sites.jsonl")).open("x") as stream:
            for row in rows:
                service.write_line(stream, row)
        result = detail(rows, counts)
        report["cohorts"][name] = result
        print(name, json.dumps(result["versus_prior"], ensure_ascii=False), flush=True)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

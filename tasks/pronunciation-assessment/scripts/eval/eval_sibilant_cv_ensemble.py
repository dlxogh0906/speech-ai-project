"""Render a fixed-order ensemble from verified identical cached runtime inputs."""
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
import shutil
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render, comparison
from eval_sibilant_cv_agreement import restored_sites, verify_phones, detail
from train_sibilant_local_encoder import write
from mdd.sibilant_cv_ensemble import annotate_heads


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    unique_gains = set()
    for name, c in p["cohorts"].items():
        records = service.read_jsonl(c["records"])
        labels = service.read_jsonl(c["labels"])
        by_uid, observations = collections.defaultdict(list), collections.defaultdict(dict)
        for label in labels:
            by_uid[label["uid"]].append(label)
        reference_inputs, reference_features = None, None
        other_views = {}
        for h in c["heads"]:
            features = service.read_jsonl(h["features"])
            data, scores = np.load(h["inputs"]), np.load(h["scores"])
            identity = [{k: f[k] for k in ("uid", "q_runtime", "clip_sha256", "vad_pcm_sha256", "left", "right")} for f in features]
            if reference_inputs is None:
                reference_inputs, reference_features = data, identity
            else:
                if identity != reference_features:
                    raise AssertionError("head inputs/ordering differ")
                for key in ("x", "relative", "lengths"):
                    np.testing.assert_array_equal(data[key], reference_inputs[key])
            for i, f in enumerate(features):
                observations[f["uid"]].setdefault(h["name"], {})[f["q_runtime"]] = {
                    "probabilities": scores["probabilities"][i].tolist(), "available": bool(scores["available"][i])}
            other_views[h["name"]] = {(r["uid"], r["q"]): r["candidate"] for r in service.read_jsonl(h["site_results"])}
        counts, rows = collections.Counter(), []
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for record in records:
                row, base = record["row"], record[c["base"]]
                candidate = dict(base)
                if base["raw"] is not None and observations[row["uid"]]:
                    raw, phones = base["raw"], service.phonemes(base["body"])
                    sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                    heads = [{"name": h["name"], "thresholds": p["thresholds"][h["name"]],
                              "observations": observations[row["uid"]].get(h["name"], {})} for h in c["heads"]]
                    sites, traces = annotate_heads(raw["canon"], sites, phones, heads)
                    for head, states in traces.items():
                        counts.update(head + "/" + state for state in states.values())
                    if any("confirmed" in states.values() for states in traces.values()):
                        candidate["body"] = render(app, row, raw, sites)
                        verify_phones(phones, service.phonemes(candidate["body"]), sites)
                        counts["changed_utterances_rendered"] += 1
                view = observe(candidate, row)
                for label in by_uid[row["uid"]]:
                    value = dict(label)
                    value["candidate"] = view[label["q"]]
                    value.update({h: v[(label["uid"], label["q"])] for h, v in other_views.items()})
                    new, prior = value["candidate"]["phone"], value["prior"]["phone"]
                    if (label["label_kind"] not in ("correct", "ambiguous") and new and new.get("errorType")
                            and new.get("alpha_jamo") == label["expected"] and new.get("actual_jamo") == label["actual"]
                            and not (prior and prior.get("errorType") and prior.get("actual_jamo") == label["actual"]
                                     and prior.get("alpha_jamo") == label["expected"])):
                        unique_gains.add((label["uid"], label["q"]))
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
        with (out / (name + ".sites.jsonl")).open("x") as stream:
            for row in rows:
                service.write_line(stream, row)
        result = detail(rows, counts)
        for head in other_views:
            result["versus_" + head] = comparison(rows, head, "candidate")
        result["identical_runtime_input_features_verified"] = True
        report["cohorts"][name] = result
        report["unique_additional_exact_sites_across_all_cohorts"] = [list(v) for v in sorted(unique_gains)]
        write(out / "report.json", report)
        print(name, json.dumps(result["versus_prior"], ensure_ascii=False), flush=True)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


if __name__ == "__main__":
    main()

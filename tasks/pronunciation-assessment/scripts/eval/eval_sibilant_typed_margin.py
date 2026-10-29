"""Fit cal-only third-consonant confidence and evaluate original API outputs."""
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
import time
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_complex import metrics
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_typed_margin import observations, annotate, calibrate


def app_from(p):
    meta = json.loads(Path(p["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(meta["source_sha256"], **meta["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("original source changed: " + path)
    paths = {Path(k).name: k for k in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    return service.app_module(config, meta["base_ckpt"])


def render(app, row, raw, sites):
    import mdd.speechsuper as formatter
    previous = formatter.SZ_RULE
    try:
        formatter.SZ_RULE = True
        return service.jsonable(service.response(app, row, raw, sites))
    finally:
        formatter.SZ_RULE = previous


def extract(p, root, split):
    if split == "test" and not (root / "policy.json").exists():
        raise ValueError("freeze cal policy before test extraction")
    path = root / (split + ".observations.jsonl")
    if path.exists():
        raise FileExistsError(path)
    app = app_from(p)
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    forward, captured = app.S["fwd"], {}

    def capture(dec, wavs):
        lp, lengths = forward(dec, wavs)
        captured["lp"] = lp[0, :lengths[0]].numpy().copy()
        return lp, lengths

    app.S["fwd"] = capture
    dec = app.S["dec"]
    records = service.read_jsonl(Path(p["source"]) / (split + ".responses.jsonl"))
    counts, started = collections.Counter(), time.monotonic()
    with path.open("w") as stream:
        for number, record in enumerate(records, 1):
            original = record["variants"]["reference_margin"]
            values, sites = {}, None
            if original["raw"] is not None:
                row = record["row"]
                _, _, raw = app.analyze(service.load_audio(row)[:320000], row["prompt"], "")
                sites, _ = annotate_margin(captured["lp"], raw["canon"], raw["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                if render(app, row, raw, sites) != original["body"] or raw["canon"] != original["raw"]["canon"]:
                    raise AssertionError("existing integrated API output not reproduced")
                values = observations(captured["lp"], raw["canon"], raw["sites"], dec.d.index, dec.blank)
                counts["accepted_reproduced"] += 1
                counts.update(v["state"] for v in values.values())
            else:
                counts["input_rejected"] += 1
            service.write_line(stream, {"uid": record["row"]["uid"], "observations": values, "base_sites": sites})
            if number % 100 == 0:
                print(split, number, dict(counts), round(time.monotonic() - started, 1), flush=True)
    (root / (split + ".extract.finished.json")).write_text(json.dumps({"counts": dict(counts), "sha256": service.sha(path), "seconds": time.monotonic() - started}) + "\n")


def calibration_rows(p, root):
    records = service.read_jsonl(Path(p["source"]) / "cal.responses.jsonl")
    values = service.read_jsonl(root / "cal.observations.jsonl")
    if len(records) != len(values):
        raise AssertionError("calibration identity mismatch")
    rows = []
    for record, scores in zip(records, values):
        row, base = record["row"], record["variants"]["reference_margin"]
        if row["uid"] != scores["uid"]:
            raise AssertionError("calibration order mismatch")
        viewed = observe(base, row)
        mapping = service.all_optimal_mapping(row["canonical"], base["raw"]["canon"]) if base["raw"] else None
        for site in row["sites"]:
            q = site["q"]
            options = mapping[q] if mapping is not None else set()
            runtime_q = next(iter(options)) if len(options) == 1 and None not in options else None
            value = scores["observations"].get(str(runtime_q)) if runtime_q is not None else None
            rows.append(dict(site, expected=site["canon"], view=viewed[q], observation=value))
    return rows


def fit(p, root):
    if not (root / "cal.extract.finished.json").exists() or (root / "policy.json").exists():
        raise ValueError("missing cal extraction or policy already exists")
    thresholds, audit = calibrate(calibration_rows(p, root))
    policy = {"thresholds": thresholds, "calibration": audit,
              "cal_observations_sha256": service.sha(root / "cal.observations.jsonl"),
              "test_read_during_fit": False, "enabled_in_service": False}
    (root / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n")
    print("frozen policy", json.dumps(policy, ensure_ascii=False), flush=True)


def comparison(rows, reference, candidate):
    result = metrics([dict(r, baseline=r[reference], margin=r[candidate], reference_margin=r[candidate]) for r in rows])
    return {"all_sites": result["all_sites"], "label_kinds": result["label_kinds"],
            "baseline": result["baseline"], "candidate": result["margin"]}


def evaluate(p, root):
    policy = json.loads((root / "policy.json").read_text())
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "policy": policy, "splits": {}}
    for split in ("cal", "test"):
        if not (root / (split + ".extract.finished.json")).exists():
            raise ValueError("extraction incomplete")
        records = service.read_jsonl(Path(p["source"]) / (split + ".responses.jsonl"))
        scores = service.read_jsonl(root / (split + ".observations.jsonl"))
        if len(records) != len(scores):
            raise AssertionError("cohort lengths differ")
        labels = []
        with (root / (split + ".responses.jsonl")).open("x") as stream:
            for record, score in zip(records, scores):
                row, base = record["row"], record["variants"]["reference_margin"]
                if row["uid"] != score["uid"]:
                    raise AssertionError("cohort order mismatch")
                candidate = dict(base)
                if base["raw"] is not None:
                    raw = base["raw"]
                    sites = annotate(raw["canon"], score["base_sites"], score["observations"], policy["thresholds"])
                    candidate["body"] = render(app, row, raw, sites)
                    old, new = service.phonemes(base["body"]), service.phonemes(candidate["body"])
                    if len(old) != len(new):
                        raise AssertionError("phone count changed")
                    for a, b in zip(old, new):
                        if (a.get("actual_jamo") != b.get("actual_jamo") or a.get("pronunciation") != b.get("pronunciation")
                                or a.get("errorType") and not b.get("errorType")):
                            raise AssertionError("recognized actual, score or existing alert changed")
                views = {"original": observe(record["variants"]["baseline"], row), "existing": observe(base, row), "typed": observe(candidate, row)}
                for site in row["sites"]:
                    value = dict(site, expected=site["canon"], uid=row["uid"], item=row["item"], speaker=row["speaker"], language=row.get("language"))
                    value.update({name: view[site["q"]] for name, view in views.items()})
                    labels.append(value)
                service.write_line(stream, {"uid": row["uid"], "rejected": candidate["rejected"], "body": candidate["body"]})
        with (root / (split + ".sites.jsonl")).open("x") as stream:
            for row in labels:
                service.write_line(stream, row)
        detail = {"versus_original": comparison(labels, "original", "typed"), "incremental": comparison(labels, "existing", "typed")}
        for field in ("label_kind", "expected", "following", "position", "item", "language"):
            detail[field] = {str(v): comparison([r for r in labels if r[field] == v], "existing", "typed")
                             for v in sorted({r[field] for r in labels}, key=str)}
        report["splits"][split] = detail
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(split, json.dumps(detail["incremental"], ensure_ascii=False), flush=True)
    (root / "finished.json").write_text(json.dumps({"sha256": service.sha(root / "report.json")}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("extract", "fit", "evaluate"), required=True)
    parser.add_argument("--split", choices=("cal", "test"))
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "protocol.json").exists():
        shutil.copy2(args.protocol, root / "protocol.json")
        shutil.copy2(__file__, root / Path(__file__).name)
    if args.stage == "extract":
        if args.split is None:
            raise ValueError("split required")
        extract(p, root, args.split)
    elif args.stage == "fit":
        fit(p, root)
    else:
        evaluate(p, root)


if __name__ == "__main__":
    main()

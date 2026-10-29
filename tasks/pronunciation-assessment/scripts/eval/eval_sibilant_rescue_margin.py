"""Fixed factorial comparison of cached speech admission and symmetric margin.

No fitting or TAG timing is used. Rescued inputs require new frozen-model logits;
original accepts reuse their recorded margins and verify the previous API output.
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
import shutil
import time

import numpy as np

import eval_sibilant_directional as service
from eval_speech_rescue import observe, metrics
from train_sibilant_transition import render
from mdd.sibilant_margin import margin_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    for path, digest in protocol["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    source = Path(protocol["features"])
    gate = Path(protocol["gate"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    meta = json.loads((source / "metadata.json").read_text())["service_metadata"]
    for path, digest in dict(meta["source_sha256"], **meta["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("operational source/scorer changed: " + path)
    paths = {Path(p).name: p for p in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"],
                               norms=paths["norms.json"], calib=paths["calibration.json"],
                               flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, meta["base_ckpt"])
    forward, captured = app.S["fwd"], {}

    def capture(dec, wavs):
        lp, lengths = forward(dec, wavs)
        if len(wavs) != 1:
            raise ValueError("singleton required")
        captured["lp"] = lp[0, :lengths[0]].numpy().copy()
        return lp, lengths

    app.S["fwd"] = capture
    dec = app.S["dec"]
    report = {"scope": protocol["scope"], "thresholds": protocol["thresholds"], "splits": {}}
    for split in protocol["splits"]:
        original = service.read_jsonl(source / (split + ".responses.jsonl"))
        admitted = service.read_jsonl(gate / (split + ".responses.jsonl"))
        old_features = service.read_jsonl(source / (split + ".features.jsonl"))
        prior_margin = service.read_jsonl(Path(protocol["margin_evaluation"]) / ("fixed_previous_margin." + split + ".responses.jsonl"))
        if not len(original) == len(admitted) == len(prior_margin):
            raise AssertionError("cohort lengths differ")
        observations, rows, times, counts = [], [], [], collections.Counter()
        streams = {name: (out / (split + "." + name + ".responses.jsonl")).open("w")
                   for name in ("baseline", "gate", "margin", "combined")}
        started = time.monotonic()
        try:
            for number, (base, accepted, previous) in enumerate(zip(original, admitted, prior_margin), 1):
                if not base["uid"] == accepted["uid"] == previous["uid"] or base["row"] != accepted["row"]:
                    raise AssertionError("cohort identity changed")
                row = base["row"]
                probabilities = {}
                if base["raw"] is not None:
                    if base["body"] != accepted["body"]:
                        raise AssertionError("gate changed an existing response")
                    probabilities = {int(q): float(1 / (1 + np.exp(-np.clip(old_features[ix]["token_margin"], -700, 700))))
                                     for q, ix in base["feature_indices"].items()}
                    counts["cached_original_accepts"] += 1
                elif accepted["raw"] is not None:
                    captured.clear()
                    tick = time.monotonic()
                    _, _, raw = app.analyze(service.load_audio(row)[:320000], row["prompt"], "")
                    if service.jsonable(service.response(app, row, raw)) != accepted["body"]:
                        raise AssertionError("rescued API response not reproduced")
                    if raw["canon"] != accepted["raw"]["canon"]:
                        raise AssertionError("canonical changed")
                    scores = margin_observations(captured["lp"], raw["canon"], dec.d.index, dec.blank)
                    probabilities = {q: value["score_ss"] for q, value in scores.items() if value["state"] == "available"}
                    times.append(1000 * (time.monotonic() - tick))
                    observations.append({"uid": base["uid"], "observations": scores})
                    counts["rescued_responses_reproduced"] += 1
                else:
                    counts["still_rejected"] += 1
                adapted = dict(accepted, feature_indices={q: ix for ix, q in enumerate(probabilities)})
                output, _ = render(app, [adapted], list(probabilities.values()), protocol["thresholds"])
                combined = dict(accepted, body=output[0]["body"])
                margin = combined if base["raw"] is not None else base
                if margin["body"] != previous["body"]:
                    raise AssertionError("original frozen margin output not reproduced")
                variants = {"baseline": base, "gate": accepted, "margin": margin, "combined": combined}
                viewed = {name: observe(record, row) for name, record in variants.items()}
                for name, record in variants.items():
                    service.write_line(streams[name], {"uid": base["uid"], "rejected": record["rejected"], "body": record["body"]})
                for site in row["sites"]:
                    q = site["q"]
                    value = {"uid": base["uid"], "speaker": row["speaker"], "item": row["item"],
                             "language": row.get("language"), "q": q, "expected": site["canon"],
                             "actual": site["actual"], "following": row["canonical"][q + 1],
                             "position": "initial" if q == 0 else "post_vowel" if row["canonical"][q - 1] in service.VOWELS else "post_consonant",
                             "rescued": bool(accepted["rescue"]["rescued"])}
                    value.update({name: values[q] for name, values in viewed.items()})
                    rows.append(value)
                if number % 100 == 0:
                    print(split, number, dict(counts), round(time.monotonic() - started, 1), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
        def compare(selected, method, reference="baseline"):
            return metrics([dict(r, baseline=r[reference], candidate=r[method]) for r in selected])
        detail = {"counts": dict(counts), "seconds": time.monotonic() - started,
                  "rescued_analysis_and_margin_ms_p50_p95": np.quantile(times, [.5, .95]).tolist() if times else None,
                  "overall": {name: compare(rows, name) for name in ("gate", "margin", "combined")},
                  "combined_vs_gate": compare(rows, "combined", "gate"),
                  "rescued_combined_vs_gate": compare([r for r in rows if r["rescued"]], "combined", "gate")}
        for field in ("following", "position", "item", "language"):
            detail[field] = {str(v): compare([r for r in rows if r[field] == v], "combined")
                             for v in sorted({r[field] for r in rows}, key=str)}
        for suffix, values in (("sites", rows), ("rescued_observations", observations)):
            with (out / (split + "." + suffix + ".jsonl")).open("w") as stream:
                for value in values:
                    service.write_line(stream, value)
        report["splits"][split] = detail
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(split, json.dumps(detail["overall"], ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

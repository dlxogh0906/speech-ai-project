"""Full-outcome API comparison on fixed complex TAG cohorts, no fitting."""
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
from mdd.canonical_time import TimePhraseBuilder, TIME_PHRASE
from mdd.sibilant_margin import annotate_margin

METHODS = ("baseline", "margin", "reference_margin")


def flagged(value):
    return bool((value["phone"] or {}).get("errorType"))


def exact(row, method):
    phone = row[method]["phone"] or {}
    return (row["label_kind"] not in ("correct", "ambiguous") and flagged(row[method])
            and phone.get("alpha_jamo") == row["expected"] and phone.get("actual_jamo") == row["actual"])


def metrics(rows):
    result = {"all_sites": len(rows), "label_kinds": dict(collections.Counter(r["label_kind"] for r in rows))}
    for method in METHODS:
        normals = [r for r in rows if r["label_kind"] == "correct"]
        known = [r for r in normals if r[method]["phone"] is not None]
        errors = [r for r in rows if r["label_kind"] not in ("correct", "ambiguous")]
        result[method] = {"normal_all": len(normals), "normal_known_accepted": len(known),
                          "normal_alerts": sum(flagged(r[method]) for r in normals),
                          "normal_alert_rate_known": sum(flagged(r[method]) for r in known) / len(known) if known else None,
                          "known_error_all": len(errors), "exact_error_alerts": sum(exact(r, method) for r in errors),
                          "any_alert_on_known_error": sum(flagged(r[method]) for r in errors),
                          "wrong_actual_alert_on_known_error": sum(flagged(r[method]) and not exact(r, method) for r in errors),
                          "exact_recall_all": sum(exact(r, method) for r in errors) / len(errors) if errors else None,
                          "states_all": dict(collections.Counter(r[method]["state"] for r in rows)),
                          "human_ambiguous_sites": sum(r["label_kind"] == "ambiguous" for r in rows)}
        if method != "baseline":
            result[method].update(additional_normal_alerts=sum(flagged(r[method]) and not flagged(r["baseline"]) for r in normals),
                                  removed_normal_alerts=sum(flagged(r["baseline"]) and not flagged(r[method]) for r in normals),
                                  additional_exact_errors=sum(exact(r, method) and not exact(r, "baseline") for r in errors),
                                  lost_exact_errors=sum(exact(r, "baseline") and not exact(r, method) for r in errors))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen input changed: " + path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    meta = json.loads(Path(p["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(meta["source_sha256"], **meta["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source changed: " + path)
    paths = {Path(k).name: k for k in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, meta["base_ckpt"])
    base_builder, forward, captured = app.S["cb"], app.S["fwd"], {}
    candidate_builder = TimePhraseBuilder(base_builder)
    cached_baselines = {}
    for path in p.get("baseline_cache", []):
        for record in service.read_jsonl(path):
            cached_baselines[record["uid"]] = record

    def capture(dec, wavs):
        lp, lengths = forward(dec, wavs)
        captured.update(result=(lp, lengths), wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    def replay(dec, wavs):
        if len(wavs) != 1 or not np.array_equal(wavs[0], captured["wav"]):
            raise AssertionError("candidate PCM changed")
        return captured["result"]

    def render_margin(row, raw):
        import mdd.speechsuper as formatter
        dec = app.S["dec"]
        sites, observations = annotate_margin(captured["lp"], raw["canon"], raw["sites"], dec.d.index, dec.blank, p["thresholds"])
        previous = formatter.SZ_RULE
        try:
            formatter.SZ_RULE = True
            body = service.jsonable(service.response(app, row, raw, sites))
        finally:
            formatter.SZ_RULE = previous
        old = service.phonemes(service.response(app, row, raw))
        new = service.phonemes(body)
        if len(old) != len(new):
            raise AssertionError("formatter position count changed")
        for q, (a, b) in enumerate(zip(old, new)):
            if "sz" not in sites[q] and a != b:
                raise AssertionError("unannotated phone changed")
            if a.get("pronunciation") != b.get("pronunciation") or a.get("errorType") and not b.get("errorType"):
                raise AssertionError("existing score or alert lost")
        return body, observations

    report = {"scope": p["scope"], "splits": {}, "deployment": False}
    for split, path in p["data"].items():
        records = service.read_jsonl(path)
        labels, counts, elapsed, started = [], collections.Counter(), [], time.monotonic()
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for number, row in enumerate(records, 1):
                x = service.load_audio(row)[:320000]
                code = "too_short" if len(x) < 4800 else app.check_audio(x, row["prompt"])[0]
                variants = {name: {"raw": None, "body": None, "rejected": code} for name in METHODS}
                observations = None
                if code is None:
                    tick = time.monotonic()
                    app.S["cb"], app.S["fwd"] = base_builder, capture
                    _, _, raw = app.analyze(x, row["prompt"], "")
                    body = service.jsonable(service.response(app, row, raw))
                    margin_body, observations = render_margin(row, raw)
                    variants["baseline"] = {"raw": raw, "body": body, "rejected": None}
                    variants["margin"] = {"raw": raw, "body": margin_body, "rejected": None}
                    raw_ref = raw
                    if TIME_PHRASE.search(row["prompt"]):
                        app.S["cb"], app.S["fwd"] = candidate_builder, replay
                        _, _, raw_ref = app.analyze(x, row["prompt"], "")
                    reference_body, _ = render_margin(row, raw_ref)
                    variants["reference_margin"] = {"raw": raw_ref, "body": reference_body, "rejected": None}
                    elapsed.append(1000 * (time.monotonic() - tick))
                    counts["accepted"] += 1
                else:
                    counts["input_rejected/" + code] += 1
                if row["uid"] in cached_baselines:
                    prior = cached_baselines[row["uid"]]
                    if prior["rejected"] != code or prior["body"] != variants["baseline"]["body"]:
                        raise AssertionError("previous baseline response not reproduced")
                    counts["prior_baseline_response_reproductions"] += 1
                viewed = {name: observe(value, row) for name, value in variants.items()}
                for site in row["sites"]:
                    value = dict(site, uid=row["uid"], speaker=row["speaker"], item=row["item"], language=row.get("language"),
                                 expected=site["canon"], utterance_all_correct=row["utterance_all_correct"])
                    value.update({name: view[site["q"]] for name, view in viewed.items()})
                    labels.append(value)
                service.write_line(stream, {"row": row, "variants": variants, "margin_observations": observations})
                if number % 100 == 0:
                    print(split, number, dict(counts), round(time.monotonic() - started, 1), flush=True)
        with (out / (split + ".sites.jsonl")).open("w") as stream:
            for value in labels:
                service.write_line(stream, value)
        detail = {"overall": metrics(labels), "counts": dict(counts), "seconds": time.monotonic() - started,
                  "three_arm_pipeline_ms_p50_p95": np.quantile(elapsed, [.5, .95]).tolist() if elapsed else None}
        for field in ("expected", "label_kind", "following", "position", "item", "language", "utterance_all_correct"):
            detail[field] = {str(v): metrics([r for r in labels if r[field] == v]) for v in sorted({r[field] for r in labels}, key=str)}
        report["splits"][split] = detail
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(split, json.dumps(detail["overall"], ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

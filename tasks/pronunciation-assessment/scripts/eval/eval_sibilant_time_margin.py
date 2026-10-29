"""Fixed integration of source-attested time reference and previous margin.

Exploratory integration after component comparisons, not unused confirmation.
The operating model, admission, scores, and all thresholds remain frozen.
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
import json
import shutil
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe, metrics
from train_sibilant_transition import render
from mdd.canonical_time import TimePhraseBuilder, TIME_PHRASE
from mdd.sibilant_margin import margin_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    source, out = Path(p["features"]), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    meta = json.loads((source / "metadata.json").read_text())["service_metadata"]
    for path, digest in dict(meta["source_sha256"], **meta["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source changed: " + path)
    paths = {Path(k).name: k for k in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"],
                               norms=paths["norms.json"], calib=paths["calibration.json"],
                               flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, meta["base_ckpt"])
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    forward, captured = app.S["fwd"], {}

    def capture(dec, wavs):
        lp, lengths = forward(dec, wavs)
        captured["lp"] = lp[0, :lengths[0]].numpy().copy()
        return lp, lengths

    app.S["fwd"] = capture
    dec = app.S["dec"]
    report = {"scope": p["scope"], "splits": {}, "deployment": False}
    for split in p["splits"]:
        originals = service.read_jsonl(source / (split + ".responses.jsonl"))
        references = service.read_jsonl(Path(p["reference_evaluation"]) / (split + ".responses.jsonl"))
        margins = service.read_jsonl(Path(p["margin_evaluation"]) / ("fixed_previous_margin." + split + ".responses.jsonl"))
        features = service.read_jsonl(source / (split + ".features.jsonl"))
        if not len(originals) == len(references) == len(margins):
            raise AssertionError("cohort lengths differ")
        rows, reproduced = [], 0
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for base, ref, margin in zip(originals, references, margins):
                if not base["uid"] == ref["uid"] == margin["uid"]:
                    raise AssertionError("cohort identities differ")
                row = base["row"]
                record = dict(base, body=ref["body"])
                scores = {int(q): float(1 / (1 + np.exp(-np.clip(features[ix]["token_margin"], -700, 700))))
                          for q, ix in base["feature_indices"].items()}
                target = bool(TIME_PHRASE.search(row["prompt"]))
                if base["raw"] is not None and target:
                    _, _, raw = app.analyze(service.load_audio(row)[:320000], row["prompt"], "")
                    if service.jsonable(service.response(app, row, raw)) != ref["body"]:
                        raise AssertionError("reference-only API not reproduced")
                    record["raw"] = raw
                    values = margin_observations(captured["lp"], raw["canon"], dec.d.index, dec.blank)
                    scores = {q: v["score_ss"] for q, v in values.items() if v["state"] == "available"}
                    reproduced += 1
                record["feature_indices"] = {q: ix for ix, q in enumerate(scores)}
                output, _ = render(app, [record], list(scores.values()), p["thresholds"])
                combined = dict(record, body=output[0]["body"])
                if not target and combined["body"] != margin["body"]:
                    raise AssertionError("non-target fixed margin response changed")
                variants = {"baseline": base, "reference": record, "margin": dict(base, body=margin["body"]), "combined": combined}
                views = {name: observe(value, row) for name, value in variants.items()}
                for site in row["sites"]:
                    q = site["q"]
                    value = {"uid": row["uid"], "speaker": row["speaker"], "item": row["item"], "language": row.get("language"),
                             "q": q, "expected": site["canon"], "actual": site["actual"], "following": row["canonical"][q + 1],
                             "position": "initial" if q == 0 else "post_vowel" if row["canonical"][q - 1] in service.VOWELS else "post_consonant"}
                    value.update({name: view[q] for name, view in views.items()})
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "rejected": base["rejected"], "body": combined["body"]})
        def compare(selected, method):
            return metrics([dict(r, candidate=r[method]) for r in selected])
        detail = {"reference_responses_reproduced": reproduced,
                  "overall": {name: compare(rows, name) for name in ("reference", "margin", "combined")}}
        for field in ("following", "position", "item", "language"):
            detail[field] = {str(v): compare([r for r in rows if r[field] == v], "combined")
                             for v in sorted({r[field] for r in rows}, key=str)}
        with (out / (split + ".sites.jsonl")).open("w") as stream:
            for value in rows:
                service.write_line(stream, value)
        report["splits"][split] = detail
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(split, json.dumps(detail["overall"]["combined"], ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

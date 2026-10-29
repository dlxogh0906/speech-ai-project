"""Offline outer-silence gate comparison using frozen baseline API responses."""
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

import eval_sibilant_directional as service
from train_sibilant_transition import metrics
from mdd.speech_span_quality import assess_span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.baseline_dir), Path(args.out)
    meta = json.loads((source / "metadata.json").read_text())["service_metadata"]
    for name, digest in meta["source_sha256"].items():
        if service.sha(name) != digest:
            raise ValueError("baseline source changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"hypothesis": "An outer-silence-dependent ratio rejects sufficient real utterances before assessment",
                "fixed_change": "ratio computed on first-to-last active30ms frame with150ms edge padding; original energy mask, minimum active seconds and30percent threshold preserved",
                "unchanged": "frozen encoder, VAD preprocessing, canonical selection, thresholds, formatter",
                "no_label_in_gate": True, "no_model_training": True,
                "no_deployment": True, "comparison": "same cal/test corpus as firstpilot; reused exploratory evidence",
                "risk_to_measure": "short noise bursts may also be accepted; energy span is not a speech classifier",
                "baseline_metadata_sha256": service.sha(source / "metadata.json")}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    for file in (Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/speech_span_quality.py"):
        shutil.copy2(file, out / file.name)
    paths = {Path(f).name: f for f in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, meta["base_ckpt"])
    report = {"scope": "existing TAG, automatic runtime boundaries, exact service source on GPU; noise safety and independent validation incomplete",
              "splits": {}}
    for split in ("cal", "test"):
        baseline = service.read_jsonl(source / (split + ".responses.jsonl"))
        counts, labelled, checked = collections.Counter(), [], 0
        started = time.monotonic()
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for number, record in enumerate(baseline, 1):
                row = record["row"]
                x = service.load_audio(row)[:320000]
                span = assess_span(x, 16000, app.min_voiced_sec(row["prompt"]), app.MIN_VOICED_RATIO)
                code = "too_short" if len(x) < 4800 else (None if span["accepted"] else "low_speech")
                candidate = dict(record)
                if record["rejected"] is None:
                    if code is not None:
                        raise AssertionError("candidate rejected an existing accepted recording")
                    counts["previously_accepted_preserved"] += 1
                    if checked < 5:
                        _, _, raw = app.analyze(x, row["prompt"], "")
                        body = service.response(app, row, raw)
                        if service.jsonable(body) != record["body"]:
                            raise AssertionError("baseline body failed fresh inference reproduction")
                        checked += 1
                elif code is None:
                    _, _, raw = app.analyze(x, row["prompt"], "")
                    candidate.update(rejected=None, raw=raw, body=service.response(app, row, raw))
                    counts["newly_accepted"] += 1
                else:
                    counts["still_rejected"] += 1
                candidate["span_quality"] = span
                service.write_line(stream, candidate)
                old_canon = record["raw"]["canon"] if record["raw"] else []
                new_canon = candidate["raw"]["canon"] if candidate["raw"] else []
                old_map = service.all_optimal_mapping(row["canonical"], old_canon) if old_canon else []
                new_map = service.all_optimal_mapping(row["canonical"], new_canon) if new_canon else []
                old_phones = service.phonemes(record["body"]) if record["body"] else []
                new_phones = service.phonemes(candidate["body"]) if candidate["body"] else []
                for s in row["sites"]:
                    q = s["q"]
                    def target(mapping, canon):
                        options = mapping[q] if mapping else set()
                        idx = next(iter(options)) if len(options) == 1 and None not in options else None
                        return idx if idx is not None and canon[idx] == s["canon"] else None
                    old_q, new_q = target(old_map, old_canon), target(new_map, new_canon)
                    entry = dict(s)
                    entry.update(uid=row["uid"], speaker=row["speaker"], item=row["item"], expected=s["canon"],
                                 following=row["canonical"][q + 1],
                                 state="available" if new_q is not None else
                                 "input_rejected" if candidate["rejected"] else "runtime_mapping_unavailable",
                                 baseline_phone=old_phones[old_q] if old_q is not None else None,
                                 candidate_phone=new_phones[new_q] if new_q is not None else None,
                                 newly_accepted=record["rejected"] is not None and candidate["rejected"] is None,
                                 baseline_rejected=record["rejected"], candidate_rejected=candidate["rejected"])
                    labelled.append(entry)
                if number % 100 == 0 or number == len(baseline):
                    print(split, number, dict(counts), "%.1fs" % (time.monotonic() - started), flush=True)
        with (out / (split + ".sites.jsonl")).open("w") as stream:
            for row in labelled:
                service.write_line(stream, row)
        report["splits"][split] = {"coverage": dict(counts), "fresh_accepted_reproductions": checked,
                                   "overall": metrics(labelled),
                                   "newly_accepted": metrics([r for r in labelled if r["newly_accepted"]]),
                                   "following": {v: metrics([r for r in labelled if r["following"] == v])
                                                 for v in sorted({r["following"] for r in labelled})},
                                   "rate_note": "overall FPR uses candidate matched accepted normals; inspect preserved original rates in v1 and extra FP counts, not denominator-driven rate reduction"}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

"""Frozen typed-margin comparison using cached outputs when ineligible.

Eligibility uses runtime canonical/prediction/UI state only, never TAG labels.
All records and all outcomes remain in metrics. Recompute every potentially
changed response and verify the existing integrated API body exactly.
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
import time
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render, comparison
from mdd.canonical_fix import CONSONANTS
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_transition import PAIR, VOWELS
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_typed_margin import observations, annotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    policy = json.loads(Path(p["policy"]).read_text())
    app = app_from(p)
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    forward, captured = app.S["fwd"], {}

    def capture(dec, wavs):
        lp, lengths = forward(dec, wavs)
        captured["lp"] = lp[0, :lengths[0]].numpy().copy()
        return lp, lengths

    app.S["fwd"] = capture
    dec = app.S["dec"]
    report = {"scope": p["scope"], "policy": policy, "cohorts": {}, "deployment": False}
    for cohort, source in p["sources"].items():
        records = service.read_jsonl(source)
        rows, counts, times, started = [], collections.Counter(), [], time.monotonic()
        with (out / (cohort + ".responses.jsonl")).open("w") as stream:
            for record in records:
                row, base = record["row"], record["existing"]
                candidate, values = dict(base), {}
                if base["raw"] is not None:
                    raw = base["raw"]
                    phones = service.phonemes(base["body"])
                    eligible = [q for q, token in enumerate(raw["canon"][:-1]) if token in PAIR and raw["canon"][q + 1] in VOWELS
                                and raw["sites"][q].get("pred") in CONSONANTS and raw["sites"][q].get("pred") not in PAIR
                                and not phones[q].get("errorType")]
                    if eligible:
                        tick = time.monotonic()
                        _, _, current = app.analyze(service.load_audio(row)[:320000], row["prompt"], "")
                        sites, _ = annotate_margin(captured["lp"], current["canon"], current["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                        if render(app, row, current, sites) != base["body"] or current["canon"] != raw["canon"]:
                            raise AssertionError("existing API output not reproduced")
                        values = {q: v for q, v in observations(captured["lp"], current["canon"], current["sites"], dec.d.index, dec.blank).items() if q in eligible}
                        annotated = annotate(current["canon"], sites, values, policy["thresholds"])
                        candidate.update(raw=current, body=render(app, row, current, annotated))
                        old, new = service.phonemes(base["body"]), service.phonemes(candidate["body"])
                        if len(old) != len(new):
                            raise AssertionError("phone count changed")
                        for q, (a, b) in enumerate(zip(old, new)):
                            if (a.get("actual_jamo") != b.get("actual_jamo") or a.get("pronunciation") != b.get("pronunciation")
                                    or a.get("errorType") and not b.get("errorType") or q not in eligible and a != b):
                                raise AssertionError("noncandidate phone, actual, score or alert changed")
                        times.append(1000 * (time.monotonic() - tick))
                        counts["eligible_utterances_reproduced"] += 1
                        counts.update(v["state"] for v in values.values())
                    else:
                        counts["cached_unchanged_accepted"] += 1
                else:
                    counts["input_rejected"] += 1
                viewed = {"original": observe(record["original"], row), "existing": observe(base, row), "typed": observe(candidate, row)}
                for site in row["sites"]:
                    value = dict(site, expected=site["canon"], uid=row["uid"], item=row["item"], speaker=row["speaker"], language=row.get("language"))
                    value.update({name: v[site["q"]] for name, v in viewed.items()})
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"], "observations": values})
        with (out / (cohort + ".sites.jsonl")).open("w") as stream:
            for row in rows:
                service.write_line(stream, row)
        detail = {"counts": dict(counts), "seconds": time.monotonic() - started,
                  "eligible_pipeline_ms_p50_p95": np.quantile(times, [.5, .95]).tolist() if times else None,
                  "versus_original": comparison(rows, "original", "typed"), "incremental": comparison(rows, "existing", "typed")}
        for field in ("label_kind", "expected", "following", "position", "item", "language"):
            detail[field] = {str(v): comparison([r for r in rows if r[field] == v], "existing", "typed")
                             for v in sorted({r[field] for r in rows}, key=str)}
        report["cohorts"][cohort] = detail
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(cohort, json.dumps({"counts": detail["counts"], "incremental": detail["incremental"]}, ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

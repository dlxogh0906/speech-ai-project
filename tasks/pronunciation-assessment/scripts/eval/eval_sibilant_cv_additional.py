"""Capture reference-only and previous pair-margin APIs for a frozen new sample.

The subsequent frozen C/V confirmation evaluator consumes the normalized cache.
Model input is audio and prompt only; TAG remains an outcome label. No listener
validation or new labels are implied by selecting additional recordings.
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
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin


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
    app, captured = app_from(p), {}
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    forward, dec = app.S["fwd"], app.S["dec"]

    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured["lp"] = lp[0, :lengths[0]].numpy().copy()
        return lp, lengths

    app.S["fwd"] = capture
    counts, tick = collections.Counter(), time.monotonic()
    with (out / "additional.jsonl").open("x") as stream:
        for i, row in enumerate(service.read_jsonl(p["additional_data"]), 1):
            x = service.load_audio(row)[:320000]
            rejected = "too_short" if len(x) < 4800 else app.check_audio(x, row["prompt"])[0]
            raw, reference_body, prior_body = None, None, None
            if rejected is None:
                _, _, raw = app.analyze(x, row["prompt"], "")
                reference_body = service.jsonable(service.response(app, row, raw))
                sites, _ = annotate_margin(captured["lp"], raw["canon"], raw["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                prior_body = render(app, row, raw, sites)
                counts["accepted"] += 1
            else:
                counts["input_rejected"] += 1
            service.write_line(stream, {"row": row, "original": {"raw": raw, "body": reference_body, "rejected": rejected},
                                       "existing": {"raw": raw, "body": prior_body, "rejected": rejected}})
            if i % 100 == 0:
                print(i, dict(counts), "seconds", round(time.monotonic() - tick, 1), flush=True)
    report = {"scope": "Original comparator in this normalized cache means reference-only; existing means reference plus fixed pair margin. No unmodified operating-service baseline is claimed for this additional sample.",
              "counts": dict(counts), "seconds": time.monotonic() - tick, "output_sha256": service.sha(out / "additional.jsonl")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

"""Validate the request-local adapter against frozen reference replay outputs.

Check every changed response plus runtime-selected unchanged guards. Benchmark
in-memory GPU analyze+fixed margin+formatter serially, without HTTP/WAV I/O.
This does not claim an NPU measurement or deployment.
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
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from train_sibilant_local_encoder import write
from mdd.canonical_time import TimePhraseBuilder
from mdd.observed_reference_adapter import ObservedReferenceAdapter
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
    app = app_from(p)
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    adapter = ObservedReferenceAdapter(app)
    expected_root = Path(p["expected_results"])
    records_out, benchmark = [], {}

    def run(row, wav, enabled):
        result, trace = adapter.analyze_with_trace(wav, row["prompt"], "", apply_reference=enabled)
        raw, lp = result[2], trace["frame_log_probs"]
        dec = app.S["dec"]
        sites, _ = annotate_margin(lp, raw["canon"], raw["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
        for q in trace["accepted_positions"]:
            sites[q].pop("sz", None)
        return render(app, row, raw, sites), trace

    try:
        for name, c in p["cohorts"].items():
            changed = {r["uid"] for r in service.read_jsonl(expected_root / (name + ".reference_changes.jsonl"))}
            expected = {r["uid"]: r for r in service.read_jsonl(expected_root / (name + ".responses.jsonl"))}
            guards = 0
            for record in service.read_jsonl(c["source"]):
                row, base = record["row"], record[c["base_key"]]
                if base["raw"] is None:
                    continue
                is_changed = row["uid"] in changed
                if not is_changed and guards >= 2:
                    continue
                if not is_changed:
                    guards += 1
                wav = service.load_audio(row)[:320000]
                body, trace = run(row, wav, True)
                if body != expected[row["uid"]]["body"] or trace["forward_calls"] != 1:
                    raise AssertionError("request-local adapter did not reproduce complete response")
                before = trace["raw_before"]
                lp, dec = trace["frame_log_probs"], app.S["dec"]
                old_sites, _ = annotate_margin(lp, before["canon"], before["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                if render(app, row, before, old_sites) != base["body"]:
                    raise AssertionError("baseline response did not reproduce")
                records_out.append({"cohort": name, "uid": row["uid"], "changed": is_changed,
                                    "reference_changed": trace["reference_changed"], "forward_calls": trace["forward_calls"],
                                    "cached_posterior_bytes": trace.get("cached_posterior_bytes", 0)})
                key = "changed" if is_changed else "guard"
                if key not in benchmark:
                    benchmark[key] = (row, wav)
        timings = {}
        for kind, (row, wav) in benchmark.items():
            values = {"baseline": [], "candidate": []}
            peaks = {}
            for i in range(7):
                for name, enabled in (("baseline", False), ("candidate", True)):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    _, trace = run(row, wav, enabled)
                    torch.cuda.synchronize()
                    ms = 1000 * (time.monotonic() - start)
                    if trace["forward_calls"] != 1:
                        raise AssertionError("extra acoustic forward")
                    if i >= 2:
                        values[name].append(ms)
                    peaks[name] = max(peaks.get(name, 0), torch.cuda.max_memory_allocated())
            timings[kind] = {"uid": row["uid"], "n_per_arm": 5,
                             "ms_p50_p95": {k: np.quantile(v, [.5, .95]).tolist() for k, v in values.items()},
                             "max_allocated_gpu_bytes": peaks}
        report = {"scope": __doc__, "validated_responses": records_out, "benchmark": timings,
                  "cohort_occurrences": len(records_out), "unique_uids": len({r["uid"] for r in records_out}),
                  "changed_occurrences": sum(r["changed"] for r in records_out), "all_bodies_exact": True,
                  "acoustic_forwards_per_request": 1, "no_request_global_canonical_mutation": True,
                  "npu_tested": False, "deployment": False}
        write(out / "report.json", report)
        write(out / "finished.json", {"sha256": service.sha(out / "report.json")})
        print(json.dumps({k: report[k] for k in ("cohort_occurrences", "unique_uids", "changed_occurrences", "benchmark")}, ensure_ascii=False), flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()

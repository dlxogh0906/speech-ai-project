"""Validate an isolated NPU reference adapter with the deployed app and graph.

No HTTP server, production callback changes, model compilation or installation.
The protocol supplies deployed paths, a private module overlay and test inputs.
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
import copy
import hashlib
import importlib.util
import json
import resource
import subprocess
import time
import wave
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, bundle, out = json.loads(Path(a.protocol).read_text()), Path(a.bundle), Path(a.out)
    for path, expected in p["deployed_sha256"].items():
        if sha(path) != expected:
            raise ValueError("deployed source changed: " + path)
    for path, expected in p["bundle_sha256"].items():
        if sha(bundle / path) != expected:
            raise ValueError("private bundle changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", p)
    def device_state():
        status = json.loads(subprocess.check_output(["rbln-stat", "--json"]))
        return next(d for d in status["devices"] if int(d["npu"]) == p["app_args"]["rbln_device"])

    before_device = device_state()
    if int(before_device["memory"]["total"]) - int(before_device["memory"]["used"]) < p["minimum_free_NPU_bytes"]:
        raise ValueError("insufficient currently free memory for the isolated runtime")
    spec = importlib.util.spec_from_file_location("isolated_npu_deployed_app", p["deployed_app"])
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    import torch
    torch.set_num_threads(2)
    started = time.monotonic()
    app.load_all(argparse.Namespace(**p["app_args"]))
    load_seconds = time.monotonic() - started
    loaded_device = device_state()
    import mdd
    mdd.__path__.insert(0, str(bundle / "overlay"))
    from mdd.observed_reference_adapter import ObservedReferenceAdapter
    from mdd.canonical_observed_time import FixedReference
    adapter = ObservedReferenceAdapter(app)
    actual_lprobs, npu_calls = app.S["dec"].lprobs, [0]

    def counted(wavs):
        npu_calls[0] += 1
        return actual_lprobs(wavs)

    app.S["dec"].lprobs = counted

    def body(text, raw):
        return serial(app.S["ss"].build(text, raw["canon"], raw["sites"], raw["margins"],
                                      utt_scores=raw["scores"], record_id=None,
                                      vowels=app.S["vowels"], feats=raw.get("feats")))

    def run(wav, text, enabled):
        calls, tick = npu_calls[0], time.monotonic()
        result, trace = adapter.analyze_with_trace(wav, text, "", apply_reference=enabled)
        if npu_calls[0] - calls != 1 or trace["forward_calls"] != 1:
            raise AssertionError("expected one actual NPU forward per admitted request")
        return result, trace, body(text, result[2]), 1000 * (time.monotonic() - tick)

    rows = [json.loads(line) for line in (bundle / "inputs.jsonl").open()]
    counts, timings, benchmarks = collections.Counter(), [], {}
    output_hashes = {}
    try:
        with (out / "responses.jsonl").open("x") as stream:
            for number, row in enumerate(rows, 1):
                with wave.open(str(bundle / row["audio"]), "rb") as wavfile:
                    if (wavfile.getframerate(), wavfile.getnchannels(), wavfile.getsampwidth()) != (16000, 1, 2):
                        raise ValueError("expected original mono PCM16")
                    wav = np.frombuffer(wavfile.readframes(wavfile.getnframes()), np.int16).astype(np.float32) / 32768.
                wav = wav[:320000]
                code = "too_short" if len(wav) < 4800 else app.check_audio(wav, row["prompt"])[0]
                if code:
                    counts["input_rejected"] += 1
                    record = {"uid": row["uid"], "baseline": {"raw": None, "body": None, "rejected": code},
                              "candidate": {"raw": None, "body": None, "rejected": code}}
                else:
                    counts["accepted"] += 1
                    original, _, oldbody, oldms = run(wav, row["prompt"], False)
                    result, trace, newbody, newms = run(wav, row["prompt"], True)
                    if body(row["prompt"], trace["raw_before"]) != oldbody:
                        raise AssertionError("NPU baseline was not deterministic")
                    if original[0]["pred_jamo"] != result[0]["pred_jamo"]:
                        raise AssertionError("reference changed acoustic recognition")
                    counts[trace["reason"]] += 1
                    if trace["reference_changed"]:
                        counts["reference_changed"] += 1
                        cb, fwd, calls = app.S["cb"], app.S["fwd"], npu_calls[0]
                        lp = trace["frame_log_probs"]

                        def replay(decoder, wavs):
                            if len(wavs) != 1:
                                raise ValueError("oracle requires one input")
                            return torch.from_numpy(lp)[None], [len(lp)]

                        try:
                            app.S["cb"] = FixedReference(row["prompt"], result[2]["canon"])
                            app.S["fwd"] = replay
                            oracle = app.analyze(wav, row["prompt"], "")
                        finally:
                            app.S["cb"], app.S["fwd"] = cb, fwd
                        if npu_calls[0] != calls or body(row["prompt"], oracle[2]) != newbody:
                            raise AssertionError("cached-posterior original-app oracle differs")
                        counts["oracle_entire_response_equal"] += 1
                    else:
                        if oldbody != newbody:
                            raise AssertionError("unchanged reference changed API response")
                        counts["unchanged_entire_response_equal"] += 1
                    timings.append({"uid": row["uid"], "baseline_ms": oldms, "candidate_ms": newms})
                    record = {"uid": row["uid"], "baseline": {"raw": original[2], "body": oldbody, "rejected": None},
                              "candidate": {"raw": result[2], "body": newbody, "rejected": None},
                              "trace": {k: trace[k] for k in ("reason", "accepted_positions", "reference_changed", "forward_calls")}}
                    if row["uid"] in p["benchmark_uids"]:
                        benchmarks[row["uid"]] = (wav, row["prompt"])
                stream.write(json.dumps(serial(record), ensure_ascii=False) + "\n")
                if number % 25 == 0:
                    print("NPU inputs", number, dict(counts), flush=True)
        bench = {}
        for uid in p["benchmark_uids"]:
            wav, text = benchmarks[uid]
            times = {"baseline": [], "candidate": []}
            for i in range(7):
                for name, enabled in (("baseline", False), ("candidate", True)):
                    _, _, _, ms = run(wav, text, enabled)
                    if i >= 2:
                        times[name].append(ms)
            bench[uid] = {"n_per_arm": 5, "ms_p50_p95": {k: np.quantile(v, [.5, .95]).tolist() for k, v in times.items()}}
        write(out / "request_timings.json", timings)
        output_hashes["responses.jsonl"] = sha(out / "responses.jsonl")
        report = {"scope": p["scope"], "device": app.S["dec"].device, "inputs": len(rows),
                  "counts": dict(counts), "actual_NPU_calls": npu_calls[0], "benchmark": bench,
                  "load_seconds": load_seconds, "max_process_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "device_before_runtime": before_device, "device_with_runtime": loaded_device,
                  "deployed_artifact_sha256": {k: sha(v) for k, v in p["deployed_artifacts"].items()},
                  "output_sha256": output_hashes, "deployment": False, "production_process_changed": False,
                  "limits": "Isolated NPU device/process, original deployed graph/npz scorer; not concurrent production NPU7 latency, GPU parity or new phonetic recognition evidence."}
        write(out / "report.json", report)
        write(out / "finished.json", {"sha256": sha(out / "report.json")})
        print(json.dumps(serial(report), ensure_ascii=False), flush=True)
    finally:
        adapter.close()
        app.S["dec"].lprobs = actual_lprobs


if __name__ == "__main__":
    main()

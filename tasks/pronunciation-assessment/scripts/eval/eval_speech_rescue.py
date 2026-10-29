"""Compare a frozen speech-supported input rescue through actual API responses."""
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
import hashlib
import json
import shutil
import time

import numpy as np

import eval_sibilant_directional as service
from mdd.speech_rescue import SpeechRescue

PAIR = {"ㅅ": "ㅆ", "ㅆ": "ㅅ"}


def observe(record, row):
    if record["rejected"]:
        return {s["q"]: {"state": "input_rejected", "phone": None} for s in row["sites"]}
    canonical = record["raw"]["canon"]
    mapping = service.all_optimal_mapping(row["canonical"], canonical)
    phones = service.phonemes(record["body"])
    output = {}
    for site in row["sites"]:
        options = mapping[site["q"]]
        q = next(iter(options)) if len(options) == 1 and None not in options else None
        state = ("runtime_mapping_ambiguous" if q is None else
                 "available" if canonical[q] == site["canon"] else "runtime_canonical_disagrees")
        output[site["q"]] = {"state": state, "phone": phones[q] if q is not None else None}
    return output


def alert(observation):
    return bool((observation["phone"] or {}).get("errorType"))


def correct(row, method):
    phone = row[method]["phone"] or {}
    return (bool(phone.get("errorType")) and phone.get("alpha_jamo") == row["expected"]
            and phone.get("actual_jamo") == row["actual"])


def metrics(rows):
    results = {}
    for expected, alternate in PAIR.items():
        normals = [r for r in rows if r["expected"] == r["actual"] == expected]
        errors = [r for r in rows if r["expected"] == expected and r["actual"] == alternate]
        detail = {"normal_all": len(normals), "errors_all": len(errors)}
        for method in ("baseline", "candidate"):
            known = [r for r in normals if r[method]["phone"] is not None]
            tp = sum(correct(r, method) for r in errors)
            fp = sum(alert(r[method]) for r in known)
            detail[method] = {"tp": tp, "recall_all": tp / len(errors) if errors else None,
                              "tag_normal_visible_alerts": fp, "normal_position_known_accepted": len(known),
                              "tag_normal_alert_rate_known": fp / len(known) if known else None,
                              "normal_states": dict(collections.Counter(r[method]["state"] for r in normals)),
                              "error_states": dict(collections.Counter(r[method]["state"] for r in errors))}
        detail.update(additional_tp=sum(correct(r, "candidate") and not correct(r, "baseline") for r in errors),
                      lost_tp=sum(correct(r, "baseline") and not correct(r, "candidate") for r in errors),
                      additional_normal_alerts=sum(alert(r["candidate"]) and not alert(r["baseline"]) for r in normals),
                      removed_normal_alerts=sum(alert(r["baseline"]) and not alert(r["candidate"]) for r in normals))
        results[expected + "→" + alternate] = detail
    return results


def controls(app, gate):
    """Known synthetic non-speech; not a real-world noise prevalence estimate."""
    rng = np.random.RandomState(20261024)
    results = []
    for duration in (.03, .12, .3, .6, 1.2):
        n = int(16000 * duration)
        t = np.arange(n) / 16000.
        from scipy.signal import butter, sosfilt
        white = rng.normal(0., .08, n).astype(np.float32)
        signals = {"white": white, "tone_200": .1 * np.sin(2 * np.pi * 200 * t),
                   "tone_2000": .1 * np.sin(2 * np.pi * 2000 * t),
                   "chirp": .1 * np.sin(2 * np.pi * (100 * t + 1000 * t * t)),
                   "harmonic": sum(.05 / k * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 8)),
                   "highpass_noise": sosfilt(butter(4, 2500, fs=16000, btype="highpass", output="sos"), white),
                   "silence": np.zeros(n)}
        for kind, signal in signals.items():
            for pad in (0, 32000):
                x = np.pad(signal.astype(np.float32), (pad, pad))
                code = "too_short" if len(x) < 4800 else app.check_audio(x, "싸다")[0]
                result = gate.assess(x, code, app.min_voiced_sec("싸다"), app.MIN_VOICED_RATIO)
                results.append({"kind": kind, "seconds": duration, "padding_each_samples": pad,
                                "pcm_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                                "baseline_accepted": code is None, "candidate": result})
    return {"scope": "70 known synthetic non-speech controls, no claim about real noise or foreground identity",
            "newly_accepted": sum(not r["baseline_accepted"] and r["candidate"]["accepted"] for r in results),
            "baseline_accepted": sum(r["baseline_accepted"] for r in results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    for path, digest in protocol["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen input changed: " + path)
    source, out = Path(protocol["baseline_dir"]), Path(args.out)
    metadata = json.loads(Path(protocol["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(metadata["source_sha256"], **metadata["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source/scorer changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    for path in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/speech_rescue.py"]:
        shutil.copy2(path, out / path.name)
    paths = {Path(p).name: p for p in metadata["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, metadata["base_ckpt"])
    gate = SpeechRescue(protocol["vad_model"], threshold=protocol["speech_threshold"])
    control_report = controls(app, gate)
    (out / "non_speech_controls.json").write_text(json.dumps(control_report, ensure_ascii=False, indent=2) + "\n")
    report = {"scope": "Existing TAG, reused calibration/test, frozen gate; independent speech gate plus original energy span, unchanged assessment PCM/model/formatter",
              "normal_rate_note": "Acceptance expands, so compare alert counts and coverage as well as rates; newly rejected/unknown sites are not successes.",
              "splits": {}}
    for split in protocol["splits"]:
        records = service.read_jsonl(source / (split + ".responses.jsonl"))
        rows, counts, gate_ms, analysis_ms = [], collections.Counter(), [], []
        started, reproduced = time.monotonic(), 0
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for number, record in enumerate(records, 1):
                row = record["row"]
                x = service.load_audio(row)[:320000]
                digest = hashlib.sha256(x.tobytes()).hexdigest()
                original_code = "too_short" if len(x) < 4800 else app.check_audio(x, row["prompt"])[0]
                if original_code != record["rejected"]:
                    raise AssertionError("original input status changed: " + row["uid"])
                tick = time.monotonic()
                decision = gate.assess(x, original_code, app.min_voiced_sec(row["prompt"]), app.MIN_VOICED_RATIO)
                gate_ms.append(1000 * (time.monotonic() - tick))
                if hashlib.sha256(x.tobytes()).hexdigest() != digest:
                    raise AssertionError("gate modified assessment PCM")
                candidate = dict(record)
                if original_code is None:
                    if not decision["accepted"]:
                        raise AssertionError("previously accepted input rejected")
                    counts["original_accepted_preserved"] += 1
                    if reproduced < 5:
                        _, _, raw = app.analyze(x, row["prompt"], "")
                        if service.jsonable(service.response(app, row, raw)) != record["body"]:
                            raise AssertionError("original response changed")
                        reproduced += 1
                elif decision["accepted"]:
                    tick = time.monotonic()
                    _, _, raw = app.analyze(x, row["prompt"], "")
                    analysis_ms.append(1000 * (time.monotonic() - tick))
                    candidate.update(rejected=None, raw=raw, body=service.response(app, row, raw))
                    counts["newly_accepted"] += 1
                else:
                    counts["still_rejected"] += 1
                candidate["rescue"] = decision
                service.write_line(stream, candidate)
                baseline_sites, candidate_sites = observe(record, row), observe(candidate, row)
                for site in row["sites"]:
                    q = site["q"]
                    rows.append({"uid": row["uid"], "speaker": row["speaker"], "item": row["item"],
                                 "language": row.get("language"), "q": q, "expected": site["canon"],
                                 "actual": site["actual"], "following": row["canonical"][q + 1],
                                 "position": "initial" if q == 0 else "post_vowel" if row["canonical"][q - 1] in "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ" else "post_consonant",
                                 "rescued": decision["rescued"], "baseline": baseline_sites[q], "candidate": candidate_sites[q]})
                if number % 100 == 0 or number == len(records):
                    print(split, number, dict(counts), round(time.monotonic() - started, 1), flush=True)
        with (out / (split + ".sites.jsonl")).open("w") as stream:
            for row in rows:
                service.write_line(stream, row)
        detail = {"coverage": dict(counts), "original_response_reproductions": reproduced,
                  "overall": metrics(rows), "newly_accepted": metrics([r for r in rows if r["rescued"]]),
                  "gate_ms_p50_p95": np.quantile(gate_ms, [.5, .95]).tolist(),
                  "new_analysis_ms_p50_p95": np.quantile(analysis_ms, [.5, .95]).tolist() if analysis_ms else None,
                  "seconds": time.monotonic() - started}
        for field in ("following", "position", "item", "language"):
            detail[field] = {str(v): metrics([r for r in rows if r[field] == v]) for v in sorted({r[field] for r in rows}, key=str)}
        report["splits"][split] = detail
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"report_sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

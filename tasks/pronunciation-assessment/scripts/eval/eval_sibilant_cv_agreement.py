"""Frozen C/V confirmation, alone and combined with the previous pair margin.

Primary uses saved runtime features/scores and renders every candidate with the
original API formatter. Regression replays every runtime-eligible utterance in
four earlier cohorts. Ineligible records remain unchanged and in denominators.
The agreement policy was proposed after inspecting earlier test cases: this is
exploratory reused-data evidence, never a fresh final evaluation.
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
import copy
import hashlib
import json
import shutil
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render, comparison
from prepare_sibilant_cv_support import retained_samples
from train_sibilant_cv_support import score
from train_sibilant_local_encoder import load_encoder, write
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_cv_policy import annotate
from mdd.sibilant_transition import PAIR, VOWELS, paired_anchor
from mdd.sibilant_margin import annotate_margin
from mdd.canonical_time import TimePhraseBuilder


def restored_sites(raw, body, pair_thresholds):
    """Recover only existing frozen pair annotations from their API trace."""
    sites = copy.deepcopy(raw["sites"])
    for q, phone in enumerate(service.phonemes(body)):
        if "detector" not in phone:
            continue
        expected, actual = raw["canon"][q], phone["actual_jamo"]
        if expected not in PAIR or actual != PAIR[expected] or not phone.get("errorType"):
            raise ValueError("cached trace is not the frozen pair margin")
        sites[q]["sz"] = {"alt": actual, "p": phone["detector"], "th": pair_thresholds[expected]}
    return sites


def verify_phones(before, after, sites):
    if len(before) != len(after):
        raise AssertionError("phone count changed")
    for q, (a, b) in enumerate(zip(before, after)):
        if (a.get("pronunciation") != b.get("pronunciation") or a.get("errorType") and a != b
                or "sz" not in sites[q] and a != b):
            raise AssertionError("score, previous alert, or noncandidate phone changed")


def detail(rows, counts):
    result = {"versus_original": comparison(rows, "original", "candidate"),
              "versus_reference": comparison(rows, "reference", "candidate"),
              "versus_prior": comparison(rows, "prior", "candidate"), "counts": dict(counts)}
    for field in ("label_kind", "expected", "following", "position", "item", "language"):
        result[field] = {str(v): comparison([r for r in rows if r[field] == v], "prior", "candidate")
                         for v in sorted({r[field] for r in rows}, key=str)}
    return result


def primary(p, out, app, policy):
    source = Path(p["model_root"])
    report = {"scope": p["scope"], "deployment": False, "arms": {}}
    for split in ("cal", "test"):
        features = service.read_jsonl(source / ("features/" + split + ".features.jsonl"))
        labels = service.read_jsonl(source / ("features/" + split + ".labels.jsonl"))
        records = service.read_jsonl(source / ("features/" + split + ".responses.jsonl"))
        scores = np.load(source / ("heads/learned_support." + split + ".scores.npz"))
        by_uid, by_label = collections.defaultdict(dict), collections.defaultdict(list)
        for i, f in enumerate(features):
            by_uid[f["uid"]][f["q_runtime"]] = {"probabilities": scores["probabilities"][i].tolist(), "available": bool(scores["available"][i])}
        for label in labels:
            by_label[label["uid"]].append(label)
        previous = {r["uid"]: r["body"] for r in service.read_jsonl(source / ("heads/learned_support." + split + ".responses.jsonl"))}
        for arm, base_key, agreement in (("head", "existing", False), ("head_agreement", "existing", True),
                                         ("union", "prior", False), ("union_agreement", "prior", True)):
            rows, counts = [], collections.Counter()
            with (out / (arm + "." + split + ".responses.jsonl")).open("x") as stream:
                for record in records:
                    row, base = record["row"], record[base_key]
                    candidate = dict(base)
                    if base["raw"] is not None:
                        raw, phones = base["raw"], service.phonemes(base["body"])
                        sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                        if render(app, row, raw, sites) != base["body"]:
                            raise AssertionError("cached baseline trace did not replay")
                        sites, states = annotate(raw["canon"], sites, phones, by_uid[row["uid"]], policy["thresholds"], agreement)
                        counts.update(states.values())
                        candidate["body"] = render(app, row, raw, sites)
                        verify_phones(phones, service.phonemes(candidate["body"]), sites)
                    if arm == "head" and candidate["body"] != previous[row["uid"]]:
                        raise AssertionError("previous ungated head output did not reproduce exactly")
                    views = {"original": observe(record["original"], row), "reference": observe(record["existing"], row),
                             "prior": observe(record["prior"], row), "candidate": observe(candidate, row)}
                    for label in by_label[row["uid"]]:
                        value = dict(label)
                        value.update({name: view[label["q"]] for name, view in views.items()})
                        rows.append(value)
                    service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
            with (out / (arm + "." + split + ".sites.jsonl")).open("x") as stream:
                for row in rows:
                    service.write_line(stream, row)
            report["arms"].setdefault(arm, {})[split] = detail(rows, counts)
            print(arm, split, json.dumps(report["arms"][arm][split]["versus_prior"], ensure_ascii=False), flush=True)
            write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


def confirmation(p, out, app, policy):
    from transformers import SeamlessM4TFeatureExtractor
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])
    forward, captured = app.S["fwd"], {}

    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S["fwd"] = capture
    artifact = torch.load(p["artifact"], map_location="cpu")
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    head = SupportReadout(encoder.config.hidden_size, True).to("cuda")
    head.load_state_dict(artifact["head"])
    # Training/calibration data are used below only to REPORT cohort overlap.
    exposure = {s: service.read_jsonl(Path(p["split_root"]) / (s + ".jsonl")) for s in ("train", "cal", "test")}
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    benchmark_inputs = []
    for name, path in p["sources"].items():
        records = service.read_jsonl(path)
        values, feature_rows, counts, timings = [], [], collections.Counter(), []
        torch.cuda.reset_peak_memory_stats()
        for record in records:
            row, base = record["row"], record["existing"]
            if base["raw"] is None:
                counts["input_rejected"] += 1
                continue
            raw, phones = base["raw"], service.phonemes(base["body"])
            eligible = [q for q, token in enumerate(raw["canon"][:-1])
                        if token in PAIR and raw["canon"][q + 1] in VOWELS
                        and raw["sites"][q].get("pred") == PAIR[token] and not phones[q].get("errorType")]
            if not eligible:
                counts["cached_unchanged_accepted"] += 1
                continue
            tick = time.monotonic()
            x = service.load_audio(row)[:320000]
            _, _, current = app.analyze(x, row["prompt"], "")
            lp, wav = captured["lp"], captured["wav"]
            dec = app.S["dec"]
            sites, _ = annotate_margin(lp, current["canon"], current["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
            if render(app, row, current, sites) != base["body"] or current["canon"] != raw["canon"]:
                raise AssertionError("original eligible pipeline did not reproduce")
            np.testing.assert_array_equal(x[retained_samples(x)], wav)
            dt = len(wav) / (16000. * len(lp))
            for q in eligible:
                anchor = paired_anchor(lp, [dec.d.index(t) for t in current["canon"]], q, [dec.d.index("ㅅ"), dec.d.index("ㅆ")], dec.blank)
                if anchor is None:
                    counts["alignment_unavailable"] += 1
                    continue
                left = max(0, int((anchor["frame"] - 15) * dt * 16000))
                right = min(len(wav), int((anchor["frame"] + 16) * dt * 16000))
                clip = wav[left:right]
                extracted = fe([clip], sampling_rate=16000, return_tensors="np", padding=True)
                length = int(extracted["attention_mask"][0].sum())
                if not 3 <= length <= p["max_frames"]:
                    counts["short_or_overlong_support"] += 1
                    continue
                centers = left + (np.arange(length) + .5) * len(clip) / length
                relative = np.clip((centers / 16000. - (anchor["frame"] + .5) * dt) / .31, -2, 2).astype(np.float32)
                values.append((extracted["input_features"][0, :length].astype(np.float32), relative))
                feature_rows.append({"uid": row["uid"], "q_runtime": q, "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
                                     "vad_pcm_sha256": hashlib.sha256(wav.tobytes()).hexdigest(), "left": left, "right": right})
            counts["eligible_utterances_reproduced"] += 1
            timings.append(1000 * (time.monotonic() - tick))
        data = {"x": np.zeros((len(values), p["max_frames"], 160), np.float32),
                "relative": np.zeros((len(values), p["max_frames"]), np.float32),
                "lengths": np.array([len(v[0]) for v in values], np.int64)}
        for i, (x, relative) in enumerate(values):
            data["x"][i, :len(x)] = x
            data["relative"][i, :len(x)] = relative
        np.savez(out / (name + ".features.npz"), **data)
        probabilities, available, seconds = score(encoder, head, data)
        np.savez(out / (name + ".scores.npz"), probabilities=probabilities, available=available)
        by_uid = collections.defaultdict(dict)
        with (out / (name + ".features.jsonl")).open("x") as stream:
            for i, f in enumerate(feature_rows):
                service.write_line(stream, f)
                by_uid[f["uid"]][f["q_runtime"]] = {"probabilities": probabilities[i].tolist(), "available": bool(available[i])}
                if len(benchmark_inputs) < 20:
                    benchmark_inputs.append({k: v[i:i + 1] for k, v in data.items()})
        rows = []
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for record in records:
                row, base = record["row"], record["existing"]
                candidate = dict(base)
                if by_uid[row["uid"]]:
                    raw, phones = base["raw"], service.phonemes(base["body"])
                    sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                    if render(app, row, raw, sites) != base["body"]:
                        raise AssertionError("old pair traces did not replay")
                    sites, states = annotate(raw["canon"], sites, phones, by_uid[row["uid"]], policy["thresholds"])
                    counts.update(states.values())
                    candidate["body"] = render(app, row, raw, sites)
                    verify_phones(phones, service.phonemes(candidate["body"]), sites)
                viewed = {"original": observe(record["original"], row), "reference": observe(base, row),
                          "prior": observe(base, row), "candidate": observe(candidate, row)}
                for site in row["sites"]:
                    value = dict(site, expected=site["canon"], uid=row["uid"], item=row["item"], speaker=row["speaker"], language=row.get("language"))
                    value.update({k: v[site["q"]] for k, v in viewed.items()})
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
        with (out / (name + ".sites.jsonl")).open("x") as stream:
            for row in rows:
                service.write_line(stream, row)
        result = detail(rows, counts)
        result.update({"local_features": len(values), "batched_local_score_seconds": seconds,
                       "eligible_original_replay_and_feature_ms_p50_p95": np.quantile(timings, [.5, .95]).tolist() if timings else None,
                       "peak_allocated_gpu_bytes_two_encoders": torch.cuda.max_memory_allocated(),
                       "overlap": {}})
        for split, prior in exposure.items():
            result["overlap"][split] = {key: len({r["row"][key] for r in records} & {r[key] for r in prior}) for key in ("uid", "speaker", "item")}
        report["cohorts"][name] = result
        print(name, json.dumps({"counts": dict(counts), "incremental": result["versus_prior"]}, ensure_ascii=False), flush=True)
        write(out / "report.json", report)
    latencies = []
    for i, value in enumerate(benchmark_inputs):
        _, _, seconds = score(encoder, head, value)
        if i >= 2:
            latencies.append(1000 * seconds)
    report["single_clip_cost"] = {"n": len(latencies), "ms_p50_p95": np.quantile(latencies, [.5, .95]).tolist() if latencies else None,
                                  "scope": "Local encoder plus head only, CUDA synchronized, first two excluded; excludes first-pass decoder and feature extraction. GPU, not NPU."}
    write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("primary", "confirmation"), required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    policy = json.loads(Path(p["policy"]).read_text())
    artifact = torch.load(p["artifact"], map_location="cpu")
    if artifact["thresholds"] != policy["thresholds"]:
        raise AssertionError("thresholds were altered")
    del artifact
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    app = app_from(p)
    if args.stage == "primary":
        app.S["dec"].model.to("cpu")
        primary(p, out, app, policy)
    else:
        confirmation(p, out, app, policy)


if __name__ == "__main__":
    main()

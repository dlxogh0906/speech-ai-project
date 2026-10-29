"""Capture runtime local inputs; attach coarse TAG support on train only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"), str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import hashlib
import json
import shutil
import time
import collections
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render
from extract_sibilant_localization_diagnostic import collect
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.canonical_fix import CONSONANTS
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_transition import paired_anchor, VOWELS, PAIR


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
    train_rows = service.read_jsonl(p["train"])
    tags = collect(p["train_annotations"], {r["uid"] for r in train_rows})
    app, captured = app_from(p), {}
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    forward, dec = app.S["fwd"], app.S["dec"]
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])

    def capture(decoder, wavs):
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths
    app.S["fwd"] = capture
    report = {}
    for split in ("train", "cal", "test"):
        source = ([{"row": r} for r in train_rows] if split == "train" else
                  service.read_jsonl(Path(p["evaluation_source"]) / (split + ".responses.jsonl")))
        arrays, labels, feature_rows, counts = [], [], [], collections.Counter()
        started = time.monotonic()
        with (out / (split + ".responses.jsonl")).open("w") as stream:
            for number, record in enumerate(source, 1):
                row = record["row"]
                x = service.load_audio(row)[:320000]
                code = "too_short" if len(x) < 4800 else app.check_audio(x, row["prompt"])[0]
                raw, body, plain_body, indices, mapping, by_q = None, None, None, None, None, {}
                if code is None:
                    _, _, raw = app.analyze(x, row["prompt"], "")
                    plain_body = service.jsonable(service.response(app, row, raw))
                    sites, _ = annotate_margin(captured["lp"], raw["canon"], raw["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                    body = render(app, row, raw, sites)
                    counts["accepted"] += 1
                    if split != "train":
                        old = record["variants"]["reference_margin"]
                        if old["rejected"] != code or old["body"] != body or old["raw"]["canon"] != raw["canon"]:
                            raise AssertionError("existing integrated body changed")
                    wav, lp = captured["wav"], captured["lp"]
                    indices = retained_samples(x)
                    np.testing.assert_array_equal(x[indices], wav)
                    dt = len(wav) / (16000. * len(lp))
                    mapping = service.all_optimal_mapping(row["canonical"], raw["canon"])
                    for q, token in enumerate(raw["canon"][:-1]):
                        if token not in PAIR or raw["canon"][q + 1] not in VOWELS:
                            continue
                        anchor = paired_anchor(lp, [dec.d.index(t) for t in raw["canon"]], q, [dec.d.index("ㅅ"), dec.d.index("ㅆ")], dec.blank)
                        if anchor is None:
                            counts["alignment_unavailable"] += 1
                            continue
                        left = max(0, int((anchor["frame"] - 15) * dt * 16000))
                        right = min(len(wav), int((anchor["frame"] + 16) * dt * 16000))
                        clip = wav[left:right]
                        value = fe([clip], sampling_rate=16000, return_tensors="np", padding=True)
                        length = int(value["attention_mask"][0].sum())
                        if length < 3:
                            counts["short_support"] += 1
                            continue
                        centers = left + (np.arange(length) + .5) * len(clip) / length
                        original_times = indices[np.minimum(centers.astype(int), len(indices) - 1)] / 16000.
                        relative = np.clip((centers / 16000. - (anchor["frame"] + .5) * dt) / (.31), -2, 2).astype(np.float32)
                        ix = len(arrays)
                        arrays.append({"x": value["input_features"][0, :length].astype(np.float32), "relative": relative,
                                       "times": original_times, "y": -1, "phase": np.full(length, -100, np.int64)})
                        by_q[q] = ix
                        feature_rows.append({"uid": row["uid"], "q_runtime": q, "canonical": token,
                                             "baseline_pred": raw["sites"][q].get("pred"), "left": left, "right": right,
                                             "vad_pcm_sha256": hashlib.sha256(wav.tobytes()).hexdigest(),
                                             "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest(), "input_frames": length})
                else:
                    counts["input_rejected"] += 1
                    if split != "train" and record["variants"]["reference_margin"]["rejected"] != code:
                        raise AssertionError("input admission changed")
                base = {"raw": raw, "body": plain_body, "rejected": code}
                viewed = observe(base, row)
                for site in row["sites"]:
                    q = site["q"]
                    options = mapping[q] if mapping is not None else set()
                    rq = next(iter(options)) if len(options) == 1 and None not in options else None
                    ix = by_q.get(rq)
                    labels.append(dict(site, expected=site["canon"], uid=row["uid"], speaker=row["speaker"], item=row["item"],
                                       language=row.get("language"), feature_index=ix, q_runtime=rq, existing=viewed[q]))
                    if split != "train" or ix is None or site["alignment_ambiguous"]:
                        continue
                    sample = arrays[ix]
                    actual = site["actual"]
                    sample["y"] = 0 if actual == "ㅅ" else 1 if actual == "ㅆ" else 2 if actual in ("ㅈ", "ㅉ", "ㅊ") else 3
                    tag = [t for t in tags[row["uid"]]["tag_phones"] if t[2] not in ("", ".", "-", "spn", "<unk>")]
                    if [t[2] for t in tag] != row["actual"]:
                        counts["tag_sequence_mismatch"] += 1
                        continue
                    ai = site["a"]
                    if ai is None or actual not in CONSONANTS or ai + 1 >= len(tag) or tag[ai + 1][2] not in VOWELS:
                        counts["phase_unavailable_actual_or_vowel"] += 1
                        continue
                    c, v = tag[ai], tag[ai + 1]
                    if not c[0] < c[1] <= v[0] < v[1]:
                        counts["phase_invalid_intervals"] += 1
                        continue
                    vowel_end = min(v[1], v[0] + p["vowel_prefix_seconds"])
                    mid = (v[0] + vowel_end) / 2
                    t = sample["times"]
                    phase = np.zeros(len(t), np.int64)
                    phase[(t >= c[0]) & (t < c[1])] = 1
                    phase[(t >= v[0]) & (t < mid)] = 2
                    phase[(t >= mid) & (t < vowel_end)] = 3
                    for boundary in (c[0], c[1], v[0], mid, vowel_end):
                        phase[abs(t - boundary) <= p["edge_ignore_seconds"]] = -100
                    sample["phase"] = phase
                    counts["phase_supervised_sites"] += 1
                service.write_line(stream, {"row": row, "existing": base,
                                           "prior": {"raw": raw, "body": body, "rejected": code},
                                           "original": record.get("variants", {}).get("baseline", base)})
                if number % 100 == 0:
                    print(split, number, dict(counts), "features", len(arrays), round(time.monotonic() - started, 1), flush=True)
        lengths = np.array([len(v["x"]) for v in arrays], np.int64)
        x = np.zeros((len(arrays), int(lengths.max()), 160), np.float32)
        relative = np.zeros(x.shape[:2], np.float32)
        phase = np.full(x.shape[:2], -100, np.int64)
        for i, sample in enumerate(arrays):
            x[i, :lengths[i]] = sample["x"]
            relative[i, :lengths[i]] = sample["relative"]
            phase[i, :lengths[i]] = sample["phase"]
        values = {"x": x, "relative": relative, "lengths": lengths}
        if split == "train":
            values.update(y=np.array([v["y"] for v in arrays], np.int64), phase=phase)
        np.savez(out / (split + ".npz"), **values)
        for suffix, rows in (("labels", labels), ("features", feature_rows)):
            with (out / (split + "." + suffix + ".jsonl")).open("w") as stream:
                for row in rows:
                    service.write_line(stream, row)
        report[split] = {"counts": dict(counts), "features": len(arrays), "labels": len(labels), "seconds": time.monotonic() - started,
                         "training_classes": dict(collections.Counter(v["y"] for v in arrays)) if split == "train" else None,
                         "phase_labels_in_model_input_file": split == "train"}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()

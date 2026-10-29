# -*- coding: utf-8 -*-
"""Evaluate a sibilant ranking pilot using deployable and oracle-only evidence.

Input: DATA/{cal,test}.jsonl, with uid, wav, frames, speaker, prompt,
actual/canonical token lists, and sites[{q,a,canon,actual}]. Each checkpoint
uses exactly these records. Calibration thresholds never use test labels.
This is an offline diagnostic, not the production screen formatter.
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
import hashlib
import itertools
import json
import math
import os
import time
import wave

import numpy as np

PAIR = {"ㅅ": "ㅆ", "ㅆ": "ㅅ"}
SCORES = ("token_margin", "canonical_llr")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data(data):
    for split, rows in data.items():
        ids = [r["uid"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate utterance uid in " + split)
        for row in rows:
            for key in ("actual", "canonical"):
                if not row[key] or any(not isinstance(t, str) or len(t) != 1 for t in row[key]):
                    raise ValueError("invalid token sequence: " + row["uid"])
            qs = []
            for site in row.get("sites", []):
                q, a = site["q"], site["a"]
                if not (0 <= q < len(row["canonical"]) and 0 <= a < len(row["actual"])):
                    raise ValueError("invalid target indices: " + row["uid"])
                if row["canonical"][q] != site["canon"] or row["actual"][a] != site["actual"]:
                    raise ValueError("target token mismatch: " + row["uid"])
                if site["canon"] not in PAIR or site["actual"] not in PAIR:
                    raise ValueError("target is not a sibilant: " + row["uid"])
                qs.append(q)
            if len(qs) != len(set(qs)):
                raise ValueError("duplicate canonical target: " + row["uid"])
    checks = {}
    for left, right in itertools.combinations(data, 2):
        for key in ("uid", "speaker", "prompt"):
            overlap = {r[key] for r in data[left]} & {r[key] for r in data[right]}
            checks[left + "/" + right + "/" + key] = len(overlap)
            if overlap:
                raise ValueError("pilot split overlap: %s/%s/%s (%d)" % (left, right, key, len(overlap)))
    return checks


def align_on_reference(reference, hypothesis):
    """Return the predicted token at each fixed reference position (None = deletion)."""
    import edlib
    if not hypothesis:
        return [None] * len(reference)
    ref, hyp = "".join(reference), "".join(hypothesis)
    aligned = edlib.getNiceAlignment(edlib.align(ref, hyp, task="path"), ref, hyp)
    return [None if h == "-" else h for r, h in zip(aligned["query_aligned"], aligned["target_aligned"])
            if r != "-"]


def phone_counts(reference, hypothesis):
    """All-phone PER plus non-sibilant substitution/deletion counts.

    Insertions are reported globally: assigning an insertion to a reference
    phone family is ambiguous. non_sibilant_sd_rate is therefore not PER.
    """
    import edlib
    out = collections.Counter(N=len(reference), non_sibilant_N=sum(t not in PAIR for t in reference))
    if not hypothesis:
        out.update(D=len(reference), non_sibilant_D=out["non_sibilant_N"])
        return dict(out)
    ref, hyp = "".join(reference), "".join(hypothesis)
    aligned = edlib.getNiceAlignment(edlib.align(ref, hyp, task="path"), ref, hyp)
    for r, h in zip(aligned["query_aligned"], aligned["target_aligned"]):
        if r == "-":
            out["I"] += 1
        elif h == "-":
            out["D"] += 1
            out["non_sibilant_D"] += r not in PAIR
        elif r != h:
            out["S"] += 1
            out["non_sibilant_S"] += r not in PAIR
    return dict(out)


def sequence_log_probs(log_probs, sequences, blank):
    """Compute CTC sums over all alignments; never turn impossible paths into 0."""
    import torch
    if not sequences:
        return []
    lp = torch.as_tensor(log_probs, dtype=torch.float32)
    n, t = len(sequences), lp.shape[0]
    with torch.inference_mode():
        losses = torch.nn.functional.ctc_loss(
            lp[:, None, :].expand(t, n, lp.shape[1]).contiguous(),
            torch.tensor([x for seq in sequences for x in seq], dtype=torch.long),
            torch.full((n,), t, dtype=torch.long),
            torch.tensor([len(seq) for seq in sequences], dtype=torch.long),
            blank=blank, reduction="none", zero_infinity=False,
        )
    return [-float(value) for value in losses]


def finite(value):
    return float(value) if math.isfinite(float(value)) else None


def evaluate_utterance(row, log_probs, decoder, aligner):
    canonical, actual = row["canonical"], row["actual"]
    index = decoder.d.index
    canonical_ids, actual_ids = [[index(t) for t in seq] for seq in (canonical, actual)]
    if decoder.d.unk() in canonical_ids + actual_ids or decoder.blank in canonical_ids + actual_ids:
        raise ValueError("unknown/blank target token: " + row["uid"])
    greedy_ids = [int(k) for k, _ in itertools.groupby(log_probs.argmax(-1)) if k != decoder.blank]
    hypothesis = [decoder.d[k] for k in greedy_ids
                  if len(decoder.d[k]) == 1 and 0x3131 <= ord(decoder.d[k]) <= 0x3163]
    projected = align_on_reference(canonical, hypothesis)
    segments = aligner(log_probs, canonical_ids, blank=decoder.blank)
    sequences = []
    sequence_indices = {}

    def register(seq):
        key = tuple(seq)
        if key not in sequence_indices:
            sequence_indices[key] = len(sequences)
            sequences.append(seq)
        return sequence_indices[key]

    canonical_i, actual_i = register(canonical_ids), register(actual_ids)
    site_indices = []
    for site in row.get("sites", []):
        q, a = site["q"], site["a"]
        cf = list(canonical_ids)
        cf[q] = index(PAIR[site["canon"]])
        af = list(actual_ids)
        af[a] = index(PAIR[site["actual"]])
        site_indices.append((register(cf), register(af)))
    ll = sequence_log_probs(log_probs, sequences, decoder.blank)
    sites = []
    for site, (cf_i, af_i) in zip(row.get("sites", []), site_indices):
        q, c = site["q"], site["canon"]
        token_margin = None
        if segments is not None and segments[q]:
            start, end = max(0, segments[q][0] - 1), min(len(log_probs), segments[q][-1] + 3)
            token_margin = finite(log_probs[start:end, index(PAIR[c])].max() - log_probs[start:end, index(c)].max())
        pred = projected[q]
        sites.append({**site, "token_margin": token_margin,
                      "canonical_llr": finite(ll[cf_i] - ll[canonical_i]),
                      "oracle_actual_vs_flip_llr": finite(ll[actual_i] - ll[af_i]),
                      "greedy": pred, "greedy_mismatch": pred != c,
                      "greedy_pair_flip": pred == PAIR[c],
                      "greedy_actual_preserved": pred == site["actual"]})
    result = {k: row[k] for k in ("uid", "speaker", "prompt", "role", "source_split") if k in row}
    result.update({
        "actual": actual, "canonical": canonical, "hypothesis": hypothesis,
        "ctc_actual_nll_per_token": finite(-ll[actual_i] / len(actual)),
        "phone_counts": phone_counts(actual, hypothesis), "sites": sites,
        "alignment_failed": segments is None,
    })
    return result


def rates(y, detected):
    y, detected = np.asarray(y, bool), np.asarray(detected, bool)
    tp, fp = int((y & detected).sum()), int((~y & detected).sum())
    ne, no = int(y.sum()), int((~y).sum())
    return {"tp": tp, "fp": fp, "n_errors": ne, "n_correct": no,
            "recall": tp / ne if ne else None, "fpr": fp / no if no else None}


def upper_quantile(values, q):
    # Compatibility with both older and newer NumPy installations.
    return float(np.sort(values)[int(math.ceil(q * (len(values) - 1)))])


def aggregate_phone_counts(rows):
    totals = collections.Counter()
    for row in rows:
        totals.update(row["phone_counts"])
    n, other_n = totals["N"], totals["non_sibilant_N"]
    return {**totals, "utterances": len(rows),
            "per": (totals["S"] + totals["D"] + totals["I"]) / n if n else None,
            "non_sibilant_sd_rate": (totals["non_sibilant_S"] + totals["non_sibilant_D"]) / other_n if other_n else None}


def summarize(predictions, metadata):
    from sklearn.metrics import roc_auc_score, roc_curve
    by_split = {s: [r for r in predictions if r["split"] == s] for s in ("cal", "test")}
    report = {"metadata": metadata, "directions": {}, "regression": {},
              "limitations": [
                  "calibrated score detectors are diagnostics, not production errorType/screen behavior",
                  "actual-vs-flipped-actual CTC likelihood uses human transcription and is oracle-only",
                  "matched test FPR ROC points use test labels descriptively, not for threshold selection",
                  "sampled error prevalence is not deployment prevalence; report counts and FPR separately",
                  "pilot split isolation does not remove earlier base-model exposure to these prompts",
                  "non-sibilant S/D rate excludes insertions; global insertions and full PER are separate",
              ]}
    for split, rows in by_split.items():
        vals = [r["ctc_actual_nll_per_token"] for r in rows if r["ctc_actual_nll_per_token"] is not None]
        report["regression"][split] = {**aggregate_phone_counts(rows),
            "mean_actual_ctc_nll_per_token": float(np.mean(vals)) if vals else None,
            "nonfinite_actual_ctc_utterances": len(rows) - len(vals),
            "alignment_failures": sum(r["alignment_failed"] for r in rows),
            "by_role": {role: aggregate_phone_counts([r for r in rows if r.get("role", "unknown") == role])
                        for role in sorted({r.get("role", "unknown") for r in rows})}}
    for canon in PAIR:
        sites = {split: [s for r in rows for s in r["sites"] if s["canon"] == canon]
                 for split, rows in by_split.items()}
        d = {"calibration_sites": len(sites["cal"]), "test_sites": len(sites["test"]), "scores": {}}
        for key in SCORES:
            cal = [s[key] for s in sites["cal"] if s["actual"] == canon and s[key] is not None]
            test = [s for s in sites["test"] if s[key] is not None]
            y = np.array([s["actual"] != canon for s in test], bool)
            score = np.array([s[key] for s in test], float)
            r = {"cal_correct": len(cal),
                 "cal_correct_missing": sum(s["actual"] == canon and s[key] is None for s in sites["cal"]),
                 "test_scored": len(test),
                 "test_missing": len(sites["test"]) - len(test),
                 "test_missing_errors": sum(s["actual"] != canon and s[key] is None for s in sites["test"]),
                 "test_missing_correct": sum(s["actual"] == canon and s[key] is None for s in sites["test"]),
                 "auc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
                 "calibrated": {}, "descriptive_test_roc": {}}
            for budget in (0.01, 0.05):
                if cal and len(test):
                    threshold = upper_quantile(cal, 1 - budget)
                    r["calibrated"][str(budget)] = {**rates(y, score > threshold), "threshold": threshold,
                        "cal_actual_fpr": float((np.array(cal) > threshold).mean())}
                if len(set(y)) == 2:
                    fpr, tpr, threshold = roc_curve(y, score)
                    valid = np.flatnonzero(fpr <= budget + 1e-12)
                    best = valid[np.argmax(tpr[valid])]
                    r["descriptive_test_roc"][str(budget)] = {"recall": float(tpr[best]), "fpr": float(fpr[best]),
                        "threshold_selected_using_test_labels": finite(threshold[best])}
            d["scores"][key] = r
        ts = sites["test"]
        labels = [s["actual"] != canon for s in ts]
        d["greedy_mismatch"] = rates(labels, [s["greedy_mismatch"] for s in ts])
        d["greedy_pair_flip"] = rates(labels, [s["greedy_pair_flip"] for s in ts])
        error_sites = [s for s in ts if s["actual"] != canon]
        d["actual_error_behavior"] = {"n": len(error_sites),
            "preserved": sum(s["greedy"] == s["actual"] for s in error_sites),
            "reverted_to_canonical": sum(s["greedy"] == canon for s in error_sites),
            "other_or_deleted": sum(s["greedy"] not in (canon, s["actual"]) for s in error_sites)}
        d["oracle_actual_vs_flip"] = {}
        for kind, group in (("error", error_sites), ("normal", [s for s in ts if s["actual"] == canon])):
            vals = [s["oracle_actual_vs_flip_llr"] for s in group if s["oracle_actual_vs_flip_llr"] is not None]
            d["oracle_actual_vs_flip"][kind] = {"n": len(vals), "mean_margin": float(np.mean(vals)) if vals else None,
                "actual_beats_flip": float((np.array(vals) > 0).mean()) if vals else None}
        report["directions"][canon] = d
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", "--batchsize", type=int, default=8)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    predictions_path, metadata_path = out / "predictions.jsonl", out / "metadata.json"
    if args.summarize_only:
        report = summarize(read_jsonl(predictions_path), json.load(open(metadata_path, encoding="utf-8")))
    else:
        if not args.ckpt:
            ap.error("--ckpt is required unless --summarize-only")
        if predictions_path.exists():
            raise FileExistsError(str(predictions_path))
        import torch
        from mdd.gop_feats_v2 import HFDecoder, forward_lprobs, viterbi_align
        from mdd.vad import trim
        torch.set_num_threads(2)
        data = {split: read_jsonl(Path(args.data) / (split + ".jsonl")) for split in ("cal", "test")}
        train_path = Path(args.data) / "train.jsonl"
        check_data = {**data, **({"train": read_jsonl(train_path)} if train_path.exists() else {})}
        checks = validate_data(check_data)
        metadata = {"checkpoint": os.path.abspath(args.ckpt), "data": os.path.abspath(args.data),
                    "input_sha256": {s: sha256_file(Path(args.data) / (s + ".jsonl")) for s in check_data},
                    "split_overlap_counts": checks,
                    "preprocessing": "full PCM16/16kHz mono, reject >20s, API VAD trim, HF fp16 encoder/fp32 log_softmax",
                    "score_window": "each checkpoint canonical CTC alignment [first_frame-1,last_frame+3)",
                    "args": vars(args)}
        out.mkdir(parents=True, exist_ok=True)
        decoder = HFDecoder(args.ckpt)
        rows = [(split, row) for split, rs in data.items() for row in rs]
        rows.sort(key=lambda pair: pair[1]["frames"])
        predictions, start = [], time.monotonic()
        with open(predictions_path, "w", encoding="utf-8") as stream:
            for pos in range(0, len(rows), args.batch_size):
                batch = rows[pos:pos + args.batch_size]
                wavs = []
                for _, row in batch:
                    with wave.open(row["wav"], "rb") as wav:
                        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                            raise ValueError("expected PCM16 mono 16kHz: " + row["wav"])
                        if wav.getnframes() > 320000:
                            raise ValueError("refusing to truncate a >20s labeled recording: " + row["uid"])
                        signal = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768.0
                    trimmed, _ = trim(signal, 16000)
                    if len(trimmed) < 640:
                        raise ValueError("empty/too-short VAD output: " + row["uid"])
                    wavs.append(trimmed)
                lp, lengths = forward_lprobs(decoder, wavs)
                for i, (split, row) in enumerate(batch):
                    pred = evaluate_utterance(row, lp[i, :int(lengths[i])].numpy(), decoder, viterbi_align)
                    pred["split"] = split
                    predictions.append(pred)
                    stream.write(json.dumps(pred, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if pos // 50 != (pos + len(batch)) // 50 or pos + len(batch) == len(rows):
                    print("evaluated %d/%d elapsed=%.1fs" % (pos + len(batch), len(rows), time.monotonic() - start), flush=True)
        metadata["elapsed_sec"] = time.monotonic() - start
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        report = summarize(predictions, metadata)
    with open(out / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Summarize an exploratory context-cropping probe with speaker-separated calibration."""
import argparse
import collections
import hashlib
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def bucket(value):
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % 3


def metrics(y, score, threshold):
    det = score > threshold
    tp, fp = int((det & y).sum()), int((det & ~y).sum())
    return {"tp": tp, "fp": fp, "n_error": int(y.sum()), "n_ok": int((~y).sum()),
            "recall": float(det[y].mean()), "fpr": float(det[~y].mean()),
            "auc": float(roc_auc_score(y, score)), "threshold": float(threshold)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--score-field", choices=("scores", "wide_scores"), default="scores")
    args = ap.parse_args()
    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    names = sorted(records[0][args.score_field])
    result = {"n_sites": len(records), "n_speakers": len(set(r["spk"] for r in records)),
              "n_prompts": len(set(r["item"] for r in records)),
              "design": "calibration: SHA256(speaker)%3=0; test: remainder; prompt-stratified exploratory sample",
              "limitations": ["sample is not representative of service error prevalence",
                              "all compared methods use the same base encoder and known training prompts",
                              "methods compared on this test set; confirmation on new data required"],
              "score_field": args.score_field, "directions": {}, "baseline_reproduction": {}}
    original = np.array([r["cached_d1"] * (1 if r["canon"] == "ㅅ" else -1) for r in records])
    current = np.array([r["scores"]["full"] for r in records])
    result["baseline_reproduction"] = {"median_abs_score_delta": float(np.median(abs(original-current))),
                                       "max_abs_score_delta": float(abs(original-current).max())}
    rng = np.random.RandomState(231023)
    for canon in ("ㅅ", "ㅆ"):
        rs = [r for r in records if r["canon"] == canon]
        y = np.array([r["actual"] != canon for r in rs])
        calibration = np.array([bucket(r["spk"]) == 0 for r in rs])
        test = ~calibration
        scores = {name: np.array([r[args.score_field][name] for r in rs]) * (1 if canon == "ㅅ" else -1)
                  for name in names}
        d = {"calibration_n": int(calibration.sum()), "test_n": int(test.sum()), "methods": {}}
        for name, s in scores.items():
            d["methods"][name] = {str(fpr): metrics(y[test], s[test], np.quantile(s[calibration & ~y], 1-fpr))
                                   for fpr in (0.01, 0.05, 0.10)}
        flag = np.array([r["flag"] for r in rs], dtype=bool)
        d["existing_screen"] = {"recall": float(flag[test & y].mean()), "fpr": float(flag[test & ~y].mean())}
        # Paired speaker bootstrap for AUC differences, keeping each utterance's
        # method predictions together. Threshold selection uses calibration only.
        speakers = sorted(set(r["spk"] for r, keep in zip(rs, test) if keep))
        by_speaker = {s: np.array([i for i, r in enumerate(rs) if r["spk"] == s], dtype=int) for s in speakers}
        diffs = collections.defaultdict(list)
        for _ in range(args.bootstrap):
            idx = np.concatenate([by_speaker[s] for s in rng.choice(speakers, len(speakers), replace=True)])
            if len(set(y[idx])) != 2:
                continue
            base = roc_auc_score(y[idx], scores["full"][idx])
            for name in names:
                if name != "full":
                    diffs[name].append(float(roc_auc_score(y[idx], scores[name][idx]) - base))
        d["paired_auc_delta_95ci"] = {k: [float(v) for v in np.quantile(ds, [0.025, 0.5, 0.975])]
                                       for k, ds in diffs.items()}
        result["directions"][canon] = d
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

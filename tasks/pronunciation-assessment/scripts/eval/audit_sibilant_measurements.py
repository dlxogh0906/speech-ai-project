"""Measure missingness and estimator sensitivity before testing discrimination.

This audit deliberately does not train a classifier or score against TAG
consonant labels. Existing TAG vowel intervals are unvalidated anchors, and
assistant-reviewed intervals remain explicitly provisional.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

METRICS = ("h1h2_raw", "h1h2_corrected", "h1a1_corrected")


def median(values):
    a = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(a)) if a else None


def describe(values):
    a = [v for v in values if v is not None and np.isfinite(v)]
    return {"available_items": len(a), "median": median(a),
            "p90": float(np.percentile(a, 90)) if a else None,
            "maximum": max(a) if a else None}


def summarize_window(rows, offset, begin, end, require_containment):
    selected = [r for r in rows if begin <= r["time"] + offset < end]
    harmonics = {c: [] for c in ("5000", "5500", "6000")}
    per_frame_spreads = defaultdict(list)
    selected_voiced = 0
    quality = Counter()
    for row in selected:
        if not row["pitch_hz"]:
            continue
        selected_voiced += 1
        by_ceiling = {}
        for c, record in row["ceilings"].items():
            h = record.get("harmonics")
            if h is None:
                continue
            if require_containment and not (h["window_start"] + offset >= begin and h["window_end"] + offset <= end):
                continue
            harmonics[c].append(h)
            by_ceiling[c] = h
            if c == "5500":
                quality[h["formant_quality"]] += 1
        for metric in METRICS:
            values = [h[metric] for h in by_ceiling.values() if h.get(metric) is not None]
            if len(values) >= 2:
                per_frame_spreads[metric].append(max(values) - min(values))
    return {"interval_original_seconds": [begin, end], "require_window_containment": require_containment,
            "containment_scope": "direct harmonic FFT window only; F0 and formant estimators use surrounding context",
            "entire_estimation_pipeline_contained": False,
            "candidate_frames": len(selected), "voiced_candidate_frames": selected_voiced,
            "default_formant_quality_counts": dict(quality),
            "by_ceiling": {c: {metric: {"median": median([h.get(metric) for h in hs]),
                "frames": sum(h.get(metric) is not None for h in hs)} for metric in METRICS} for c, hs in harmonics.items()},
            "same_frame_ceiling_spread_db": {m: describe(per_frame_spreads[m]) for m in METRICS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--visual-landmarks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a measurement audit")
    manifest = a.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines()]
    run = json.loads((a.measurements / "run.json").read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != run["packet_sha256"]:
        ap.error("measurement packet differs")
    visual = {}
    if a.visual_landmarks:
        visual = {r["id"]: r for r in map(json.loads, a.visual_landmarks.read_text().splitlines())}
    records = []
    for item in items:
        measured = json.loads((a.measurements / "tracks" / (item["id"] + ".json")).read_text())
        anchor = item["tag_vowel"][0]
        windows = {}
        for shift in (-.010, 0., .010, .020):
            start = anchor + shift
            # Fixed 40ms inspection window. It may extend past the TAG vowel
            # end: the condition is exposed, never silently treated as pure V.
            key = "tag_shift_%+dms" % round(shift * 1000)
            windows[key] = summarize_window(measured["rows"], measured["offset"], start, start + .040, True)
        windows["tag_center_only_legacy"] = summarize_window(measured["rows"], measured["offset"], anchor, anchor + .040, False)
        v = visual.get(item["id"])
        if v:
            if v["source_sha256"] != item["source_sha256"]:
                ap.error("visual review source differs")
            if v.get("voicing_first") is not None:
                start = v["voicing_first"]
                windows["assistant_visual_first_voicing"] = summarize_window(measured["rows"], measured["offset"], start, start + .040, True)
        shifts = [windows["tag_shift_%+dms" % ms] for ms in (-10, 0, 10, 20)]
        sensitivity = {}
        for m in METRICS:
            values = [w["by_ceiling"]["5500"][m]["median"] for w in shifts]
            values = [v for v in values if v is not None]
            sensitivity[m] = {"available_anchor_shifts": len(values), "range_of_medians_db": max(values) - min(values) if len(values) >= 2 else None}
        records.append({"id": item["id"], "following_vowel": item["following_vowel"],
            "coupled_TAG_interval": not item["tag_intervals_isolated"],
            "TAG_vowel_duration_ms": 1000 * (item["tag_vowel"][1] - anchor),
            "has_40ms_within_TAG_vowel": anchor + .040 <= item["tag_vowel"][1] + 1e-9,
            "visual_review_status": v["review_status"] if v else None,
            "windows": windows, "anchor_sensitivity": sensitivity})
    groups = {}
    for vowel in sorted({r["following_vowel"] for r in records}):
        rr = [r for r in records if r["following_vowel"] == vowel]
        groups[vowel] = {"items": len(rr), "metrics": {}}
        for m in METRICS:
            base = [r["windows"]["tag_shift_+0ms"] for r in rr]
            groups[vowel]["metrics"][m] = {
                "available_items": sum(w["by_ceiling"]["5500"][m]["median"] is not None for w in base),
                "ceiling_spread_db_of_per_item_median_spreads": describe([w["same_frame_ceiling_spread_db"][m]["median"] for w in base]),
                "anchor_shift_range_db": describe([r["anchor_sensitivity"][m]["range_of_medians_db"] for r in rr])}
    result = {"status": "estimator_audit_not_classification_performance", "items": len(records),
        "units": "item counts, not independent frame observations", "by_vowel": groups,
        "limitations": ["TAG boundaries are unvalidated", "assistant reviews are not human gold",
                        "Window containment checks only the direct harmonic FFT window, not all inputs to the F0/formant estimators.",
                        "Default Burg formants use a 50ms physical window plus resampling/interpolation context; a 40ms harmonic summary is not exclusively estimated from those 40ms.",
                        "Praat pitch selection also uses global peak normalization and cross-frame path optimization.",
                        "ceiling agreement does not prove correct formants", "anchor shifts combine measurement sensitivity with genuine temporal change",
                        "no validity cutoff learned from this audit", "unmeasurable items remain in denominators"],
        "packet_sha256": run["packet_sha256"], "measurements_run_sha256": hashlib.sha256((a.measurements / "run.json").read_bytes()).hexdigest(),
        "visual_landmarks": {"path": str(a.visual_landmarks),
            "sha256": hashlib.sha256(a.visual_landmarks.read_bytes()).hexdigest(),
            "records": len(visual)} if a.visual_landmarks else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"items": len(records), "by_vowel": groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()

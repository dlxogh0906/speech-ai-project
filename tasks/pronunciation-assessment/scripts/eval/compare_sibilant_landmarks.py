"""Exploratory comparisons within the same prompt, site, and vowel context.

Existing TAG labels and provisional assistant boundaries are not new blinded
human ground truth. No thresholds, performance scores, or classifiers are fit.
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


def describe(values):
    xs = [x for x in values if x is not None and np.isfinite(x)]
    return {"total": len(values), "available": len(xs), "missing": len(values) - len(xs),
            "median": float(np.median(xs)) if xs else None,
            "minimum": min(xs) if xs else None, "maximum": max(xs) if xs else None}


def difference_ms(review, first, second):
    if review.get(first) is None or review.get(second) is None:
        return None, None
    lo1, hi1 = review[first + "_range"]
    lo2, hi2 = review[second + "_range"]
    return 1000 * (review[second] - review[first]), [1000 * (lo2 - hi1), 1000 * (hi2 - lo1)]


def scalar_window(track, begin, end, key, half_window):
    times = track["times"] + float(track["offset"])
    selected = (times - half_window >= begin) & (times + half_window <= end)
    vals = track[key][selected]
    vals = vals[np.isfinite(vals)]
    return {"frames": len(vals), "median": float(np.median(vals)) if len(vals) else None}


def cycle_check(review, measured):
    begin, end = review.get("first_regular_run_start"), review.get("first_regular_run_end")
    if begin is None or end is None:
        return {"status": "no_provisional_regular_window"}
    if not end > begin:
        raise ValueError("invalid regular-cycle window")
    implied = 3 / (end - begin)
    pitches = [r["pitch_hz"] for r in measured["rows"]
               if begin <= r["time"] + measured["offset"] <= end and r["pitch_hz"]]
    median = float(np.median(pitches)) if pitches else None
    ratio = implied / median if median else None
    u = review["regular_point_uncertainty_seconds"]
    duration = end - begin
    return {"status": "consistency_check_not_validated_F0", "assumed_cycles": 3,
            "duration_ms": 1000 * duration, "implied_F0_hz": implied,
            "endpoint_range_implied_F0_hz": [3 / (duration + 2 * u), 3 / (duration - 2 * u) if duration > 2 * u else None],
            "praat_median_F0_hz": median, "praat_frames": len(pitches),
            "implied_to_praat_ratio": ratio,
            "outside_20_percent_review_flag": ratio < .8 or ratio > 1.2 if ratio else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--visual-landmarks", type=Path, required=True)
    ap.add_argument("--measurement-audit", type=Path, required=True)
    ap.add_argument("--pcm-audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a comparison")
    manifest = a.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines()]
    visual = {r["id"]: r for r in map(json.loads, a.visual_landmarks.read_text().splitlines())}
    audit = json.loads(a.measurement_audit.read_text())
    packet_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    visual_sha = hashlib.sha256(a.visual_landmarks.read_bytes()).hexdigest()
    if audit["packet_sha256"] != packet_sha or audit["visual_landmarks"]["sha256"] != visual_sha:
        ap.error("audit provenance differs")
    measured_run = a.measurements / "run.json"
    if hashlib.sha256(measured_run.read_bytes()).hexdigest() != audit["measurements_run_sha256"]:
        ap.error("measurement run differs")
    acoustic = {r["id"]: r for r in audit["records"]}
    pcm_audit = json.loads(a.pcm_audit.read_text())
    if pcm_audit["provenance"].get(str(manifest)) != packet_sha:
        ap.error("PCM audit packet differs")
    pcm = {r["id"]: r for r in pcm_audit["records"]}
    records = []
    for item in items:
        if item["id"] not in visual:
            continue
        v = visual[item["id"]]
        if v["source_sha256"] != item["source_sha256"]:
            ap.error("visual source differs")
        total, total_range = difference_ms(v, "frication_start", "voicing_first")
        vf = v.get("voicing_first")
        w = acoustic[item["id"]]["windows"].get("assistant_visual_first_voicing")
        metrics = {"noise_onset_to_first_voicing_ms": total,
                   "TAG_anchor_to_first_voicing_ms": 1000 * (vf - item["tag_vowel"][0]) if vf is not None else None,
                   "first_to_regular_delay_ms": 1000 * (v["first_regular_run_start"] - vf) if vf is not None and v.get("first_regular_run_start") is not None else None}
        for metric in ("h1h2_raw", "h1h2_corrected", "h1a1_corrected"):
            metrics[metric + "_early40ms_db"] = w["by_ceiling"]["5500"][metric]["median"] if w else None
        tr = json.loads((a.measurements / "tracks" / (item["id"] + ".json")).read_text())
        cycle = cycle_check(v, tr)
        trajectory = {}
        if vf is not None:
            with np.load(a.measurements / "tracks" / (item["id"] + "_balance.npz")) as track:
                for offset in (-60, -40, -20, 0, 20):
                    begin, end = vf + offset / 1000, vf + (offset + 20) / 1000
                    trajectory[str(offset)] = {str(c): scalar_window(track, begin, end, "upper_lower_db_%d" % c, .005) for c in (2000, 3000, 4000)}
        records.append({"id": item["id"], "item": item["item"], "q": item["q"], "word": item["tag_word"][2],
            "norm_prompt": item["norm_prompt"], "canon": item["canon"], "existing_TAG_actual": item["actual"],
            "following_vowel": item["following_vowel"], "preceding_canonical": item["preceding_canonical"],
            "selection_role": item["selection_role"], "speaker": item["speaker"],
            "review_status": v["review_status"], "metrics": metrics,
            "noise_onset_to_first_voicing_endpoint_range_ms": total_range,
            "transition_status": v["transition_status"], "candidate_diffuse_noise_onset": v.get("frication_to_diffuse_noise"),
            "cycle_consistency": cycle, "balance_20ms_windows_relative_to_first_voicing": trajectory,
            "early40ms_window_detail": w, "pcm_observations": pcm[item["id"]]})
    grouped = defaultdict(list)
    for r in records:
        if r["selection_role"].startswith("focus_"):
            key = (r["norm_prompt"], r["q"], r["canon"], r["following_vowel"], r["preceding_canonical"])
            grouped[key].append(r)
    groups = []
    for key, rows in sorted(grouped.items()):
        groups.append({"context": {"prompt": key[0], "site": key[1], "canon": key[2], "following_vowel": key[3], "preceding_canonical": key[4], "item": rows[0]["item"], "word": rows[0]["word"]},
            "by_existing_TAG": {label: {"items": sum(r["existing_TAG_actual"] == label for r in rows),
                "metrics": {metric: describe([r["metrics"][metric] for r in rows if r["existing_TAG_actual"] == label]) for metric in rows[0]["metrics"]}}
                for label in sorted({r["existing_TAG_actual"] for r in rows})}, "ids": [r["id"] for r in rows]})
    result = {"status": "exploratory_existing_TAG_comparison_not_independent_accuracy", "reviewed_items": len(records),
        "focus_items": sum(r["selection_role"].startswith("focus_") for r in records),
        "transition_status_counts": dict(Counter(r["transition_status"] for r in records)),
        "cycle_review_flags": [r["id"] for r in records if r["cycle_consistency"].get("outside_20_percent_review_flag")],
        "limitations": ["Assistant boundaries are provisional and are not human gold.", "Existing TAG labels are not new blinded listener consensus.",
            "Same lexical context does not control speaker, prosody, rate, or recording channel.",
            "Total noise-onset-to-voicing time is neither pure frication nor aspiration duration.",
            "Endpoint ranges are visual uncertainty ranges, not statistical confidence intervals.",
            "Missing endpoints remain missing even when a possible-event range exists.",
            "Early40ms harmonic summaries can mix irregular and regular voice; formant correction remains unvalidated.",
            "Early40ms contains direct harmonic windows only; F0/formant estimates can use preceding and following audio.",
            "The canonical field originates in existing LAB phonemic transcription and is not independently validated standard pronunciation.",
            "Frame counts are not independent sample counts; all items remain in denominators.",
            "20 percent F0 disagreement is only a review flag; agreement is not validation."],
        "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest, a.visual_landmarks, a.measurement_audit, a.pcm_audit, measured_run, Path(__file__)]},
        "groups": groups, "records": records}
    a.out.mkdir(parents=True)
    (a.out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (a.out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    render(a.out / "duration_comparison.png", groups, records)
    print(json.dumps({"reviewed_items": len(records), "focus_items": result["focus_items"], "groups": groups}, ensure_ascii=False))


def render(path, groups, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {r["id"]: r for r in records}
    fig, axes = plt.subplots(1, len(groups), figsize=(16, 5), sharey=True, squeeze=False)
    item_counts = Counter(g["context"]["item"] for g in groups)
    item_variant = Counter()
    for ax, group in zip(axes[0], groups):
        for j, label in enumerate(("ㅅ", "ㅆ")):
            rows = [lookup[k] for k in group["ids"] if lookup[k]["existing_TAG_actual"] == label]
            valid = [r for r in rows if r["metrics"]["noise_onset_to_first_voicing_ms"] is not None]
            for dx, r in zip(np.linspace(-.13, .13, max(len(valid), 1)), valid):
                value = r["metrics"]["noise_onset_to_first_voicing_ms"]
                lo, hi = r["noise_onset_to_first_voicing_endpoint_range_ms"]
                ax.errorbar(j + dx, value, yerr=[[max(0, value - lo)], [max(0, hi - value)]], fmt="o", ms=4,
                            color="#b45436" if j == 0 else "#28718b", capsize=2)
            ax.text(j, .97, "%d/%d available" % (len(valid), len(rows)), ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["TAG s", "TAG ss"])
        item = group["context"]["item"]
        item_variant[item] += 1
        title = item
        if item_counts[item] > 1:
            title += "\nprompt %d / site %d" % (item_variant[item], group["context"]["site"])
        ax.set_xlim(-.4, 1.4); ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0][0].set_ylabel("Noise onset to first vowel voicing (ms)")
    fig.suptitle("Same prompt and site; provisional assistant landmarks; existing TAG labels")
    fig.text(.5, .01, "Bars show visual endpoint ranges. Missing cases retained in counts. No native-listener validation or accuracy claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

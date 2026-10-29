"""Offline equal-duration interventions: prevoicing noise versus early vowel.

This probes sensitivity of a frozen CTC model, not pronunciation accuracy or
isolated physiological cues. Provisional assistant landmarks and existing TAG
vowel endpoints are used only to define the diagnostic regions. No training.
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
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import time
import wave

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pcm(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError("expected mono PCM16 16kHz")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()


def intervene(x, begin, end, operator, sample_rate=16000, strength=1.):
    """Change only the requested samples, using a 5ms raised-cosine blend."""
    lo, hi = round(begin * sample_rate), round(end * sample_rate)
    if not 0 <= lo < hi <= len(x):
        raise ValueError("intervention lies outside input")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("intervention strength must be in [0, 1]")
    n, ramp = hi - lo, round(.005 * sample_rate)
    if n < 2 * ramp:
        raise ValueError("region too short for taper")
    weight = np.ones(n, dtype=np.float64)
    edge = .5 - .5 * np.cos(np.linspace(0, np.pi, ramp))
    weight[:ramp], weight[-ramp:] = edge, edge[::-1]
    segment = x[lo:hi].astype(np.float64)
    if operator == "attenuate_20db":
        replacement = segment * .1
    elif operator == "reverse":
        replacement = segment[::-1]
    else:
        raise ValueError("unknown intervention")
    y = x.copy()
    y[lo:hi] = (segment + strength * weight * (replacement - segment)).astype(x.dtype)
    return y


def constant_run(samples):
    if not len(samples):
        return 0
    cuts = np.r_[0, np.flatnonzero(np.diff(samples.astype(np.int32))) + 1, len(samples)]
    return int(np.diff(cuts).max())


def pair_scores(log_probs, tokens, position, s_id, ss_id, blank):
    import torch
    from torch.nn.functional import ctc_loss
    targets = torch.tensor([tokens, tokens], dtype=torch.long)
    targets[0, position], targets[1, position] = s_id, ss_id
    lp = torch.as_tensor(log_probs, dtype=torch.float32).unsqueeze(1).repeat(1, 2, 1)
    losses = ctc_loss(lp, targets, torch.tensor([len(lp)] * 2),
                      torch.tensor([len(tokens)] * 2), blank=blank, reduction="none", zero_infinity=False)
    values = (-losses).tolist()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite CTC target likelihood")
    return {"logP_s": values[0], "logP_ss": values[1], "ss_minus_s": values[1] - values[0]}


def summarize(run):
    """Describe paired perturbations without fitting a threshold or classifier."""
    run = Path(run)
    if (run / "summary.json").exists():
        raise FileExistsError("refusing to overwrite a diagnostic summary")
    plan = json.loads((run / "plan.json").read_text())
    complete = json.loads((run / "completed.json").read_text())
    if digest(run / "plan.json") != complete["plan_sha256"] or digest(run / "scores.jsonl") != complete["scores_sha256"]:
        raise ValueError("diagnostic artifacts changed")
    rows = [json.loads(s) for s in (run / "scores.jsonl").read_text().splitlines()]
    if len(rows) != len(plan["selected"]) or {r["id"] for r in rows} != {r["id"] for r in plan["selected"]}:
        raise ValueError("incomplete diagnostic items")
    regions = ["prevoicing_middle40", "vowel_initial40", "vowel_later40"]
    summaries = {}
    for operator in ("attenuate_20db", "reverse"):
        for suffix in (["", "__equal_squared_change"] if plan["method"].get("equal_squared_change") else [""]):
            values = {region: np.asarray([r["scores"][region + "__" + operator + suffix]["delta_ss_minus_s"] for r in rows])
                      for region in regions}
            first, early, late = [np.abs(values[region]) for region in regions]
            group = {"by_region": {k: {"median_signed_delta": float(np.median(v)),
                "median_absolute_delta": float(np.median(abs(v))), "minimum": float(v.min()), "maximum": float(v.max())}
                for k, v in values.items()}, "early_absolute_greater_than_prevoicing": int(sum(early > first)),
                "early_absolute_greater_than_later": int(sum(early > late)), "items": len(rows),
                "median_paired_absolute_early_minus_later": float(np.median(early - late))}
            if suffix:
                errors = []
                for row in rows:
                    energies = [row["scores"][region + "__" + operator + suffix]["whole_input_squared_difference"] for region in regions]
                    errors.append((max(energies) - min(energies)) / max(energies) if max(energies) else 0.)
                group["maximum_relative_squared_change_mismatch"] = max(errors)
            summaries[operator + suffix] = group
    summary = {"status": "frozen_model_region_sensitivity_not_accuracy_or_specific_cue_identification",
        "items": len(rows), "speakers": len({r["speaker"] for r in plan["selected"]}),
        "vowels": dict(Counter(r["following_vowel"] for r in rows)),
        "identity_repeat_max_absolute_delta": max(abs(r["scores"]["identity_repeat"]["delta_ss_minus_s"]) for r in rows),
        "comparisons": summaries, "limitations": plan["limitations"],
        "interpretation": ["These results test regional sensitivity, not whether the model correctly recognizes breathiness, aspiration, or H1-H2.",
            "A region affecting the margin is not proof that it improves pronunciation-error detection.",
            "Absolute perturbation-energy matching does not equate relative attenuation or perceptual change."],
        "provenance": {str(p): digest(p) for p in [run / "plan.json", run / "completed.json", run / "scores.jsonl", run / "source_snapshot.py", Path(__file__)]}}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2 if plan["method"].get("equal_squared_change") else 1, figsize=(11, 8), squeeze=False)
    for oi, operator in enumerate(("attenuate_20db", "reverse")):
        for ci, suffix in enumerate((["", "__equal_squared_change"] if axes.shape[1] == 2 else [""])):
            ax = axes[oi, ci]
            vals = np.array([[abs(r["scores"][region + "__" + operator + suffix]["delta_ss_minus_s"]) for region in regions] for r in rows])
            for values in vals:
                ax.plot(range(3), values, color="#8a94a0", alpha=.22, lw=.7)
            ax.plot(range(3), np.median(vals, axis=0), color="#a44428", marker="o", lw=2, label="Median")
            ax.set_xticks(range(3)); ax.set_xticklabels(["Prevoicing\nmiddle 40 ms", "Vowel onset\n0–40 ms", "Vowel later\n60–100 ms"])
            ax.set_ylabel("Absolute change in CTC logP(ss) − logP(s)")
            ax.set_title(("Attenuation" if oi == 0 else "Local reversal") + ("; equal squared signal change" if ci else "; full planned strength"))
            ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    fig.suptitle("Paired regional sensitivity of the frozen CTC model")
    roman = {"ㅓ": "eo", "ㅡ": "eu", "ㅏ": "a", "ㅣ": "i", "ㅜ": "u", "ㅗ": "o"}
    vowel_counts = ", ".join("%d %s" % (n, roman.get(v, v)) for v, n in summary["vowels"].items())
    caption = "%d selected items (%s); provisional assistant landmarks. Lines pair the same recording.\n" % (len(rows), vowel_counts)
    caption += "Prevoicing noise may include aspiration. These are unnatural interventions, not human labels or accuracy results."
    fig.text(.5, .025, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(run / "paired_region_sensitivity.png", dpi=180)
    fig.savefig(run / "paired_region_sensitivity.svg")
    plt.close(fig)
    (run / "summary_source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--precision", choices=("float16", "float32"), default="float32")
    ap.add_argument("--equal-squared-change", action="store_true", help="also match absolute perturbation energy across regions within each item/operator")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite a diagnostic run")
    manifest = a.packet / "items.private.jsonl"
    items = [json.loads(s) for s in manifest.read_text().splitlines()]
    landmarks = {r["id"]: r for r in map(json.loads, a.landmarks.read_text().splitlines())}
    cached = {}
    for line in a.baseline.open():
        record = json.loads(line)
        cached[record["row"]["uid"]] = record
    mapper_path = Path(__file__).with_name("eval_sibilant_directional.py")
    spec = importlib.util.spec_from_file_location("region_diagnostic_mapper", str(mapper_path))
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    selected, exclusions = [], []
    for item in items:
        if not item["selection_role"].startswith("focus_"):
            exclusions.append({"id": item["id"], "reason": "outside_focus_cohort"})
            continue
        review = landmarks[item["id"]]
        if review["source_sha256"] != item["source_sha256"]:
            raise ValueError("landmark source differs")
        fs, vf = review.get("frication_start"), review.get("voicing_first")
        reason = ("missing_provisional_landmark" if fs is None or vf is None
                  else "prevoicing_shorter_than_80ms" if vf - fs < .08
                  else "voicing_TAG_anchor_disagreement_over_20ms" if abs(vf - item["tag_vowel"][0]) > .02
                  else "TAG_vowel_does_not_cover_later_control" if item["tag_vowel"][1] - vf < .105 else None)
        if reason:
            exclusions.append({"id": item["id"], "reason": reason})
            continue
        capture = cached[item["uid"]]
        row = capture["row"]
        canon = (capture.get("raw") or {}).get("canon")
        if not canon:
            exclusions.append({"id": item["id"], "reason": "cached_input_rejected_or_no_canonical"})
            continue
        mapping = mapper.all_optimal_mapping(row["canonical"], canon)[item["q"]]
        if len(mapping) != 1 or None in mapping:
            exclusions.append({"id": item["id"], "reason": "ambiguous_canonical_target_mapping"})
            continue
        q = next(iter(mapping))
        if canon[q] != item["canon"]:
            exclusions.append({"id": item["id"], "reason": "canonical_target_disagreement"})
            continue
        source = load_pcm(item["source_wav"])
        if digest(item["source_wav"]) != item["source_sha256"]:
            raise ValueError("source audio changed")
        pcm = load_pcm(row["wav"])
        offset = row.get("source_segment", [0])[0]
        s0 = round(offset * 16000)
        if not np.array_equal(pcm, source[s0:s0 + len(pcm)]):
            raise ValueError("evaluation PCM is not the declared source segment")
        center = (fs + vf) / 2
        regions = {"prevoicing_middle40": [center - .020 - offset, center + .020 - offset],
                   "vowel_initial40": [vf - offset, vf + .040 - offset],
                   "vowel_later40": [vf + .060 - offset, vf + .100 - offset]}
        if any(not 0 <= lo < hi <= len(pcm) / 16000 for lo, hi in regions.values()):
            exclusions.append({"id": item["id"], "reason": "region_outside_evaluation_audio"})
            continue
        runs = {k: constant_run(pcm[round(lo * 16000):round(hi * 16000)]) for k, (lo, hi) in regions.items()}
        if max(runs.values()) >= 16:
            exclusions.append({"id": item["id"], "reason": "constant_PCM_run_at_least_1ms_in_changed_region", "runs_samples": runs})
            continue
        selected.append({"id": item["id"], "uid": item["uid"], "speaker": item["speaker"],
            "item": item["item"], "following_vowel": item["following_vowel"], "existing_TAG_actual": item["actual"],
            "source_wav": item["source_wav"], "source_sha256": item["source_sha256"],
            "input_wav": row["wav"], "input_sha256": digest(row["wav"]), "source_offset_seconds": offset,
            "input_duration_seconds": len(pcm) / 16000, "canonical": canon, "target_position": q,
            "regions_relative_seconds": regions, "landmark_status": review["review_status"]})
    if not selected:
        raise ValueError("no eligible diagnostic items")
    a.out.mkdir(parents=True)
    plan = {"status": "frozen_before_model_inference", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hypothesis": "Compare sensitivity to equal-duration prevoicing noise and early/later vowel regions; no assumed winning cue.",
        "method": {"region_ms": 40, "taper_ms_each_edge": 5, "attenuation_gain": .1,
            "later_control_offset_ms": [60, 100], "operators": ["attenuate_20db", "reverse"],
            "input": "original evaluation PCM segment; no VAD, resampling, time removal, or added noise",
            "score": "whole-sequence CTC logP(target ss) minus logP(target s); other target tokens identical and frozen",
            "identity_repeat": True, "model_training": False},
        "limitations": ["Assistant landmarks and TAG vowel endpoints are provisional, not expert truth.",
            "Prevoicing noise can include aspiration; it is not a pure frication annotation.",
            "Perturbations are unnatural and do not isolate a single articulatory or phonatory cue.",
            "Early versus later effects are descriptive; unequal removed energy is recorded, not assumed controlled.",
            "No causal claim about human perception and no pronunciation classification accuracy.",
            "Eligibility favors long vowels with close TAG/visual anchor agreement; not all Korean vowel contexts.",
            "No VAD makes this a model diagnostic, not exact end-to-end serving reproduction."],
        "items_considered": len(items), "selected": selected, "exclusions": exclusions,
        "provenance": {str(p): digest(p) for p in [manifest, a.landmarks, a.baseline, Path(__file__), mapper_path]},
        "checkpoint": {str(p.relative_to(a.ckpt)): digest(p) for p in sorted(a.ckpt.iterdir()) if p.is_file()}}
    plan["method"].update(precision=a.precision, tf32=False,
        equal_squared_change=a.equal_squared_change,
        equal_squared_change_definition="Within each item and operator, blend strength sqrt(min(region squared change)/region squared change). Perceptual salience is not equated.")
    (a.out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (a.out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"selected": len(selected), "exclusions": dict(Counter(r["reason"] for r in exclusions))}), flush=True)
    import torch
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dec = HFDecoder(str(a.ckpt))
    if a.precision == "float32":
        dec.model.float()
    elif dec.device != "cuda":
        raise ValueError("float16 diagnostic requires CUDA")
    s_id, ss_id = dec.d.index("ㅅ"), dec.d.index("ㅆ")
    started = time.monotonic()
    with (a.out / "scores.jsonl").open("w") as stream:
        for index, record in enumerate(selected):
            if digest(record["input_wav"]) != record["input_sha256"]:
                raise ValueError("input file changed after plan was frozen")
            x = load_pcm(record["input_wav"]).astype(np.float32) / 32768.
            before = hashlib.sha256(x.tobytes()).hexdigest()
            variants = [("original", x), ("identity_repeat", x.copy())]
            strengths = {"original": 0., "identity_repeat": 0.}
            for region, (lo, hi) in record["regions_relative_seconds"].items():
                for operator in ("attenuate_20db", "reverse"):
                    name = region + "__" + operator
                    variants.append((name, intervene(x, lo, hi, operator)))
                    strengths[name] = 1.
            if a.equal_squared_change:
                for operator in ("attenuate_20db", "reverse"):
                    energies = {name.split("__")[0]: float(np.sum((y.astype(np.float64) - x) ** 2))
                                for name, y in variants if name.endswith("__" + operator)}
                    target_energy = min(energies.values())
                    for region, (lo, hi) in record["regions_relative_seconds"].items():
                        strength = np.sqrt(target_energy / energies[region]) if energies[region] > 0 else 0.
                        name = region + "__" + operator + "__equal_squared_change"
                        variants.append((name, intervene(x, lo, hi, operator, strength=strength)))
                        strengths[name] = float(strength)
            if hashlib.sha256(x.tobytes()).hexdigest() != before:
                raise ValueError("source memory mutated")
            input_hashes = [hashlib.sha256(y.tobytes()).hexdigest() for _, y in variants]
            lp, lengths = forward_lprobs(dec, [y for _, y in variants])
            if input_hashes != [hashlib.sha256(y.tobytes()).hexdigest() for _, y in variants]:
                raise ValueError("feature extraction mutated a diagnostic input")
            if len(set(lengths)) != 1:
                raise ValueError("same-length inputs produced different output lengths")
            tokens = [dec.d.index(t) for t in record["canonical"]]
            if dec.d.unk() in tokens:
                raise ValueError("unknown canonical token")
            scores = {}
            for k, (name, y) in enumerate(variants):
                score = pair_scores(lp[k, :lengths[k]].numpy(), tokens, record["target_position"], s_id, ss_id, dec.blank)
                score["whole_input_squared_difference"] = float(np.sum((y.astype(np.float64) - x) ** 2))
                score["whole_input_energy_change"] = float(np.sum(y.astype(np.float64) ** 2) - np.sum(x.astype(np.float64) ** 2))
                score["intervention_strength"] = strengths[name]
                scores[name] = score
            original = scores["original"]["ss_minus_s"]
            for score in scores.values():
                score["delta_ss_minus_s"] = score["ss_minus_s"] - original
            output = {"id": record["id"], "item": record["item"], "following_vowel": record["following_vowel"],
                      "existing_TAG_actual": record["existing_TAG_actual"], "scores": scores}
            stream.write(json.dumps(output, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            np.savez_compressed(a.out / (record["id"] + "_logprobs.npz"), names=np.asarray([n for n, _ in variants]),
                                logprobs=lp[:, :lengths[0]].numpy())
            print("%d/%d %.1fs" % (index + 1, len(selected), time.monotonic() - started), flush=True)
    (a.out / "completed.json").write_text(json.dumps({"items": len(selected), "elapsed_seconds": time.monotonic() - started,
        "plan_sha256": digest(a.out / "plan.json"), "torch_version": torch.__version__, "device": str(dec.device),
        "scores_sha256": digest(a.out / "scores.jsonl")}, indent=2) + "\n")
    summarize(a.out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Choose observation-adding on valid, then evaluate a frozen test candidate.

Reuses the service checkpoint, production phone flag rule and earlier synthetic
noise helpers. Only TSV/LTR/CANON are read; no app uploads or speaker metadata.
The original weight is alpha: alpha=1 is baseline, alpha=0 is RNNoise only.
"""

import sys as _sys
import pathlib as _pl

_root = _pl.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
_sys.path[:0] = [str(_root), *(str(p) for p in sorted(_root.glob("tasks/*/src")))]

import argparse
import hashlib
import json
import random
import time
from types import SimpleNamespace

import numpy as np

import compare_noise_preprocess as previous
from summarize_noise_preprocess import summarize
from shared.paths import DATA_ROOT, MDD_OUTPUTS

ALPHAS = (1.0, 0.0, 0.25, 0.5, 0.75)
METHOD = {1.0: "baseline", 0.0: "rnnoise", 0.25: "oa_025",
          0.5: "oa_050", 0.75: "oa_075"}
PRIMARY = ("babble10", "babble5")
SAFETY = ("clean", "babble20", "babble10", "babble5", "rumble10", "rumble0",
          "babble10+webrtc", "babble5+webrtc")
COUNTS = ("api_tp", "api_fp", "raw_tp", "raw_fp", "preserved", "reverted",
          "jer_num", "insertions", "deletions", "prediction_sha256")


def blend(original, enhanced, alpha):
    """Mix aligned equal-gain inputs without normalizing away attenuation."""
    if original.shape != enhanced.shape or not 0 <= alpha <= 1:
        raise ValueError("Invalid aligned input/weight")
    result = original if alpha == 1 else enhanced if alpha == 0 else (
        alpha * original + (1 - alpha) * enhanced)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1.00001:
        raise ValueError("Nonfinite or clipped observation-adding signal")
    return np.asarray(result, dtype=np.float32)


def load_split(manifest, split):
    root, rows = previous.read_rows(manifest, split)
    directory = _pl.Path(manifest)
    tags = (directory / (split + ".ltr")).read_text(encoding="utf-8").splitlines()
    canons = (directory / (split + ".canon")).read_text(encoding="utf-8").splitlines()
    if not len(rows) == len(tags) == len(canons):
        raise RuntimeError("Manifest/label count mismatch")
    return root, rows, tags, canons


def choose_samples(data, normal, errors, seed, excluded):
    root, rows, tags, canons = data
    eligible = [row if str(root / row[0]) not in excluded else [row[0], "0"]
                for row in rows]
    selection, stats = previous.select_samples(
        SimpleNamespace(normal=normal, errors=errors, seed=seed), eligible, root, tags, canons)
    paths = {str(root / rows[i][0]) for i, _ in selection}
    if len(paths) != len(selection):
        raise RuntimeError("Duplicate selected source audio")
    return selection, stats, paths


def noisy_conditions(wav, row_index, seed, pool):
    babble = previous.make_babble(random.Random(seed * 1000003 + row_index), pool, len(wav))
    rumble = previous.make_rumble(seed * 7919 + row_index, len(wav))
    for condition, kind, snr, browser in previous.CONDITIONS:
        noisy = wav if kind is None else previous.mix(wav, babble if kind == "babble" else rumble, snr)
        if browser:
            noisy = previous.webrtc(noisy)
        yield condition, noisy


def diagnose(decoder, probabilities, canon, tag, duration, formatter):
    from mdd.gop_feats_v2 import utt_features
    from mdd.jamo_to_kog2p import VOWEL
    from mdd.session import phone_context

    targets = [decoder.d.index(t) for t in canon]
    if decoder.d.unk() in targets:
        raise RuntimeError("Canonical token outside model vocabulary")
    _, extra = utt_features(decoder, probabilities, targets, canon, duration)
    predicted = [t for t in extra["pred"] if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
    result = previous.phone_diagnostics(canon, tag, predicted, extra["gops_m"],
                                        formatter, set(VOWEL), phone_context)
    result["prediction_sha256"] = hashlib.sha256("".join(predicted).encode()).hexdigest()
    return result


def infer(decoder, inputs, canon, tag, formatter, batch_size):
    from mdd.gop_feats_v2 import forward_lprobs

    results = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start:start + batch_size]
        started = time.perf_counter()
        probabilities, lengths = forward_lprobs(decoder, chunk)
        elapsed_ms = (time.perf_counter() - started) * 1000
        for j, wav in enumerate(chunk):
            result = diagnose(decoder, probabilities[j, :int(lengths[j])].numpy(),
                              canon, tag, len(wav) / previous.SR, formatter)
            result.update(model_ms=elapsed_ms / len(chunk), model_batch_ms=elapsed_ms,
                          model_batch_size=len(chunk))
            results.append(result)
    return results


def verify_batch(decoder, formatter, denoise, data, selection, seed, pool, batch_size):
    """Fallback to single-input inference if the smoke pairs change diagnostics."""
    from mdd.vad import trim

    if batch_size == 1:
        return 1, {"effective_batch_size": 1, "mode": "production single-input inference; no batching"}
    root, rows, tags, canons = data
    checked = different = 0
    for i, _ in selection[:4]:
        wav = previous.read_wav(root / rows[i][0])
        for _, noisy in noisy_conditions(wav, i, seed, pool):
            enhanced, _ = denoise(noisy)
            inputs = [trim(blend(noisy, enhanced, a), previous.SR)[0] for a in ALPHAS]
            batched = infer(decoder, inputs, canons[i].split(), tags[i].split(), formatter, batch_size)
            single = infer(decoder, inputs, canons[i].split(), tags[i].split(), formatter, 1)
            for a, b in zip(batched, single):
                checked += 1
                different += any(a[key] != b[key] for key in COUNTS)
    effective = batch_size if different == 0 else 1
    result = {"records_checked": checked, "different_diagnostics": different,
              "requested_batch_size": batch_size, "effective_batch_size": effective,
              "scope": "Four valid utterances, all conditions/weights; counts and decoded sequence"}
    print("[batch parity] " + json.dumps(result), flush=True)
    return effective, result


def run_split(args, directory, split, data, selection, stats, seed, pool, alphas,
              decoder, formatter, denoise, batch_size, parity):
    from mdd.vad import trim

    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "utterances.jsonl"
    if result_path.exists():
        raise RuntimeError("Refusing to overwrite experiment results")
    metadata = {"args": vars(args), "split": split, "seed": seed,
                "n_utts": len(selection), "selection": stats,
                "methods": [METHOD[a] for a in alphas], "alphas_original": list(alphas),
                "conditions": previous.CONDITIONS,
                "source_row_indices": [i for i, _ in selection],
                "scope": "GPU fp16 service checkpoint, fixed production phone-flag rule; synthetic noise; no HTTP uploads",
                "snr_definition": "whole clean-utterance RMS; matches prior experiment",
                "mix_definition": "alpha * original + (1-alpha) * delay-compensated RNNoise; no independent gain normalization",
                "latency_note": "CPU denoising reused across weights; model_ms amortized over batch; not request latency",
                "batch_parity": parity, "noise_pool_count": len(pool),
                "rnnoise_library_sha256": previous.digest(args.rnnoise_library),
                "calibration_sha256": previous.digest(args.calib),
                "source_hashes": {str(p.relative_to(_root)): previous.digest(p) for p in [
                    _pl.Path(__file__).resolve(), _pl.Path(previous.__file__).resolve(),
                    _pl.Path(__file__).with_name("summarize_noise_mixing.py").resolve(),
                    _pl.Path(__file__).with_name("summarize_noise_preprocess.py").resolve(),
                    _root / "tasks/pronunciation-assessment/src/mdd/noise_preprocess.py",
                    _root / "tasks/pronunciation-assessment/src/mdd/gop_feats_v2.py",
                    _root / "tasks/pronunciation-assessment/src/mdd/vad.py",
                    _root / "tasks/pronunciation-assessment/src/mdd/speechsuper.py"]}}
    (directory / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    root, rows, tags, canons = data
    started = time.monotonic()
    with result_path.open("w", encoding="utf-8") as f:
        for k, (i, cohort) in enumerate(selection):
            wav = previous.read_wav(root / rows[i][0])
            for condition, noisy in noisy_conditions(wav, i, seed, pool):
                enhanced, denoise_info = denoise(noisy)
                original_check = previous.audio_check(noisy)
                prepared = []
                for alpha in alphas:
                    t0 = time.perf_counter()
                    transformed = blend(noisy, enhanced, alpha)
                    processed, trim_info = trim(transformed, previous.SR)
                    prepared.append((processed, previous.audio_check(transformed), trim_info,
                                     (time.perf_counter() - t0) * 1000))
                results = infer(decoder, [p[0] for p in prepared], canons[i].split(), tags[i].split(),
                                formatter, batch_size)
                for alpha, record, (processed, post_check, trim_info, mix_ms) in zip(alphas, results, prepared):
                    prep_ms = mix_ms + (denoise_info["preprocess_ms"] if alpha != 1 else 0)
                    record.update(k=k, cohort=cohort, condition=condition, method=METHOD[alpha],
                                  alpha_original=alpha, original_sec=len(wav) / previous.SR,
                                  processed_sec=len(processed) / previous.SR,
                                  original_check_pass=original_check["pass"],
                                  post_denoise_check_pass=post_check["pass"],
                                  silero_check_pass=None, silero_no_speech=False,
                                  prep_ms=prep_ms, total_ms=prep_ms + record["model_ms"],
                                  cut_sec=trim_info.get("cut_sec", 0),
                                  rnnoise_delay_ms=denoise_info["delay_compensation_ms"])
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k + 1 == len(selection):
                elapsed = time.monotonic() - started
                remaining = elapsed / (k + 1) * (len(selection) - k - 1)
                print("[progress %s] %d/%d elapsed=%.1fs remaining=%.1fs" %
                      (split, k + 1, len(selection), elapsed, remaining), flush=True)
    result = summarize(directory, args.bootstraps)
    print("[complete %s] %.1fs" % (split, time.monotonic() - started), flush=True)
    return result


def select_alpha(summary):
    """Predeclared safety constraints; primary objective is babble false flags."""
    metrics = summary["metrics"]
    candidates = []
    for alpha in (0.25, 0.5, 0.75):
        method = METHOD[alpha]
        deltas = summary["paired_vs_baseline"]
        safe = all(deltas[c][method]["preservation"]["delta_pp"] >= -2.0
                   and deltas[c][method]["api_recall"]["delta_pp"] >= -2.0 for c in SAFETY)
        safe = safe and deltas["clean"][method]["api_frr"]["delta_pp"] <= 0.3
        primary_frr = float(np.mean([metrics[c][method]["api_frr"] for c in PRIMARY]))
        primary_delta = float(np.mean([deltas[c][method]["api_frr"]["delta_pp"] for c in PRIMARY]))
        candidates.append({"alpha": alpha, "method": method, "valid_safety_pass": safe,
                           "primary_frr": primary_frr, "primary_frr_delta_pp": primary_delta})
    eligible = [c for c in candidates if c["valid_safety_pass"]]
    # If every mixture violates safety, still freeze a conservative diagnostic
    # candidate for test; it is explicitly not a deployment recommendation.
    chosen = min(eligible, key=lambda c: (c["primary_frr"], -c["alpha"])) if eligible else candidates[-1]
    return {"selection_split": "valid only", "primary_conditions": PRIMARY,
            "policy": "Minimize macro babble10/babble5 API false flags; per safety condition recall/preservation drop <=2pp; clean false flags increase <=0.3pp; tie favors more original",
            "safety_conditions": SAFETY, "candidates": candidates, "chosen": chosen,
            "validation_supports_use": bool(eligible and chosen["primary_frr_delta_pp"] < 0),
            "test_policy": "Evaluate frozen candidate plus baseline/RNNoise endpoints; never retune on test"}


def main():
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    from mdd.noise_preprocess import RNNoise16k
    from mdd.speechsuper import SpeechSuperFormatter

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    ap.add_argument("--noise-manifest", default=str(MDD_OUTPUTS / "manifest_kspon_stage1"))
    ap.add_argument("--ckpt", default=str(DATA_ROOT / "checkpoints/w2vbert_stage2/step_38000"))
    ap.add_argument("--norms", default=str(MDD_OUTPUTS / "score_phoneme/norms.json"))
    ap.add_argument("--calib", default=str(MDD_OUTPUTS / "score_phoneme/calibration.json"))
    ap.add_argument("--rnnoise-library", required=True)
    ap.add_argument("--exclude-prior", required=True, help="Previous experiment.json; exclude its test samples")
    ap.add_argument("--valid-per-cohort", type=int, default=150)
    ap.add_argument("--test-per-cohort", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--bootstraps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if min(args.valid_per_cohort, args.test_per_cohort, args.batch_size, args.bootstraps) < 1:
        ap.error("Counts and batch size must be positive")
    out = _pl.Path(args.out)
    if (out / "selection.json").exists() or (out / "valid/utterances.jsonl").exists():
        raise RuntimeError("Refusing to overwrite this run")
    out.mkdir(parents=True, exist_ok=True)
    valid_data = load_split(args.manifest, "valid")
    test_data = load_split(args.manifest, "test")
    prior = json.loads(_pl.Path(args.exclude_prior).read_text(encoding="utf-8"))
    if _pl.Path(prior["args"]["manifest"]).resolve() != _pl.Path(args.manifest).resolve():
        raise RuntimeError("Prior exclusion manifest mismatch")
    excluded = {str(test_data[0] / test_data[1][i][0]) for i in prior["source_row_indices"]}
    valid_sel, valid_stats, valid_paths = choose_samples(
        valid_data, args.valid_per_cohort, args.valid_per_cohort, args.seed, excluded)
    test_sel, test_stats, test_paths = choose_samples(
        test_data, args.test_per_cohort, args.test_per_cohort, args.seed + 1, excluded | valid_paths)
    assert not valid_paths & test_paths and not (valid_paths | test_paths) & excluded
    noise_root, noise_rows = previous.read_rows(args.noise_manifest, "valid")
    noise_pool = list(dict.fromkeys(noise_root / row[0] for row in noise_rows))
    random.Random(args.seed).shuffle(noise_pool)
    middle = len(noise_pool) // 2
    valid_noise, test_noise = noise_pool[:middle], noise_pool[middle:]
    if not valid_noise or not test_noise or set(valid_noise) & set(test_noise):
        raise RuntimeError("Invalid noise holdout")
    audit = {"valid_samples": len(valid_sel), "test_samples": len(test_sel),
             "prior_excluded_samples": len(excluded), "selected_audio_path_overlap": 0,
             "valid_noise_files": len(valid_noise), "test_noise_files": len(test_noise),
             "noise_audio_path_overlap": 0, "speaker_overlap": "Not audited; speaker metadata is not read",
             "endpoint_identity": "alpha=1 returns original exactly; alpha=0 returns RNNoise exactly",
             "gain_policy": "No separate gain normalization; preserve enhancer attenuation and phase"}
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("[split audit] " + json.dumps(audit), flush=True)
    decoder = HFDecoder(args.ckpt)
    formatter = SpeechSuperFormatter(args.norms, calib_path=args.calib)
    if formatter.cal is None:
        raise RuntimeError("Production calibration not loaded")
    denoise = RNNoise16k(args.rnnoise_library, delay_frames=2)
    forward_lprobs(decoder, [np.zeros(2 * previous.SR, np.float32)])
    batch_size, parity = verify_batch(decoder, formatter, denoise, valid_data,
                                     valid_sel, args.seed, valid_noise, args.batch_size)
    (out / "batch_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    valid = run_split(args, out / "valid", "valid", valid_data, valid_sel, valid_stats,
                      args.seed, valid_noise, ALPHAS, decoder, formatter, denoise, batch_size, parity)
    selection = select_alpha(valid)
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[frozen selection] " + json.dumps(selection, ensure_ascii=False), flush=True)
    test = run_split(args, out / "test", "test", test_data, test_sel, test_stats,
                     args.seed + 1, test_noise, (1.0, 0.0, selection["chosen"]["alpha"]),
                     decoder, formatter, denoise, batch_size, parity)
    print("[complete all] %d valid + %d test records" % (valid["n_records"], test["n_records"]), flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Paired baseline/Silero/RNNoise evaluation with human actual-phone labels.

Uses TSV, LTR and CANON only; never reads speaker metadata or app uploads.
Audio and synthetic mixtures stay in memory. Results contain anonymous sample
numbers, counts and aggregate diagnostics, not recording paths or identities.
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
import collections
import hashlib
import json
import random
import time
import wave

import numpy as np

from shared.paths import DATA_ROOT, MDD_OUTPUTS

SR = 16000
METHODS = ("baseline", "silero", "rnnoise")
CONDITIONS = (
    ("clean", None, None, False),
    ("babble20", "babble", 20, False),
    ("babble10", "babble", 10, False),
    ("babble5", "babble", 5, False),
    ("babble0", "babble", 0, False),
    ("rumble10", "rumble", 10, False),
    ("rumble0", "rumble", 0, False),
    ("babble10+webrtc", "babble", 10, True),
    ("babble5+webrtc", "babble", 5, True),
)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            raise ValueError("Expected mono PCM16 at 16 kHz")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def read_rows(directory, split):
    with open(_pl.Path(directory) / (split + ".tsv"), encoding="utf-8") as f:
        root = _pl.Path(f.readline().rstrip("\n"))
        rows = [line.rstrip("\n").split("\t") for line in f]
    return root, rows


def axis(canon, sequence):
    import edlib

    a, b = "".join(canon), "".join(sequence)
    if not b:
        return [None] * len(canon), 0
    result = edlib.getNiceAlignment(edlib.align(a, b, task="path"), a, b)
    aligned = [None if y == "-" else y
               for x, y in zip(result["query_aligned"], result["target_aligned"])
               if x != "-"]
    inserted = result["query_aligned"].count("-")
    return aligned, inserted


def audio_check(wav):
    from mdd.vad import voiced_mask

    mask, _ = voiced_mask(wav, SR)
    ratio = float(mask.mean()) if mask.size else 0.0
    seconds = float(mask.sum()) * 0.03
    return {"pass": seconds >= 1.0 and ratio >= 0.30,
            "voiced_sec": seconds, "voiced_ratio": ratio}


def make_babble(rng, pool, n):
    noise = np.zeros(n, np.float32)
    for _ in range(6):
        voice = read_wav(rng.choice(pool))
        while len(voice) < n:
            voice = np.concatenate((voice, read_wav(rng.choice(pool))))
        start = rng.randrange(len(voice) - n + 1)
        noise += voice[start:start + n] * rng.uniform(0.6, 1.0)
    return noise


def make_rumble(seed, n):
    from scipy.signal import lfilter

    rng = np.random.RandomState(seed % (2 ** 32))
    a = np.exp(-2 * np.pi * 300 / SR)
    filtered = lfilter([1 - a], [1, -a], rng.randn(n + 4000))[4000:]
    filtered = filtered / (filtered.std() + 1e-9) + 0.1 * rng.randn(n)
    return filtered.astype(np.float32)


def mix(wav, noise, snr):
    # Same whole-utterance RMS convention as the earlier Claude experiment.
    speech_power = float(np.mean(wav ** 2)) + 1e-10
    noise_power = float(np.mean(noise ** 2)) + 1e-10
    gain = np.sqrt(speech_power / (noise_power * 10 ** (snr / 10.0)))
    result = wav + gain * noise
    result = result / max(1.0, float(np.max(np.abs(result))) / 0.99)
    return result.astype(np.float32)


def webrtc(wav):
    from webrtc_noise_gain import AudioProcessor

    processor = AudioProcessor(0, 3)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    n = ((len(pcm) + 159) // 160) * 160
    padded = np.pad(pcm, (0, n - len(pcm)))
    chunks = [processor.Process10ms(padded[i:i + 160].tobytes()).audio
              for i in range(0, n, 160)]
    return np.frombuffer(b"".join(chunks), np.int16).astype(np.float32)[:len(wav)] / 32768.0


def select_samples(args, rows, root, tags, canons):
    """Random strata, then clean-input eligibility; never select by predictions."""
    pool = {"normal": [], "error": []}
    for i, row in enumerate(rows):
        if not 1.5 * SR <= int(row[1]) <= 10 * SR:
            continue
        tag, canon = tags[i].split(), canons[i].split()
        if not canon or not tag or any(len(t) != 1 for t in canon + tag):
            continue
        if any(not 0x3131 <= ord(t) <= 0x3163 for t in canon + tag):
            continue
        cohort = "normal" if tag == canon else "error"
        pool[cohort].append(i)
    rng = random.Random(args.seed)
    selected = []
    stats = {}
    for cohort, target in (("normal", args.normal), ("error", args.errors)):
        rng.shuffle(pool[cohort])
        scanned = rejected = insertion_only = 0
        accepted = 0
        for i in pool[cohort]:
            if accepted == target:
                break
            scanned += 1
            wav = read_wav(root / rows[i][0])
            if not audio_check(wav)["pass"]:
                rejected += 1
                continue
            canon, tag = canons[i].split(), tags[i].split()
            tag_axis, _ = axis(canon, tag)
            if cohort == "error" and not any(a != b for a, b in zip(canon, tag_axis)):
                insertion_only += 1
                continue
            selected.append((i, cohort))
            accepted += 1
        stats[cohort] = {"candidate_count": len(pool[cohort]), "scanned": scanned,
                         "low_speech_rejected": rejected,
                         "insertion_only_excluded": insertion_only, "selected": accepted}
        if accepted != target:
            raise RuntimeError("Not enough clean-eligible " + cohort + " utterances")
    rng.shuffle(selected)
    return selected, stats


def phone_diagnostics(canon, tag, predicted, margins, formatter, vowels, context):
    """Use actual production score/FLAG_MAX rules on canonical phone positions."""
    import edlib
    from mdd.speechsuper import FLAG_MAX

    reference, tag_insertions = axis(canon, tag)
    hypothesis, insertions = axis(canon, predicted)
    truth_error = np.array([a != b for a, b in zip(canon, reference)])
    raw_flag = np.array([a != b for a, b in zip(canon, hypothesis)])
    production_flag = []
    scores = []
    for i, (expected, actual) in enumerate(zip(canon, hypothesis)):
        margin = margins[i] if i < len(margins) else None
        score = (formatter._score(context(canon, i, vowels), margin)
                 if actual is not None else None)
        # Same removal of errorType as SpeechSuperFormatter.build(). Deletions
        # have no phone score and remain flagged in the current implementation.
        flag = actual != expected and (score is None or score < FLAG_MAX)
        production_flag.append(flag)
        scores.append(score)
    production_flag = np.array(production_flag, dtype=bool)
    result = {"n_err": int(truth_error.sum()), "n_ok": int((~truth_error).sum()),
              "raw_tp": int((raw_flag & truth_error).sum()),
              "raw_fp": int((raw_flag & ~truth_error).sum()),
              "api_tp": int((production_flag & truth_error).sum()),
              "api_fp": int((production_flag & ~truth_error).sum()),
              "preserved": sum(h == t for h, t, e in zip(hypothesis, reference, truth_error) if e),
              "reverted": sum(h == c for h, c, e in zip(hypothesis, canon, truth_error) if e),
              "jer_num": edlib.align("".join(tag), "".join(predicted))["editDistance"],
              "jer_den": len(tag), "n_canon": len(canon), "n_pred": len(predicted),
              "insertions": insertions, "tag_insertions": tag_insertions,
              "deletions": sum(t is None for t in hypothesis),
              "api_flagged_utterance": bool(production_flag.any()),
              "phone_score_mean": float(np.mean([x for x in scores if x is not None]))
                                   if any(x is not None for x in scores) else None}
    return result


def digest(path):
    return hashlib.sha256(_pl.Path(path).read_bytes()).hexdigest()


def evaluate(args):
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs, utt_features
    from mdd.jamo_to_kog2p import VOWEL
    from mdd.noise_preprocess import RNNoise16k, SileroEdges
    from mdd.session import SessionScorer, phone_context
    from mdd.speechsuper import SpeechSuperFormatter
    from mdd.vad import trim

    out = _pl.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "utterances.jsonl"
    if result_path.exists():
        raise RuntimeError("Refusing to overwrite existing experiment output")
    root, rows = read_rows(args.manifest, args.split)
    def labels(ext):
        return (_pl.Path(args.manifest) / (args.split + "." + ext)).read_text(encoding="utf-8").splitlines()
    tags, canons = labels("ltr"), labels("canon")
    if not len(rows) == len(tags) == len(canons):
        raise RuntimeError("Manifest/label length mismatch")
    selection, selection_stats = select_samples(args, rows, root, tags, canons)
    print("[selection] " + json.dumps(selection_stats, ensure_ascii=False), flush=True)
    noise_root, noise_rows = read_rows(args.noise_manifest, "valid")
    noise_pool = [noise_root / row[0] for row in noise_rows]
    dec = HFDecoder(args.ckpt)
    silero = SileroEdges(args.silero_model)
    rnnoise = RNNoise16k(args.rnnoise_library, delay_frames=args.rnnoise_delay_frames)
    formatter = SpeechSuperFormatter(args.norms, calib_path=args.calib)
    if formatter.cal is None:
        raise RuntimeError("Production phone calibration was not loaded")
    scorer = SessionScorer(args.scorer, args.norms)
    vowels = set(VOWEL)
    meta = {"args": vars(args), "selection": selection_stats,
            "n_utts": len(selection), "cohort_balanced": args.normal == args.errors,
            "methods": METHODS, "conditions": CONDITIONS,
            "source_row_indices": [i for i, _ in selection],
            "label_reference": "human TAG actual phones (.ltr); canonical positions (.canon)",
            "silero_sha256": digest(args.silero_model),
            "rnnoise_library_sha256": digest(args.rnnoise_library),
            "calibration_sha256": digest(args.calib),
            "scope": "GPU fp16 service checkpoint, production phone-flag rule; synthetic noise; no HTTP uploads",
            "scoring_policy": "diagnose every paired input, including prospective gate rejects; report rejection separately",
            "snr_definition": "whole clean utterance RMS; same definition as earlier experiment",
            "source_hashes": {str(p.relative_to(_root)): digest(p) for p in [
                _pl.Path(__file__).resolve(), _root / "tasks/pronunciation-assessment/src/mdd/noise_preprocess.py",
                _root / "tasks/pronunciation-assessment/src/mdd/speechsuper.py",
                _root / "tasks/pronunciation-assessment/src/mdd/vad.py"]}}
    (out / "experiment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # Warm each backend; first-call library loading isn't per-request latency.
    warm = np.zeros(2 * SR, np.float32)
    silero(warm)
    rnnoise(warm)
    forward_lprobs(dec, [warm])
    started = time.monotonic()
    with result_path.open("w", encoding="utf-8") as f:
        for k, (i, cohort) in enumerate(selection):
            wav = read_wav(root / rows[i][0])
            canon, tag = canons[i].split(), tags[i].split()
            y = [dec.d.index(t) for t in canon]
            if dec.d.unk() in y:
                raise RuntimeError("Selected canonical phone is outside model vocabulary")
            noise_babble = make_babble(random.Random(args.seed * 1000003 + i), noise_pool, len(wav))
            noise_rumble = make_rumble(args.seed * 7919 + i, len(wav))
            for condition, kind, snr, browser_ns in CONDITIONS:
                noisy = wav if kind is None else mix(wav, noise_babble if kind == "babble" else noise_rumble, snr)
                if browser_ns:
                    noisy = webrtc(noisy)
                original_check = audio_check(noisy)
                for method in METHODS:
                    t0 = time.perf_counter()
                    preprocessing = {}
                    if method == "baseline":
                        transformed = noisy
                        processed, trim_info = trim(transformed, SR)
                    elif method == "silero":
                        transformed = noisy
                        processed, preprocessing = silero(noisy)
                        trim_info = {"cut_sec": (len(noisy) - len(processed)) / SR}
                    else:
                        transformed, preprocessing = rnnoise(noisy)
                        processed, trim_info = trim(transformed, SR)
                    transformed_check = audio_check(transformed)
                    prep_ms = (time.perf_counter() - t0) * 1000
                    t1 = time.perf_counter()
                    lp, lengths = forward_lprobs(dec, [processed])
                    probabilities = lp[0, :int(lengths[0])].numpy()
                    model_ms = (time.perf_counter() - t1) * 1000
                    features, extra = utt_features(dec, probabilities, y, canon, len(processed) / SR)
                    predicted = [t for t in extra["pred"] if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
                    record = phone_diagnostics(canon, tag, predicted, extra["gops_m"],
                                               formatter, vowels, phone_context)
                    human_scores = scorer.utt_scores(features)
                    record.update({"k": k, "cohort": cohort, "condition": condition,
                                   "method": method, "original_sec": len(wav) / SR,
                                   "processed_sec": len(processed) / SR,
                                   "original_check_pass": original_check["pass"],
                                   "post_denoise_check_pass": transformed_check["pass"],
                                   "silero_check_pass": bool(preprocessing.get("speech_sec", 0) >= 1.0
                                                            and preprocessing.get("speech_ratio", 0) >= 0.3)
                                                        if method == "silero" else None,
                                   "silero_no_speech": preprocessing.get("no_speech", False),
                                   "prep_ms": prep_ms, "model_ms": model_ms,
                                   "total_ms": (time.perf_counter() - t0) * 1000,
                                   "cut_sec": trim_info.get("cut_sec", 0),
                                   "raw_human_scores": human_scores,
                                   "preprocessing": preprocessing})
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k + 1 == len(selection):
                elapsed = time.monotonic() - started
                remaining = elapsed / (k + 1) * (len(selection) - k - 1)
                print("[progress] %d/%d elapsed=%.1fs estimated_remaining=%.1fs" %
                      (k + 1, len(selection), elapsed, remaining), flush=True)
    print("[complete] %d utterances, %d records, %.1fs" %
          (len(selection), len(selection) * len(CONDITIONS) * len(METHODS), time.monotonic() - started), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--noise-manifest", default=str(MDD_OUTPUTS / "manifest_kspon_stage1"))
    ap.add_argument("--ckpt", default=str(DATA_ROOT / "checkpoints/w2vbert_stage2/step_38000"))
    ap.add_argument("--scorer", default=str(MDD_OUTPUTS / "score_v2_w2vbert/score_models_v2_w2vbert.joblib"))
    ap.add_argument("--norms", default=str(MDD_OUTPUTS / "score_phoneme/norms.json"))
    ap.add_argument("--calib", default=str(MDD_OUTPUTS / "score_phoneme/calibration.json"))
    ap.add_argument("--normal", type=int, default=250)
    ap.add_argument("--errors", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--silero-model", required=True)
    ap.add_argument("--rnnoise-library", required=True)
    ap.add_argument("--rnnoise-delay-frames", type=int, default=2)
    ap.add_argument("--out", required=True)
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()

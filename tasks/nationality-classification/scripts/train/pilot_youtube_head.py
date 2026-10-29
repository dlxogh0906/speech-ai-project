"""Small YouTube pilot on the deployed encoder, with an original-data control.

The encoder and normalization are frozen. The validation features use static
12s padding, per-part silence normalization and mean-log aggregation. Recorded
NPU Korean-gate probabilities can be reused because the gate, encoder, input,
and grouping stay fixed. Top-1 after this gate is Korea if gate_p > .9, otherwise
the main head's argmax. This reference must agree with recorded NPU top-1 before
results can be presented as a comparison with that baseline.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import subprocess

import numpy as np
from shared.paths import NAT_OUTPUTS


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_audio(path):
    import soundfile as sf
    try:
        wave, sr = sf.read(path, dtype="float32")
        if wave.ndim == 1 and sr == 16000:
            return wave
    except (RuntimeError, ValueError):
        pass
    import os
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    decoded = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                              "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=env)
    return np.frombuffer(decoded.stdout, np.float32).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-train", type=_pl.Path, required=True)
    parser.add_argument("--youtube-train", type=_pl.Path, required=True)
    parser.add_argument("--validation-responses", type=_pl.Path, required=True)
    parser.add_argument("--main-deploy", type=_pl.Path, default=NAT_OUTPUTS / "webmic_kd/run_zeroth/deploy")
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--replay-per-country", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8, help="Encoder feature extraction batch size")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fractions", default="0.05,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-features", type=_pl.Path,
                        help="Prior pilot directory with matching replay and validation conditions")
    parser.add_argument("--kd-policy", choices=("all", "correct-only"), default="all",
                        help="Keep teacher predictions only where its top-1 matches the training label")
    parser.add_argument("--tag-prefix", default="")
    args = parser.parse_args()
    fractions = [float(x) for x in args.fractions.split(",")]
    if any(not 0 < x < 1 for x in fractions):
        parser.error("YouTube fractions must be between zero and one")
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    from country_classifier.npu_encoder import load_pure, conv_out_len
    from country_classifier.wav_augment import silence_normalize
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.main_deploy / "head_np.npz", allow_pickle=True) as h:
        head_values = {key: h[key].copy() for key in h.files}
    labels = [str(x) for x in head_values["labels"]]
    c2i = {label: index for index, label in enumerate(labels)}
    korea = c2i["Korea"]
    youtube = read_rows(args.youtube_train)
    if not youtube or any(row["label_en"] not in c2i for row in youtube):
        raise ValueError("YouTube labels must belong to the deployed head")
    all_original = read_rows(args.original_train)
    candidates = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in all_original:
        if (row["label_en"] in c2i and row["task"] != "RW" and float(row["duration_sec"] or 0) >= 3
                and _pl.Path(row["audio_path"]).is_file()):
            candidates[row["label_en"]][row["speaker"]].append(row)
    rng = np.random.default_rng(args.seed)
    replay = []
    for label in labels:
        by_speaker = candidates[label]
        if not by_speaker:
            raise ValueError("No replay speech for " + label)
        queues = [list(rows) for _, rows in sorted(by_speaker.items())]
        for queue in queues:
            rng.shuffle(queue)
        rng.shuffle(queues)
        added = 0
        while added < args.replay_per_country and any(queues):
            for queue in queues:
                if queue and added < args.replay_per_country:
                    replay.append(queue.pop())
                    added += 1
    del all_original, candidates
    records_by_id = {}
    for line in args.validation_responses.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records_by_id[record["job_id"]] = record
    records = sorted(records_by_id.values(), key=lambda r: r["job_id"])
    if any(r["status"] != 200 or len(r["source_paths"]) != 3 for r in records):
        raise ValueError("Expected successful three-sentence validation records")
    if any(r["source_indices"] and len(r["source_indices"]) != 3 for r in records):
        raise ValueError("Invalid validation grouping")
    replay_speakers = {r["speaker"] for r in replay}
    youtube_speakers = {r["speaker"] for r in youtube}
    valid_speakers = {r["speaker"] for r in records}
    if (replay_speakers | youtube_speakers) & valid_speakers:
        raise ValueError("Training and validation speakers overlap")
    fingerprint = {"encoder": sha(args.main_deploy / "encoder_pure_l17.pt"),
                   "head": sha(args.main_deploy / "head_np.npz"),
                   "youtube": sha(args.youtube_train), "validation": sha(args.validation_responses),
                   "seed": args.seed, "replay_per_country": args.replay_per_country,
                   "preprocessing": "static12s_silence_norm_per_part_rms0.0015_minvoiced1_max60s_v1"}
    fingerprint["replay_selection"] = hashlib.sha256(json.dumps(replay, sort_keys=True).encode()).hexdigest()
    meta_path = args.out / "features_meta.json"
    cache_path = args.out / "features.npz"
    if cache_path.exists():
        if json.loads(meta_path.read_text()) != fingerprint:
            raise ValueError("Feature cache conditions differ; use a new output directory")
        with np.load(cache_path) as cache:
            XR, XY, XV = cache["replay"], cache["youtube"], cache["validation"]
    else:
        reusable = None
        if args.reuse_features:
            previous = json.loads((args.reuse_features / "features_meta.json").read_text())
            for key in fingerprint:
                if key != "youtube" and fingerprint[key] != previous.get(key):
                    raise ValueError("Reusable feature conditions differ: " + key)
            reusable = np.load(args.reuse_features / "features.npz")
            XR, XV = reusable["replay"], reusable["validation"]
        encoder = load_pure(args.main_deploy / "encoder_pure_l17.pt").to(device)
        mu = head_values["mu"].reshape(-1)
        sd = head_values["sd"].reshape(-1)

        @torch.no_grad()
        def encode(waves):
            result = []
            for offset in range(0, len(waves), args.batch_size):
                batch = waves[offset:offset + args.batch_size]
                src = np.zeros((len(batch), 192000), np.float32)
                valid = np.zeros((len(batch), conv_out_len(192000)), np.float32)
                for index, wave in enumerate(batch):
                    n = min(len(wave), 192000)
                    src[index, :n] = wave[:n]
                    valid[index, :conv_out_len(n)] = 1
                pooled = encoder(torch.from_numpy(src).to(device), torch.from_numpy(valid).to(device)).cpu().numpy()
                pooled = np.clip(np.nan_to_num(pooled), -50, 50)
                result.append((pooled - mu) / sd)
            return np.concatenate(result)

        def training_features(rows, name):
            values = []
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                waves = [silence_normalize(read_audio(row["audio_path"])[:128000]) for row in batch]
                values.append(encode(waves))
                if offset % (args.batch_size * 25) == 0:
                    print("[features {}] {}/{}".format(name, offset, len(rows)), flush=True)
            return np.concatenate(values)

        if reusable is None:
            XR = training_features(replay, "replay")
        XY = training_features(youtube, "youtube")
        values = []
        actual_parts = []
        for offset in range(0, len(records) if reusable is None else 0, args.batch_size):
            flat, counts = [], []
            for record in records[offset:offset + args.batch_size]:
                wave = np.concatenate([read_audio(path) for path in record["source_paths"]])
                buffer = io.BytesIO()
                sf.write(buffer, wave, 16000, format="WAV", subtype="PCM_16")
                buffer.seek(0)
                wave = sf.read(buffer, dtype="float32")[0][:960000]
                sec = len(wave) / 16000
                n = 3 if sec >= 9 else max(1, int(sec // 3))
                n = max(n, int(math.ceil(sec / 12)))
                cuts = [round(j * len(wave) / n) for j in range(n + 1)]
                chunks = []
                for a, b in zip(cuts[:-1], cuts[1:]):
                    chunk = wave[a:b]
                    frame_count = len(chunk) // 480
                    rms = np.sqrt(np.square(chunk[:frame_count * 480].reshape(frame_count, 480), dtype=np.float64).mean(1))
                    if len(chunk) >= 8000 and float((rms >= .0015).sum() * .03) >= 1:
                        chunks.append(silence_normalize(chunk))
                if len(chunks) != record["payload"]["n"]:
                    raise ValueError("Split does not match NPU for " + record["job_id"])
                counts.append(len(chunks))
                actual_parts.append(len(chunks))
                flat.extend(chunks)
            features = encode(flat)
            start = 0
            for count in counts:
                values.append(features[start:start + count].mean(0))
                start += count
            if offset % (args.batch_size * 10) == 0:
                print("[features validation] {}/{}".format(offset, len(records)), flush=True)
        if reusable is None:
            XV = np.stack(values)
        else:
            reusable.close()
        np.savez(cache_path, replay=XR, youtube=XY, validation=XV)
        dump_json(meta_path, fingerprint)
        del encoder
        torch.cuda.empty_cache()
    yr = np.array([c2i[r["label_en"]] for r in replay], np.int64)
    yy = np.array([c2i[r["label_en"]] for r in youtube], np.int64)
    yv = np.array([c2i[r["country"]] for r in records], np.int64)
    gate = np.array([float(r["payload"]["gate_korea"] or 0) > .9 for r in records])
    countries = np.array([r["country"] for r in records])
    W0, b0 = head_values["W"], head_values["b"]

    def evaluate(W, b):
        logits = XV @ W.T + b
        prediction = logits.argmax(1)
        prediction[gate] = korea
        correct = prediction == yv
        foreign = yv != korea
        per_country = {}
        for label in sorted(set(countries)):
            mask = countries == label
            per_country[label] = {"n": int(mask.sum()), "top1": float(correct[mask].mean()),
                                  "errors": dict(collections.Counter(np.array(labels)[prediction[mask & ~correct]].tolist()))}
        return {"foreign_top1": float(correct[foreign].mean()),
                "foreign_macro_top1": float(np.mean([r["top1"] for c, r in per_country.items() if c != "Korea"])),
                "korean_top1": float(correct[~foreign].mean()), "per_country": per_country}, prediction

    baseline, prediction = evaluate(W0, b0)
    npu_prediction = np.array([c2i[r["payload"]["top5"][0]["label"]] for r in records])
    differences = np.nonzero(prediction != npu_prediction)[0]
    parity = {"groups": len(records), "matching_top1": int((prediction == npu_prediction).sum()),
              "differences": [{"job_id": records[i]["job_id"], "reference": labels[prediction[i]],
                               "npu": labels[npu_prediction[i]]} for i in differences]}
    dump_json(args.out / "reference_baseline.json", {"metrics": baseline, "npu_parity": parity})
    print("[baseline] " + json.dumps({k: v for k, v in baseline.items() if k != "per_country"}), flush=True)
    print("[parity] " + json.dumps(parity), flush=True)
    if len(differences) > max(2, int(len(records) * .01)):
        raise ValueError("Reference disagrees with NPU; resolve inference parity before training")
    XT = torch.from_numpy(XR).to(device)
    YT = torch.from_numpy(yr).to(device)
    XYT = torch.from_numpy(XY).to(device)
    YYT = torch.from_numpy(yy).to(device)
    teacher = torch.from_numpy(XR @ W0.T + b0).to(device)
    runs = []
    for fraction in fractions:
        for use_youtube in (False, True):
            tag = args.tag_prefix + ("youtube" if use_youtube else "replay_control") + "_{:.2f}".format(fraction)
            torch.manual_seed(args.seed)
            head = torch.nn.Linear(XR.shape[1], len(labels)).to(device)
            with torch.no_grad():
                head.weight.copy_(torch.from_numpy(W0).to(device))
                head.bias.copy_(torch.from_numpy(b0).to(device))
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
            rng = np.random.default_rng(args.seed)
            best, best_state, history = None, None, []
            for step in range(1, args.steps + 1):
                idx = torch.from_numpy(rng.integers(0, len(XR), size=256)).to(device)
                logits = head(XT[idx])
                if args.kd_policy == "correct-only":
                    kd = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(teacher[idx] / 2, 1),
                                  reduction="none").sum(1) * 4
                    weight = .7 * (teacher[idx].argmax(1) == YT[idx]).float()
                    ce = F.cross_entropy(logits, YT[idx], reduction="none")
                    loss = (1 - fraction) * ((1 - weight) * ce + weight * kd).mean()
                else:
                    kd = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(teacher[idx] / 2, 1), reduction="batchmean") * 4
                    loss = (1 - fraction) * (.3 * F.cross_entropy(logits, YT[idx]) + .7 * kd)
                yi = torch.from_numpy(rng.integers(0, len(XY), size=64)).to(device)
                if use_youtube:
                    loss = loss + fraction * F.cross_entropy(head(XYT[yi]), YYT[yi])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 25 == 0 or step == args.steps:
                    W, b = head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy()
                    metrics, _ = evaluate(W, b)
                    history.append({"step": step, **metrics})
                    score = metrics["foreign_top1"] + metrics["foreign_macro_top1"]
                    if best is None or score > best["score"]:
                        best = {"step": step, "score": score, **metrics}
                        best_state = (W.copy(), b.copy())
                    print("[{} step {}] micro {:.4f} macro {:.4f} Korea {:.4f}".format(
                        tag, step, metrics["foreign_top1"], metrics["foreign_macro_top1"], metrics["korean_top1"]), flush=True)
            run_dir = args.out / tag
            run_dir.mkdir(exist_ok=True)
            output_head = dict(head_values)
            output_head["W"], output_head["b"] = best_state
            np.savez(run_dir / "head_np.npz", **output_head)
            dump_json(run_dir / "history.json", history)
            dump_json(run_dir / "result.json", {"best_validation": best, "training_fraction": fraction,
                                               "uses_youtube": use_youtube, "kd_policy": args.kd_policy,
                                               "encoder_sha256": fingerprint["encoder"]})
            runs.append({"tag": tag, "uses_youtube": use_youtube, "fraction": fraction, "best_validation": best})
    summary = {"baseline": baseline, "npu_parity": parity, "runs": runs,
               "replay_clips": len(replay), "youtube_clips": len(youtube),
               "youtube_speakers": len(youtube_speakers), "replay_speakers": len(replay_speakers),
               "validation_groups": len(records), "encoder_frozen": True,
               "note": "Validation chooses the checkpoint; results are a small feasibility pilot, not independent test results."}
    dump_json(args.out / "summary.json", summary)
    with (args.out / "replay.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replay[0]))
        writer.writeheader()
        writer.writerows(replay)
    print("[done] " + str(args.out / "summary.json"), flush=True)


if __name__ == "__main__":
    main()

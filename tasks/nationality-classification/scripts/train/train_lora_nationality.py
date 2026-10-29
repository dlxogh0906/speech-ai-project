"""Controlled LoRA adaptation of the deployed nationality encoder on A100.

Compare original-data replay, YouTube supervision, and YouTube plus waveform
consistency. Frozen prefix caches are fingerprinted; validation never supplies
training gradients. Merged encoder/head pairs require separate physical NPU
verification before any deployment.
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
import hashlib
import json
import time

import numpy as np

from shared.paths import NAT_OUTPUTS
from pilot_youtube_head import read_audio, read_rows, sha, dump_json


def augment_wave(wave, index, seed):
    """Fixed training views: mild noise, phone bandwidth, or mu-law quantization."""
    from scipy.signal import resample_poly
    rng = np.random.default_rng(seed + index * 1009)
    result = wave.copy()
    mode = index % 3
    if mode == 1:
        result = resample_poly(resample_poly(result, 1, 2), 2, 1)[:len(wave)]
    elif mode == 2:
        peak = max(float(np.abs(result).max()), 1e-5)
        unit = np.clip(result / peak, -1, 1)
        encoded = np.sign(unit) * np.log1p(255 * np.abs(unit)) / np.log(256)
        encoded = np.round((encoded + 1) * 127.5) / 127.5 - 1
        result = np.sign(encoded) * np.expm1(np.abs(encoded) * np.log(256)) / 255 * peak
    if mode in (0, 2):
        frame_n = len(wave) // 480
        frame_rms = np.sqrt(np.square(wave[:frame_n * 480].reshape(frame_n, 480)).mean(1))
        voiced = wave[:frame_n * 480].reshape(frame_n, 480)[frame_rms >= .0015]
        rms = float(np.sqrt(np.square(voiced).mean())) if voiced.size else .01
        snr = float(rng.uniform(20, 30)) if mode == 0 else 30.0
        noise = rng.standard_normal(len(result)).astype(np.float32)
        noise *= rms / (10 ** (snr / 20))
        result = result + noise
    return np.clip(result, -1, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=_pl.Path, required=True)
    parser.add_argument("--youtube-train", type=_pl.Path, required=True)
    parser.add_argument("--validation-responses", type=_pl.Path, required=True)
    parser.add_argument("--reuse-prefix-cache", type=_pl.Path, required=True)
    parser.add_argument("--main-deploy", type=_pl.Path, default=NAT_OUTPUTS / "webmic_kd/run_zeroth/deploy")
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--from-layer", type=int, default=16)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--youtube-fraction", type=float, default=.1)
    parser.add_argument("--consistency-weight", type=float, default=.1)
    parser.add_argument("--augmentation-per-country", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", default="replay,lora_youtube,lora_youtube_aux")
    args = parser.parse_args()
    if not 0 < args.from_layer < 18 or args.rank < 1 or not 0 < args.youtube_fraction < 1:
        parser.error("Invalid layer, rank or YouTube fraction")
    arms = args.arms.split(",")
    if set(arms) - {"replay", "lora_youtube", "lora_youtube_aux"}:
        parser.error("Unknown experiment arm")
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.nn import functional as F
    from country_classifier.npu_encoder import load_pure, conv_out_len
    from country_classifier.wav_augment import silence_normalize
    from country_classifier.encoder_adaptation import (
        attach_lora, frozen_prefix, suffix_features, merged_encoder_state,
        probability_consistency,
    )
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the requested CUDA GPU")
    device = torch.device("cuda")
    with np.load(args.main_deploy / "head_np.npz", allow_pickle=True) as values:
        h = {key: values[key].copy() for key in values.files}
    labels = list(map(str, h["labels"]))
    c2i = {label: index for index, label in enumerate(labels)}
    korea = c2i["Korea"]
    replay, youtube = read_rows(args.replay), read_rows(args.youtube_train)
    records_by_id = {r["job_id"]: r for r in map(json.loads, args.validation_responses.read_text().splitlines())}
    records = [records_by_id[key] for key in sorted(records_by_id)]
    if any(r["status"] != 200 for r in records):
        raise ValueError("Incomplete original validation")
    if {r["speaker"] for r in replay + youtube} & {r["speaker"] for r in records}:
        raise ValueError("Training/validation speaker overlap")
    if any(row["label_en"] not in c2i for row in replay + youtube):
        raise ValueError("Unknown training label")
    encoder = load_pure(args.main_deploy / "encoder_pure_l17.pt").to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    mu, sd = [torch.from_numpy(h[key].reshape(-1)).to(device) for key in ("mu", "sd")]
    W0, b0 = [torch.from_numpy(h[key]).to(device) for key in ("W", "b")]
    frames = conv_out_len(192000)
    source = {"encoder": sha(args.main_deploy / "encoder_pure_l17.pt"),
              "head": sha(args.main_deploy / "head_np.npz"),
              "prefix_layers": args.from_layer,
              "preprocessing": "static12s_silence_norm_rms0.0015_minvoiced1_max60s_float32_v1"}

    def reused(name, row_hash):
        folder = args.reuse_prefix_cache / name
        meta = json.loads((folder / "meta.json").read_text())
        if meta["conditions"] != dict(source, rows=row_hash):
            raise ValueError("Cached prefix conditions differ: " + name)
        return {"x": np.load(folder / "hidden.npy", mmap_mode="r"),
                "mask": np.load(folder / "mask.npy", mmap_mode="r"),
                "teacher": np.load(folder / "teacher.npy"), "groups": meta["groups"]}

    R = reused("replay", sha(args.replay))
    V = reused("validation", sha(args.validation_responses))
    if len(R["x"]) != len(replay) or len(set(V["groups"])) != len(records):
        raise ValueError("Cached row counts differ")

    def cached(name, rows, condition, augmentation=False):
        folder = args.out / "prefix_cache" / name
        folder.mkdir(parents=True, exist_ok=True)
        meta_path = folder / "meta.json"
        conditions = dict(source, rows=condition)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["conditions"] != conditions:
                raise ValueError("Experiment prefix conditions differ")
            return {"x": np.load(folder / "hidden.npy", mmap_mode="r"),
                    "mask": np.load(folder / "mask.npy", mmap_mode="r")}
        n = len(rows)
        hidden = np.lib.format.open_memmap(folder / "hidden.npy", mode="w+", dtype=np.float32,
                                          shape=(n, frames, 1024))
        masks = np.lib.format.open_memmap(folder / "mask.npy", mode="w+", dtype=np.float32, shape=(n, frames))
        started = time.monotonic()
        for offset in range(0, n, args.extract_batch_size):
            chunk = rows[offset:offset + args.extract_batch_size]
            src = np.zeros((len(chunk), 192000), np.float32)
            valid = np.zeros((len(chunk), frames), np.float32)
            for index, row in enumerate(chunk):
                wave = read_audio(row["audio_path"])[:128000]
                if augmentation:
                    wave = augment_wave(wave, offset + index, args.seed)
                wave = silence_normalize(wave)
                src[index, :len(wave)] = wave
                valid[index, :conv_out_len(len(wave))] = 1
            tx = frozen_prefix(encoder, torch.from_numpy(src).to(device),
                               torch.from_numpy(valid).to(device), args.from_layer)
            hidden[offset:offset + len(chunk)] = tx.cpu().numpy()
            masks[offset:offset + len(chunk)] = valid
            if offset % 100 == 0:
                print("[cache {}] {}/{} {:.1f}s".format(name, offset, n, time.monotonic() - started), flush=True)
        hidden.flush()
        masks.flush()
        dump_json(meta_path, {"conditions": conditions, "n": n, "dtype": "float32"})
        return {"x": hidden, "mask": masks}

    Y = cached("youtube", youtube, sha(args.youtube_train))
    aug_rng = np.random.default_rng(args.seed + 17)
    per_country = collections.defaultdict(list)
    for index, row in enumerate(replay):
        per_country[row["label_en"]].append(index)
    aug_indices = np.array(sorted(index for country in sorted(per_country)
                                  for index in aug_rng.choice(per_country[country],
                                  min(args.augmentation_per_country, len(per_country[country])), replace=False)))
    aug_conditions = {"seed": args.seed, "version": "mild_noise_phone_mulaw_v1",
                      "replay_indices": aug_indices.tolist(), "youtube": sha(args.youtube_train)}
    condition_hash = hashlib.sha256(json.dumps(aug_conditions, sort_keys=True).encode()).hexdigest()
    A = cached("augmented", [replay[i] for i in aug_indices] + youtube, condition_hash, augmentation=True)
    dump_json(args.out / "augmentation.json", aug_conditions)
    yr = torch.tensor([c2i[row["label_en"]] for row in replay], device=device)
    yy = torch.tensor([c2i[row["label_en"]] for row in youtube], device=device)
    yv = np.array([c2i[row["country"]] for row in records])
    gate = np.array([float(row["payload"]["gate_korea"] or 0) > .9 for row in records])
    groups = np.array(V["groups"])
    counts = np.bincount(groups)
    foreign = yv != korea

    def batch(data, indices):
        return (torch.from_numpy(np.array(data["x"][indices])).to(device),
                torch.from_numpy(np.array(data["mask"][indices])).to(device))

    def logits_for(head, data, indices):
        tx, valid = batch(data, indices)
        return head(suffix_features(encoder, tx, valid, mu, sd, args.from_layer))

    def aggregate(logits):
        # Match service mean-log probabilities, including its 1e-12 clipping.
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        log_p = np.log(np.maximum(p, 1e-12))
        result = np.zeros((len(records), len(labels)), np.float64)
        np.add.at(result, groups, log_p)
        return result / counts[:, None]

    def metrics(group_logits):
        prediction = group_logits.argmax(1)
        prediction[gate] = korea
        hit = prediction == yv
        per = {}
        for country in sorted(set(row["country"] for row in records)):
            mask = yv == c2i[country]
            per[country] = {"n": int(mask.sum()), "speakers": len({row["speaker"] for row in records if row["country"] == country}),
                            "correct_top1": int(hit[mask].sum()), "top1": float(hit[mask].mean()),
                            "errors": dict(collections.Counter(np.array(labels)[prediction[mask & ~hit]].tolist()))}
        return {"foreign_n": int(foreign.sum()), "foreign_top1": float(hit[foreign].mean()),
                "foreign_macro_top1": float(np.mean([m["top1"] for country, m in per.items() if country != "Korea"])),
                "korean_top1": float(hit[~foreign].mean()), "per_country": per}, prediction

    @torch.no_grad()
    def evaluate(head):
        parts = []
        for offset in range(0, len(V["x"]), args.batch_size):
            indices = np.arange(offset, min(offset + args.batch_size, len(V["x"])))
            parts.append(logits_for(head, V, indices).cpu().numpy())
        part_logits = np.concatenate(parts)
        score, prediction = metrics(aggregate(part_logits))
        return score, part_logits, prediction

    baseline_metrics, baseline_pred = metrics(aggregate(V["teacher"]))
    recorded_pred = np.array([c2i[row["payload"]["top5"][0]["label"]] for row in records])
    mismatches = int((baseline_pred != recorded_pred).sum())
    dump_json(args.out / "reference_baseline.json", {"metrics": baseline_metrics, "n": len(records),
              "matching_npu_top1": int((baseline_pred == recorded_pred).sum()),
              "note": "GPU reference; physical NPU results remain authoritative"})
    if mismatches > max(2, int(len(records) * .01)):
        raise ValueError("Resolve baseline GPU/NPU disagreement")
    paths = attach_lora(encoder, args.from_layer, args.rank, args.alpha)
    adapter_params = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    initial_adapters = {key: value.detach().clone() for key, value in encoder.state_dict().items()
                        if key.endswith(".A") or key.endswith(".B")}
    speaker_counts = collections.Counter(row["speaker"] for row in youtube)
    weights = np.array([1.0 / speaker_counts[row["speaker"]] for row in youtube])
    weights /= weights.sum()
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, _pl.Path) else value for key, value in config.items()}
    config.update({"source": source, "gpu": torch.cuda.get_device_name(device),
                   "trainable_adapter_parameters": sum(parameter.numel() for parameter in adapter_params),
                   "trainable_head_parameters": int(W0.numel() + b0.numel()),
                   "replay_clips": len(replay), "replay_speakers": len({row["speaker"] for row in replay}),
                   "youtube_clips": len(youtube), "youtube_speakers": len(speaker_counts),
                   "kd_policy": "correct-only", "youtube_sampling": "equal probability per speaker",
                   "selection": "validation micro + country mean; prefer candidates preserving baseline micro/Korea/Uzbek/Philippines",
                   "evaluation_status": "Repeated original validation, not a new blind test",
                   "production_replacement": False})
    dump_json(args.out / "experiment_config.json", config)
    print("[training] GPU {} adapters {} head {}".format(config["gpu"], config["trainable_adapter_parameters"],
                                                       config["trainable_head_parameters"]), flush=True)
    runs = []
    for arm in arms:
        with torch.no_grad():
            state = encoder.state_dict()
            for key, value in initial_adapters.items():
                state[key].copy_(value)
        head = torch.nn.Linear(1024, len(labels)).to(device)
        with torch.no_grad():
            head.weight.copy_(W0)
            head.bias.copy_(b0)
        optimizer = torch.optim.AdamW([{"params": adapter_params, "lr": args.adapter_lr},
                                        {"params": head.parameters(), "lr": args.head_lr}], weight_decay=1e-4)
        replay_rng = np.random.default_rng(args.seed)
        youtube_rng = np.random.default_rng(args.seed + 1)
        aux_rng = np.random.default_rng(args.seed + 2)
        best_any, best_eligible, best_state = None, None, None
        history = []
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            indices = replay_rng.integers(0, len(replay), args.batch_size)
            logits = logits_for(head, R, indices)
            target = torch.from_numpy(R["teacher"][indices]).to(device)
            ce = F.cross_entropy(logits, yr[indices], reduction="none")
            kd = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(target / 2, 1), reduction="none").sum(1) * 4
            correct = (target.argmax(1) == yr[indices]).float()
            replay_loss = ((1 - .7 * correct) * ce + .7 * correct * kd).mean()
            ((1 - args.youtube_fraction) * replay_loss).backward()
            yi = youtube_rng.choice(len(youtube), max(4, args.batch_size // 4), p=weights)
            youtube_loss = torch.tensor(0.0, device=device)
            consistency = torch.tensor(0.0, device=device)
            if arm != "replay":
                youtube_loss = F.cross_entropy(logits_for(head, Y, yi), yy[yi])
                (args.youtube_fraction * youtube_loss).backward()
            if arm == "lora_youtube_aux":
                ai = aux_rng.integers(0, len(aug_indices), 4)
                noisy = logits_for(head, A, ai)
                clean = logits_for(head, R, aug_indices[ai])
                ri_consistency = probability_consistency(clean, noisy)
                noisy_y = logits_for(head, A, len(aug_indices) + yi)
                clean_y = logits_for(head, Y, yi)
                consistency = (.5 * ri_consistency + .5 * probability_consistency(clean_y, noisy_y))
                (args.consistency_weight * consistency).backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter_params + list(head.parameters()), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite training gradient")
            optimizer.step()
            if step % 25 == 0:
                print("[{}] {}/{} {:.1f}s replay {:.4f} youtube {:.4f} consistency {:.4f}".format(
                      arm, step, args.steps, time.monotonic() - started, float(replay_loss.detach()),
                      float(youtube_loss.detach()), float(consistency.detach())), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                score, part_logits, prediction = evaluate(head)
                eligible = (score["foreign_top1"] >= baseline_metrics["foreign_top1"]
                            and score["korean_top1"] >= baseline_metrics["korean_top1"]
                            and all(score["per_country"][country]["top1"] >= baseline_metrics["per_country"][country]["top1"]
                                    for country in ("Uzbekistan", "Philippines")))
                key = score["foreign_top1"] + score["foreign_macro_top1"]
                item = {"step": step, "score": key, "eligible": eligible, **score}
                history.append(item)
                choose = False
                if best_any is None or key > best_any["score"]:
                    best_any = item
                    choose = best_eligible is None
                if eligible and (best_eligible is None or key > best_eligible["score"]):
                    best_eligible = item
                    choose = True
                if choose:
                    best_state = ({key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
                                   if key.endswith(".A") or key.endswith(".B")},
                                  head.weight.detach().cpu().numpy().copy(), head.bias.detach().cpu().numpy().copy(),
                                  part_logits.copy(), prediction.copy())
                print("[{} eval {}] micro {:.4f} country mean {:.4f} eligible {}".format(
                      arm, step, score["foreign_top1"], score["foreign_macro_top1"], eligible), flush=True)
                dump_json(args.out / (arm + "_history.json"), history)
        selected = best_eligible or best_any
        with torch.no_grad():
            state = encoder.state_dict()
            for key, value in best_state[0].items():
                state[key].copy_(value.to(device))
            head.weight.copy_(torch.from_numpy(best_state[1]).to(device))
            head.bias.copy_(torch.from_numpy(best_state[2]).to(device))
        folder = args.out / arm
        deploy = folder / "deploy"
        deploy.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "qv_lora_v1", "rank": args.rank, "alpha": args.alpha,
                    "from_layer": args.from_layer, "source": source,
                    "adapters": best_state[0]}, folder / "adapters.pt")
        merged = merged_encoder_state(encoder, paths)
        torch.save(merged, deploy / "encoder_pure_l17.pt")
        np.savez(deploy / "head_np.npz", **dict(h, W=best_state[1], b=best_state[2]))
        np.savez(folder / "validation_logits.npz", part_logits=best_state[3], prediction=best_state[4],
                 job_ids=np.array([row["job_id"] for row in records]), labels=np.array(labels))
        info = {"arm": arm, "selected_validation": selected, "best_unconstrained": best_any,
                "eligible_at_selection": best_eligible is not None, "config": config,
                "encoder_sha256": sha(deploy / "encoder_pure_l17.pt"),
                "head_sha256": sha(deploy / "head_np.npz"), "npu_verification": "pending"}
        dump_json(folder / "result.json", info)
        dump_json(deploy / "encoder_pair.json", info)
        runs.append(info)
        dump_json(args.out / "summary.json", {"baseline": baseline_metrics, "runs": runs,
                  "status": "GPU training; physical NPU verification pending", "production_replacement": False})
        print("[saved] {} step {}".format(arm, selected["step"]), flush=True)
    print("[done] " + str(args.out), flush=True)


if __name__ == "__main__":
    main()

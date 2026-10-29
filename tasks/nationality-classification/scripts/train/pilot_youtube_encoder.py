"""Controlled fine-tuning of the last deployed encoder layers.

Cache the frozen prefix in float32, preserving static 12s padding and masks.
Train only the suffix and matched linear head. Every arm starts at production
weights. The validation split never supplies training examples or gradients.
These GPU service-reference scores require a separate NPU compilation check.
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
import io
import json
import math
import time

import numpy as np
from shared.paths import NAT_OUTPUTS
from pilot_youtube_head import read_audio, read_rows, sha, dump_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=_pl.Path, required=True)
    ap.add_argument("--youtube-train", type=_pl.Path)
    ap.add_argument("--validation-responses", type=_pl.Path, required=True)
    ap.add_argument("--main-deploy", type=_pl.Path, default=NAT_OUTPUTS / "webmic_kd/run_zeroth/deploy")
    ap.add_argument("--out", type=_pl.Path, required=True)
    ap.add_argument("--unfreeze-from", type=int, default=16)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--extract-batch-size", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=1e-6)
    ap.add_argument("--head-lr", type=float, default=5e-5)
    ap.add_argument("--fractions", default="0.10")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.unfreeze_from < 18:
        ap.error("unfreeze-from must be 1..17")
    if not args.prepare_only and args.youtube_train is None:
        ap.error("youtube-train is required for training")
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn.functional as F
    import soundfile as sf
    from country_classifier.npu_encoder import load_pure, conv_out_len
    from country_classifier.wav_augment import silence_normalize
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.main_deploy / "head_np.npz", allow_pickle=True) as data:
        h = {key: data[key].copy() for key in data.files}
    labels = [str(x) for x in h["labels"]]
    c2i = {label: index for index, label in enumerate(labels)}
    korea = c2i["Korea"]
    replay = read_rows(args.replay)
    youtube = read_rows(args.youtube_train) if args.youtube_train else []
    by_id = {r["job_id"]: r for r in map(json.loads, args.validation_responses.read_text().splitlines())}
    records = [by_id[k] for k in sorted(by_id)]
    if any(r["status"] != 200 for r in records):
        raise ValueError("Validation contains failed requests")
    if {r["speaker"] for r in replay + youtube} & {r["speaker"] for r in records}:
        raise ValueError("Training/evaluation speaker overlap")
    encoder = load_pure(args.main_deploy / "encoder_pure_l17.pt").to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    suffix = torch.nn.ModuleList(list(encoder.encoder.layers)[args.unfreeze_from:])
    initial_suffix = {k: v.detach().clone() for k, v in suffix.state_dict().items()}
    mu = torch.from_numpy(h["mu"].reshape(-1)).to(device)
    sd = torch.from_numpy(h["sd"].reshape(-1)).to(device)
    W0, b0 = torch.from_numpy(h["W"]).to(device), torch.from_numpy(h["b"]).to(device)
    frames = conv_out_len(192000)

    def run_suffix(x, mask):
        bias = ((mask - 1) * 1e4)[:, None, None, :]
        for layer in suffix:
            x = layer(x, bias)
        m = mask.unsqueeze(-1)
        pooled = (x * m).sum(1) / m.sum(1).clamp_min(1)
        return (pooled.clamp(-50, 50) - mu) / sd

    @torch.no_grad()
    def prefix(source, mask):
        x = source.unsqueeze(1)
        for seq in encoder.feature_extractor.conv_layers:
            x = F.gelu(seq[2][1](seq[0](x).transpose(1, 2)).transpose(1, 2))
        x = encoder.post_extract_proj(encoder.layer_norm(x.transpose(1, 2)))
        x = x * mask.unsqueeze(-1)
        pc = encoder.encoder.pos_conv[0](x.transpose(1, 2))[:, :, :-1]
        x = x + F.gelu(pc).transpose(1, 2)
        bias = ((mask - 1) * 1e4)[:, None, None, :]
        for layer in list(encoder.encoder.layers)[:args.unfreeze_from]:
            x = layer(x, bias)
        return x

    fingerprint = {"encoder": sha(args.main_deploy / "encoder_pure_l17.pt"),
                   "head": sha(args.main_deploy / "head_np.npz"),
                   "prefix_layers": args.unfreeze_from,
                   "preprocessing": "static12s_silence_norm_rms0.0015_minvoiced1_max60s_float32_v1"}

    def training_wave(row):
        return [silence_normalize(read_audio(row["audio_path"])[:128000])]

    def validation_waves(record):
        wave = np.concatenate([read_audio(path) for path in record["source_paths"]])
        buf = io.BytesIO()
        sf.write(buf, wave, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        wave = sf.read(buf, dtype="float32")[0][:960000]
        sec = len(wave) / 16000
        requested = int(record["payload"].get("parts_requested", 3))
        n = requested if sec >= requested * 3 else max(1, int(sec // 3))
        n = max(n, int(math.ceil(sec / 12)))
        cuts = [round(j * len(wave) / n) for j in range(n + 1)]
        waves = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = wave[a:b]
            count = len(chunk) // 480
            if count == 0:
                continue
            rms = np.sqrt(np.square(chunk[:count * 480].reshape(count, 480), dtype=np.float64).mean(1))
            if len(chunk) >= 8000 and float((rms >= .0015).sum() * .03) >= 1:
                waves.append(silence_normalize(chunk))
        if len(waves) != record["payload"]["n"]:
            raise ValueError("Split differs: " + record["job_id"])
        return waves

    def cache(name, rows, row_hash, loader):
        folder = args.out / "prefix_cache" / name
        folder.mkdir(parents=True, exist_ok=True)
        meta_path = folder / "meta.json"
        condition = dict(fingerprint, rows=row_hash)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["conditions"] != condition:
                raise ValueError("Cache mismatch for " + name)
            return {"x": np.load(folder / "hidden.npy", mmap_mode="r"),
                    "mask": np.load(folder / "mask.npy", mmap_mode="r"),
                    "teacher": np.load(folder / "teacher.npy"), "groups": meta["groups"]}
        counts = ([r["payload"]["n"] for r in rows] if name == "validation" else [1] * len(rows))
        n = sum(counts)
        xcache = np.lib.format.open_memmap(folder / "hidden.npy", mode="w+", dtype=np.float32, shape=(n, frames, 1024))
        masks = np.lib.format.open_memmap(folder / "mask.npy", mode="w+", dtype=np.float32, shape=(n, frames))
        teacher = np.zeros((n, len(labels)), np.float32)
        groups = np.repeat(np.arange(len(rows)), counts).tolist()
        pending, pos = [], 0

        def flush():
            nonlocal pos
            src = np.zeros((len(pending), 192000), np.float32)
            valid = np.zeros((len(pending), frames), np.float32)
            for i, wave in enumerate(pending):
                ns = min(len(wave), 192000)
                src[i, :ns] = wave[:ns]
                valid[i, :conv_out_len(ns)] = 1
            ts = torch.from_numpy(src).to(device)
            tm = torch.from_numpy(valid).to(device)
            with torch.no_grad():
                tx = prefix(ts, tm)
                logits = F.linear(run_suffix(tx, tm), W0, b0)
            xcache[pos:pos + len(pending)] = tx.cpu().numpy()
            masks[pos:pos + len(pending)] = valid
            teacher[pos:pos + len(pending)] = logits.cpu().numpy()
            pos += len(pending)
            pending.clear()

        for index, row in enumerate(rows):
            pending.extend(loader(row))
            if len(pending) >= args.extract_batch_size:
                flush()
            if index % 100 == 0:
                print("[cache {}] {}/{}".format(name, index, len(rows)), flush=True)
        if pending:
            flush()
        assert pos == n
        xcache.flush()
        masks.flush()
        np.save(folder / "teacher.npy", teacher)
        dump_json(meta_path, {"conditions": condition, "groups": groups, "dtype": "float32", "n": n})
        return {"x": xcache, "mask": masks, "teacher": teacher, "groups": groups}

    R = cache("replay", replay, sha(args.replay), training_wave)
    V = cache("validation", records, sha(args.validation_responses), validation_waves)
    if args.prepare_only:
        print("[done] frozen prefix caches ready", flush=True)
        return
    Y = cache("youtube", youtube, sha(args.youtube_train), training_wave)
    yr = torch.tensor([c2i[r["label_en"]] for r in replay], device=device)
    yy = torch.tensor([c2i[r["label_en"]] for r in youtube], device=device)
    yv = np.array([c2i[r["country"]] for r in records])
    foreign = yv != korea
    gate = np.array([float(r["payload"]["gate_korea"] or 0) > .9 for r in records])
    groups = np.array(V["groups"])
    counts = np.bincount(groups)

    def batch(data, indices):
        return (torch.from_numpy(np.array(data["x"][indices])).to(device),
                torch.from_numpy(np.array(data["mask"][indices])).to(device))

    def metrics(logits):
        prediction = logits.argmax(1)
        prediction[gate] = korea
        hit = prediction == yv
        countries = {}
        for label in sorted(set(r["country"] for r in records)):
            mask = yv == c2i[label]
            countries[label] = {"n": int(mask.sum()), "top1": float(hit[mask].mean()),
                                "errors": dict(collections.Counter(np.array(labels)[prediction[mask & ~hit]].tolist()))}
        return {"foreign_top1": float(hit[foreign].mean()),
                "foreign_macro_top1": float(np.mean([x["top1"] for c, x in countries.items() if c != "Korea"])),
                "korean_top1": float(hit[~foreign].mean()), "per_country": countries}, prediction

    def aggregate(logits):
        result = np.zeros((len(records), len(labels)), np.float64)
        np.add.at(result, groups, logits)
        return result / counts[:, None]

    @torch.no_grad()
    def evaluate(head):
        part_logits = []
        for offset in range(0, len(V["x"]), args.batch_size):
            ix = np.arange(offset, min(offset + args.batch_size, len(V["x"])))
            tx, tm = batch(V, ix)
            part_logits.append(head(run_suffix(tx, tm)).cpu().numpy())
        logits = aggregate(np.concatenate(part_logits))
        return metrics(logits)[0], logits

    baseline_logits = aggregate(V["teacher"])
    baseline_metrics, pred = metrics(baseline_logits)
    npu = np.array([c2i[r["payload"]["top5"][0]["label"]] for r in records])
    dump_json(args.out / "reference_baseline.json", {"metrics": baseline_metrics,
              "matching_npu_top1": int((pred == npu).sum()), "n": len(records)})
    if int((pred != npu).sum()) > max(2, int(len(records) * .01)):
        raise ValueError("Resolve GPU/NPU reference disagreement before training")
    runs = []
    for fraction in [float(v) for v in args.fractions.split(",")]:
        if not 0 < fraction < 1:
            raise ValueError("Fractions must be 0..1")
        for use_youtube in (False, True):
            tag = ("encoder_youtube" if use_youtube else "encoder_control") + "_{:.2f}".format(fraction)
            suffix.load_state_dict(initial_suffix)
            for p in suffix.parameters():
                p.requires_grad_(True)
            head = torch.nn.Linear(1024, len(labels)).to(device)
            with torch.no_grad():
                head.weight.copy_(W0)
                head.bias.copy_(b0)
            opt = torch.optim.AdamW([{"params": suffix.parameters(), "lr": args.encoder_lr},
                                     {"params": head.parameters(), "lr": args.head_lr}], weight_decay=1e-4)
            rng = np.random.default_rng(args.seed)
            best, best_state, history = None, None, []
            started = time.monotonic()
            for step in range(1, args.steps + 1):
                idx = rng.integers(0, len(R["x"]), size=args.batch_size)
                tx, tm = batch(R, idx)
                logits = head(run_suffix(tx, tm))
                target = torch.from_numpy(R["teacher"][idx]).to(device)
                kd = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(target / 2, 1), reduction="batchmean") * 4
                loss = (1 - fraction) * (.3 * F.cross_entropy(logits, yr[idx]) + .7 * kd)
                opt.zero_grad()
                loss.backward()
                # Draw in both arms so the original replay batches stay matched.
                yi = rng.integers(0, len(Y["x"]), size=max(4, args.batch_size // 4))
                if use_youtube:
                    tx, tm = batch(Y, yi)
                    yloss = fraction * F.cross_entropy(head(run_suffix(tx, tm)), yy[yi])
                    yloss.backward()
                torch.nn.utils.clip_grad_norm_(list(suffix.parameters()) + list(head.parameters()), 1.)
                opt.step()
                if step % 10 == 0:
                    print("[{}] step {} {:.1f}s".format(tag, step, time.monotonic() - started), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    score, logits = evaluate(head)
                    history.append({"step": step, **score})
                    key = score["foreign_top1"] + score["foreign_macro_top1"]
                    if best is None or key > best["score"]:
                        best = {"step": step, "score": key, **score}
                        best_state = ({k: v.detach().cpu().clone() for k, v in suffix.state_dict().items()},
                                      head.weight.detach().cpu().numpy().copy(),
                                      head.bias.detach().cpu().numpy().copy(), logits.copy())
                    print("[{} eval {}] micro {:.4f} macro {:.4f} Korea {:.4f}".format(
                        tag, step, score["foreign_top1"], score["foreign_macro_top1"], score["korean_top1"]), flush=True)
            folder = args.out / tag
            deploy = folder / "deploy"
            deploy.mkdir(parents=True, exist_ok=True)
            suffix.load_state_dict(best_state[0])
            torch.save({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, deploy / "encoder_pure_l17.pt")
            nh = dict(h, W=best_state[1], b=best_state[2])
            np.savez(deploy / "head_np.npz", **nh)
            np.savez(folder / "validation_logits.npz", logits=best_state[3],
                     job_ids=np.array([r["job_id"] for r in records]), labels=np.array(labels))
            dump_json(folder / "history.json", history)
            info = {"best_validation": best, "uses_youtube": use_youtube, "fraction": fraction,
                    "unfreeze_from": args.unfreeze_from, "steps": args.steps, "seed": args.seed,
                    "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
                    "source": fingerprint, "replay_sha256": sha(args.replay),
                    "youtube_sha256": sha(args.youtube_train), "validation_sha256": sha(args.validation_responses),
                    "encoder_sha256": sha(deploy / "encoder_pure_l17.pt"), "head_sha256": sha(deploy / "head_np.npz"),
                    "npu_verification": "pending; GPU reference only"}
            dump_json(folder / "result.json", info)
            dump_json(deploy / "encoder_pair.json", info)
            runs.append(dict(tag=tag, **info))
    dump_json(args.out / "summary.json", {"baseline": baseline_metrics, "runs": runs,
             "note": "Validation selects checkpoints; no production replacement; requires NPU verification."})
    print("[done] " + str(args.out), flush=True)


if __name__ == "__main__":
    main()

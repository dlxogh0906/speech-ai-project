"""Expand training-speaker replay and compare frozen-encoder head refinements.

Validation uses cached NPU parts with the serving probability clipping and gate.
The independent test split is never loaded by this script.
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
import json
import time

import numpy as np
from pilot_youtube_head import read_audio, read_rows, sha, dump_json


def save_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_head(path):
    with np.load(path, allow_pickle=True) as values:
        return {key: values[key].copy() for key in values.files}


def select_round_robin(by_speaker, count, rng, prefer_new=None):
    queues = [(speaker, list(rows)) for speaker, rows in sorted(by_speaker.items())]
    for _, rows in queues:
        rng.shuffle(rows)
    rng.shuffle(queues)
    if prefer_new is not None:
        queues.sort(key=lambda pair: pair[0] in prefer_new)
    selected = []
    while len(selected) < count and any(rows for _, rows in queues):
        for _, rows in queues:
            if rows and len(selected) < count:
                row = rows.pop()
                if _pl.Path(row["audio_path"]).is_file():
                    selected.append(row)
    return selected


def prepare(args, base):
    import torch
    from country_classifier.npu_encoder import load_pure, conv_out_len
    from country_classifier.wav_augment import silence_normalize
    labels = list(map(str, base["labels"]))
    old_rows = read_rows(args.prior / "replay.csv")
    excluded = {row["audio_path"] for row in old_rows}
    old_speakers = collections.defaultdict(set)
    for row in old_rows:
        old_speakers[row["label_en"]].add(row["speaker"])
    sentence, word = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(2))
    for row in read_rows(args.original_train):
        if row["label_en"] not in labels or row["audio_path"] in excluded:
            continue
        sec = float(row["duration_sec"] or 0)
        if row["task"] != "RW" and sec >= 3:
            sentence[row["label_en"]][row["speaker"]].append(row)
        elif row["task"] == "RW" and sec >= 1:
            word[row["label_en"]][row["speaker"]].append(row)
    rng = np.random.default_rng(args.seed)
    extra, short = [], []
    for label in labels:
        extra.extend(select_round_robin(sentence[label], args.extra_per_country, rng, old_speakers[label]))
        selected = select_round_robin(word[label], args.short_per_country, rng)
        for row in selected:
            short.append(dict(row, feature_kind="word", crop_start_sec="0", crop_duration_sec="8"))
        needed = args.short_per_country - len(selected)
        if needed:
            for row in select_round_robin(sentence[label], needed, rng):
                length = float(rng.uniform(2, 4))
                start = float(rng.uniform(0, max(0, float(row["duration_sec"]) - length)))
                short.append(dict(row, feature_kind="short_crop", crop_start_sec=str(start), crop_duration_sec=str(length)))
    missing = [r["audio_path"] for r in extra + short if not _pl.Path(r["audio_path"]).is_file()]
    if missing:
        raise ValueError("Missing training audio: " + str(missing[:5]))
    youtube = read_rows(args.youtube_train)
    valid = [json.loads(line) for line in args.validation_responses.read_text().splitlines()]
    train_speakers = {r["speaker"] for r in old_rows + extra + short + youtube}
    if train_speakers & {r["speaker"] for r in valid}:
        raise ValueError("Training/validation speaker overlap")
    meta = {"encoder_sha256": sha(args.main_deploy / "encoder_pure_l17.pt"),
            "production_head_sha256": sha(args.main_deploy / "head_np.npz"),
            "prior_features_sha256": sha(args.prior / "features.npz"),
            "prior_replay_sha256": sha(args.prior / "replay.csv"),
            "youtube_train_sha256": sha(args.youtube_train), "original_train_sha256": sha(args.original_train),
            "extra_per_country": args.extra_per_country, "short_per_country": args.short_per_country,
            "seed": args.seed, "extra_rows_sha256": hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest(),
            "short_rows_sha256": hashlib.sha256(json.dumps(short, sort_keys=True).encode()).hexdigest(),
            "preprocessing": "first8s_static12s_silence_normalize_rms.0015"}
    prior_meta = json.loads((args.prior / "features_meta.json").read_text())
    if prior_meta["encoder"] != meta["encoder_sha256"] or prior_meta["head"] != meta["production_head_sha256"]:
        raise ValueError("Prior features belong to a different encoder/head")
    reusable_replay = None
    if args.reuse_replay_dir:
        previous = json.loads((args.reuse_replay_dir / "features_meta.json").read_text())
        for key, value in meta.items():
            if key != "youtube_train_sha256" and previous.get(key) != value:
                raise ValueError("Reusable replay feature conditions differ: " + key)
        reusable_replay = args.reuse_replay_dir / "training_features.npz"
        meta["reused_replay_features_sha256"] = sha(reusable_replay)
    cache = args.out / "training_features.npz"
    if cache.exists():
        if json.loads((args.out / "features_meta.json").read_text()) != meta:
            raise ValueError("Cached feature fingerprint differs")
        print("[prepare] using completed cache", flush=True)
        return
    save_rows(args.out / "extra_sentences.csv", extra)
    save_rows(args.out / "short_speech.csv", short)
    save_rows(args.out / "sentence_replay.csv", old_rows + extra)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    encoder = load_pure(args.main_deploy / "encoder_pure_l17.pt").to(device)
    mu, sd = base["mu"].reshape(-1), base["sd"].reshape(-1)
    started = time.monotonic()

    @torch.no_grad()
    def extract(rows, name):
        path = args.out / (name + "_features.npy")
        if path.exists():
            stamp = json.loads(path.with_suffix(".json").read_text())
            if stamp != meta:
                raise ValueError("Partial feature fingerprint differs")
            return np.load(path)
        features = []
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            source = np.zeros((len(batch), 192000), np.float32)
            mask = np.zeros((len(batch), conv_out_len(192000)), np.float32)
            for i, row in enumerate(batch):
                wave = read_audio(row["audio_path"])
                start = int(float(row.get("crop_start_sec", 0)) * 16000)
                length = int(float(row.get("crop_duration_sec", 8)) * 16000)
                wave = silence_normalize(wave[start:start + length])[:192000]
                source[i, :len(wave)] = wave
                mask[i, :max(0, conv_out_len(len(wave)))] = 1
            pooled = encoder(torch.from_numpy(source).to(device), torch.from_numpy(mask).to(device)).cpu().numpy()
            features.append((np.clip(np.nan_to_num(pooled), -50, 50) - mu) / sd)
            if offset % (args.batch_size * 25) == 0:
                print("[{}] {}/{} elapsed {:.0f}s".format(name, offset, len(rows), time.monotonic() - started), flush=True)
        values = np.concatenate(features)
        np.save(path, values)
        dump_json(path.with_suffix(".json"), meta)
        return values

    if reusable_replay:
        with np.load(reusable_replay) as previous:
            sentence_features = previous["sentence"].copy()
            short_features = previous["short"].copy()
        if len(sentence_features) != len(old_rows + extra) or len(short_features) != len(short):
            raise ValueError("Reusable replay feature row counts differ")
    else:
        extra_features = extract(extra, "extra")
        short_features = extract(short, "short")
        with np.load(args.prior / "features.npz") as prior:
            sentence_features = np.concatenate([prior["replay"], extra_features])
    youtube_features = (extract(youtube, "youtube")
                        if prior_meta["youtube"] != meta["youtube_train_sha256"] else None)
    with np.load(args.prior / "features.npz") as prior:
        np.savez(cache, sentence=sentence_features, short=short_features,
                 youtube=prior["youtube"] if youtube_features is None else youtube_features)
    dump_json(args.out / "features_meta.json", meta)
    dump_json(args.out / "dataset_summary.json", {
        "sentence_clips": len(old_rows + extra), "sentence_speakers": len({r["speaker"] for r in old_rows + extra}),
        "short_clips": len(short), "short_speakers": len({r["speaker"] for r in short}),
        "short_types": dict(collections.Counter(r["feature_kind"] for r in short)),
        "youtube_clips": len(youtube), "youtube_speakers": len({r["speaker"] for r in youtube}),
        "all_train_speakers": len(train_speakers), "validation_overlap": 0,
        "new_sentence_speakers": len({r["speaker"] for r in extra} - {r["speaker"] for r in old_rows})})
    print("[prepare done] " + str(cache), flush=True)


def validation_parts(args, base):
    ids = {json.loads(s)["job_id"] for s in args.validation_responses.read_text().splitlines()}
    records = {}
    for path in sorted(args.npu_validation.glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["job_id"] in ids:
                records[row["job_id"]] = row
    if set(records) != ids:
        raise ValueError("Incomplete NPU validation cache")
    records = [records[key] for key in sorted(records)]
    features, gates, starts, counts = [], [], [], []
    labels = list(map(str, base["labels"]))
    for row in records:
        with np.load(args.npu_validation / "pooled" / (row["capture_rid"] + ".npz")) as f:
            if list(map(str, f["labels"])) != labels:
                raise ValueError("NPU feature labels differ")
            starts.append(sum(counts))
            counts.append(len(f["pooled"]))
            features.append((f["pooled"] - base["mu"]) / base["sd"])
            g = np.log(np.clip(f["gate"], 1e-12, 1)).mean(0)
            pg = np.exp(g - g.max()); pg /= pg.sum()
            gates.append(pg[labels.index("Korea")] > .9)
    return records, np.concatenate(features), np.array(starts), np.array(counts), np.array(gates)


def train(args, base):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    anchor = load_head(args.anchor_head)
    meta = json.loads((args.out / "features_meta.json").read_text())
    expected = {"encoder_sha256": sha(args.main_deploy / "encoder_pure_l17.pt"),
                "production_head_sha256": sha(args.main_deploy / "head_np.npz"),
                "youtube_train_sha256": sha(args.youtube_train), "original_train_sha256": sha(args.original_train),
                "prior_replay_sha256": sha(args.prior / "replay.csv"),
                "prior_features_sha256": sha(args.prior / "features.npz")}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Training feature provenance differs from requested encoder or manifests")
    for key in ("mu", "sd", "labels"):
        if not np.array_equal(anchor[key], base[key]):
            raise ValueError("Anchor and deployed encoder conditions differ")
    labels = list(map(str, base["labels"])); c2i = {c: i for i, c in enumerate(labels)}
    korea = c2i["Korea"]
    old_rows = read_rows(args.prior / "replay.csv")
    all_rows = read_rows(args.out / "sentence_replay.csv")
    short_rows = read_rows(args.out / "short_speech.csv")
    youtube_rows = read_rows(args.youtube_train)
    with np.load(args.out / "training_features.npz") as f:
        arrays = {k: f[k].copy() for k in f.files}
    arrays["small"] = arrays["sentence"][:len(old_rows)]
    rows = {"small": old_rows, "sentence": all_rows, "short": short_rows, "youtube": youtube_rows}
    if args.sentence_domain == "web":
        for key in ("small", "sentence"):
            available = {r["label_en"] for r in rows[key] if r["domain"].startswith("web")}
            keep = [i for i, r in enumerate(rows[key]) if r["domain"].startswith("web")
                    or r["label_en"] == "Korea" or r["label_en"] not in available]
            arrays[key] = arrays[key][keep]
            rows[key] = [rows[key][i] for i in keep]
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    targets = {k: torch.tensor([c2i[r["label_en"]] for r in rows[k]], dtype=torch.long) for k in rows}
    initial = base if args.restart_from_production else anchor
    teacher = {k: torch.from_numpy(v @ initial["W"].T + initial["b"]) for k, v in arrays.items()}
    country_speakers = collections.defaultdict(set)
    for row in read_rows(args.original_train):
        country_speakers[row["label_en"]].add(row["speaker"])
    probabilities = {}
    for key in rows:
        counts_by_label = collections.Counter(r["label_en"] for r in rows[key])
        p = np.array([len(country_speakers[r["label_en"]]) ** args.country_sampling_power / counts_by_label[r["label_en"]]
                      for r in rows[key]], dtype=np.float64)
        probabilities[key] = p / p.sum()
    records, xv, starts, counts, gate = validation_parts(args, base)
    yv = np.array([c2i[r["country"]] for r in records])
    foreign = yv != korea

    def evaluate(w, b):
        z = xv @ w.T + b
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        logp = np.log(np.clip(p, 1e-12, 1))
        agg = np.add.reduceat(logp, starts, axis=0) / counts[:, None]
        prediction = agg.argmax(1); prediction[gate] = korea
        hit = prediction == yv
        per = {c: float(hit[yv == c2i[c]].mean()) for c in sorted({r["country"] for r in records})}
        return {"foreign_top1": float(hit[foreign].mean()),
                "foreign_macro_top1": float(np.mean([v for c, v in per.items() if c != "Korea"])),
                "korean_top1": float(hit[~foreign].mean()) if (~foreign).any() else None, "per_country_top1": per}, prediction

    baseline, pred = evaluate(base["W"], base["b"])
    expected = np.array([c2i[r["variants"]["baseline"]["top5"][0]["label"]] for r in records])
    if not np.array_equal(pred, expected):
        raise ValueError("Validation inference differs from recorded NPU baseline")
    previous, _ = evaluate(anchor["W"], anchor["b"])
    configs = [
        {"tag": "small_replay", "pool": "small", "short": 0., "youtube": .1, "use_youtube": False, "kd": .7},
        {"tag": "expanded_replay", "pool": "sentence", "short": 0., "youtube": .1, "use_youtube": False, "kd": .7},
        {"tag": "expanded_short", "pool": "sentence", "short": .2, "youtube": .1, "use_youtube": False, "kd": .7},
        {"tag": "expanded_short_youtube", "pool": "sentence", "short": .2, "youtube": .1, "use_youtube": True, "kd": .7},
        {"tag": "expanded_short_youtube_kd40", "pool": "sentence", "short": .2, "youtube": .1, "use_youtube": True, "kd": .4},
    ]
    selected_configs = {x for x in args.arms.split(",") if x}
    configs = [c for c in configs if not selected_configs or c["tag"] in selected_configs]
    for config in configs:
        config["youtube"] = args.youtube_fraction
        if config["short"]:
            config["short"] = args.short_fraction
    youtube_counts = collections.Counter(r["speaker"] for r in youtube_rows)
    youtube_p = np.array([1. / youtube_counts[r["speaker"]] for r in youtube_rows])
    youtube_p /= youtube_p.sum()
    outcomes = []
    for config in configs:
        tag = args.tag_prefix + config["tag"]
        torch.manual_seed(args.seed)
        head = torch.nn.Linear(arrays["sentence"].shape[1], len(labels))
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(initial["W"])); head.bias.copy_(torch.from_numpy(initial["b"]))
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        history, best, best_state, diagnostic, diagnostic_state = [], None, None, None, None

        def replay_loss(key, indexes):
            logits = head(tensors[key][indexes])
            truth = targets[key][indexes]
            kd = F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(teacher[key][indexes] / 2, 1), reduction="none").sum(1) * 4
            weight = config["kd"] * (teacher[key][indexes].argmax(1) == truth).float()
            ce = F.cross_entropy(logits, truth, reduction="none")
            return ((1 - weight) * ce + weight * kd).mean()

        for step in range(1, args.steps + 1):
            # All arms draw all three indices, preserving matched replay batches.
            if args.country_sampling_power or args.sentence_domain != "all":
                si = torch.from_numpy(rng.choice(len(arrays[config["pool"]]), 256, p=probabilities[config["pool"]]))
                wi = torch.from_numpy(rng.choice(len(arrays["short"]), 64, p=probabilities["short"]))
            else:
                si = torch.from_numpy(rng.integers(0, len(arrays[config["pool"]]), 256))
                wi = torch.from_numpy(rng.integers(0, len(arrays["short"]), 64))
            yi = torch.from_numpy(rng.choice(len(arrays["youtube"]), 64, p=youtube_p)
                                 if args.youtube_speaker_balanced else rng.integers(0, len(arrays["youtube"]), 64))
            loss = (1 - config["short"] - config["youtube"]) * replay_loss(config["pool"], si)
            if config["short"]:
                loss += config["short"] * replay_loss("short", wi)
            if config["use_youtube"]:
                loss += config["youtube"] * F.cross_entropy(head(tensors["youtube"][yi]), targets["youtube"][yi])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % args.eval_every == 0 or step == args.steps:
                w, b = head.weight.detach().numpy(), head.bias.detach().numpy()
                m, _ = evaluate(w, b)
                score = m["foreign_top1"] + m["foreign_macro_top1"]
                korea_ok = previous["korean_top1"] is None or m["korean_top1"] >= previous["korean_top1"]
                eligible = korea_ok and m["foreign_top1"] >= previous["foreign_top1"]
                history.append(dict(step=step, eligible=eligible, score=score, **m))
                if diagnostic is None or score > diagnostic["score"]:
                    diagnostic = dict(step=step, score=score, **m)
                    diagnostic_state = (w.copy(), b.copy())
                if eligible and (best is None or score > best["score"]):
                    best = dict(step=step, score=score, **m)
                    best_state = (w.copy(), b.copy())
                print("[{} {}] micro {:.4f} macro {:.4f} eligible {}".format(tag, step, m["foreign_top1"], m["foreign_macro_top1"], eligible), flush=True)
        directory = args.out / tag
        directory.mkdir(exist_ok=True)
        dump_json(directory / "history.json", history)
        diagnostic_values = dict(anchor)
        diagnostic_values["W"], diagnostic_values["b"] = diagnostic_state
        np.savez(directory / "diagnostic_head.npz", **diagnostic_values)
        if best_state is None:
            best = dict(step=0, score=previous["foreign_top1"] + previous["foreign_macro_top1"], **previous)
            best_state = (anchor["W"], anchor["b"])
        values = dict(anchor); values["W"], values["b"] = best_state
        np.savez(directory / "head_np.npz", **values)
        result = {"config": config, "best_validation": best, "steps": args.steps,
                  "diagnostic_validation": diagnostic, "country_sampling_power": args.country_sampling_power,
                  "restart_from_production": args.restart_from_production,
                  "youtube_speaker_balanced": args.youtube_speaker_balanced,
                  "sentence_domain": args.sentence_domain,
                  "lr": args.lr, "seed": args.seed, "anchor_sha256": sha(args.anchor_head)}
        dump_json(directory / "result.json", result)
        outcomes.append(dict(tag=tag, **result))
    summary = {"baseline": baseline, "previous_candidate": previous, "runs": outcomes,
               "selection": "micro + country mean, subject to no reduction in foreign micro or Korean validation recall",
               "evaluation": "Original validation only; actual NPU parts and fixed gate; exploratory repeated selection"}
    path = args.out / (args.tag_prefix + "training_summary.json")
    dump_json(path, summary)
    print("[train done] " + str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--original-train", type=_pl.Path, required=True)
    parser.add_argument("--youtube-train", type=_pl.Path, required=True)
    parser.add_argument("--validation-responses", type=_pl.Path, required=True)
    parser.add_argument("--main-deploy", type=_pl.Path, required=True)
    parser.add_argument("--prior", type=_pl.Path, required=True)
    parser.add_argument("--reuse-replay-dir", type=_pl.Path,
                        help="Reuse matching frozen sentence/short features while extracting a new YouTube manifest")
    parser.add_argument("--anchor-head", type=_pl.Path, required=True)
    parser.add_argument("--npu-validation", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--extra-per-country", type=int, default=200)
    parser.add_argument("--short-per-country", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", default="")
    parser.add_argument("--tag-prefix", default="")
    parser.add_argument("--country-sampling-power", type=float, default=0.,
                        help="Replay probability proportional to training-speaker count to this power; zero balances countries")
    parser.add_argument("--restart-from-production", action="store_true",
                        help="Initialize and distill from production while retaining the previous candidate as the selection floor")
    parser.add_argument("--short-fraction", type=float, default=.2)
    parser.add_argument("--youtube-fraction", type=float, default=.1)
    parser.add_argument("--youtube-speaker-balanced", action="store_true",
                        help="Give each reviewed YouTube training speaker equal sampling probability")
    parser.add_argument("--sentence-domain", choices=("all", "web"), default="all",
                        help="Prefer browser recordings; keep all Korean sources and fallback countries without browser speech")
    args = parser.parse_args()
    if (not 0 <= args.short_fraction <= .9 or not 0 < args.youtube_fraction < 1
            or args.short_fraction + args.youtube_fraction >= 1
            or args.steps < 1 or args.eval_every < 1 or args.lr <= 0):
        parser.error("Invalid training fraction, steps, evaluation interval, or learning rate")
    args.out.mkdir(parents=True, exist_ok=True)
    base = load_head(args.main_deploy / "head_np.npz")
    (prepare if args.mode == "prepare" else train)(args, base)


if __name__ == "__main__":
    main()

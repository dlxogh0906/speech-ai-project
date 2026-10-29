"""Isolated noise-augmentation pilot, with a matched clean-training control.

Only TSV/LTR/CANON inputs are read. Actual-phone LTR labels remain unchanged.
Training/checkpoint selection never reads test predictions or speaker metadata.
"""

import sys as _sys
import pathlib as _pl
_root = _pl.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
_sys.path[:0] = [str(_root), *(str(p) for p in sorted(_root.glob("tasks/*/src"))),
                str(_pl.Path(__file__).resolve().parents[1] / "eval")]

import argparse
import collections
import hashlib
import json
import random
import time

import numpy as np

from shared.paths import DATA_ROOT, MDD_OUTPUTS
from mdd.noise_augment import NoiseAugmenter, read_pcm
from train_w2vbert_ctc import batches_by_frames, tri_stage


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with _pl.Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def selected_data(manifest, split, selected):
    """Read requested TSV/LTR rows, without materializing the entire corpus."""
    wanted = set(selected)
    rows, labels = {}, {}
    with (manifest / (split + ".tsv")).open() as stream:
        root = _pl.Path(next(stream).rstrip("\n"))
        for i, line in enumerate(stream):
            if i in wanted:
                path, frames = line.rstrip("\n").split("\t")
                rows[i] = (root / path, int(frames))
    with (manifest / (split + ".ltr")).open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                labels[i] = line.split()
    if set(rows) != wanted or set(labels) != wanted:
        raise ValueError("Selected rows missing")
    return [rows[i] for i in selected], [labels[i] for i in selected]


def noise_paths(plan, split):
    source_split = "train" if split == "train" else "valid"
    manifest = _pl.Path(plan["noise_manifest"])
    selected = set(plan["noise_indices"][split])
    result = []
    with (manifest / (source_split + ".tsv")).open() as stream:
        root = _pl.Path(next(stream).rstrip("\n"))
        for i, line in enumerate(stream):
            if i in selected:
                result.append(root / line.rstrip("\n").split("\t")[0])
    if len(result) != len(selected):
        raise ValueError("Noise selection missing")
    return result


def prepare(args):
    import compare_noise_mixing as evaluation
    import compare_noise_preprocess as previous

    out = _pl.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise RuntimeError("Refusing to overwrite experiment plan")
    vocab = json.loads((_pl.Path(args.ckpt) / "jamo_vocab.json").read_text())["vocab"]
    manifest = _pl.Path(args.train_manifest)
    pool = {"normal": [], "error": []}
    with (manifest / "train.tsv").open() as ft, \
         (manifest / "train.ltr").open(encoding="utf-8") as fl, \
         (manifest / "train.canon").open(encoding="utf-8") as fc:
        root = _pl.Path(next(ft).rstrip("\n"))
        for i, (row, tag, canon) in enumerate(zip(ft, fl, fc)):
            path, frames = row.rstrip("\n").split("\t")
            if not 1.5 * 16000 <= int(frames) <= 10 * 16000:
                continue
            toks = tag.split()
            if not toks or not canon.strip() or any(t not in vocab for t in toks):
                continue
            pool["normal" if toks == canon.split() else "error"].append(i)
    rng = random.Random(args.seed)
    selected = rng.sample(pool["normal"], int(args.train_n * 0.6))
    selected += rng.sample(pool["error"], args.train_n - len(selected))
    rng.shuffle(selected)
    train_items, train_labels = selected_data(manifest, "train", selected)
    train_paths = {str(p) for p, _ in train_items}
    if len(train_paths) != len(selected):
        raise RuntimeError("Duplicate training paths")
    # Confirm each selected TAG clip belongs to the original stage-2 train split.
    production = _pl.Path(args.manifest)
    original_members = set()
    with (production / "train.tsv").open() as stream:
        proot = _pl.Path(next(stream).rstrip("\n"))
        for line in stream:
            p = str(proot / line.rstrip("\n").split("\t")[0])
            if p in train_paths:
                original_members.add(p)
    if original_members != train_paths:
        raise RuntimeError("Pilot samples outside original training split")
    valid_data = evaluation.load_split(production, "valid")
    test_data = evaluation.load_split(production, "test")
    excluded = set(train_paths)
    for path in args.exclude_prior:
        prior = json.loads(_pl.Path(path).read_text())
        prior_split = prior.get("split", prior["args"].get("split", "test"))
        data = valid_data if prior_split == "valid" else test_data
        excluded.update(str(data[0] / data[1][i][0]) for i in prior["source_row_indices"])
    vsel, vstats, vpaths = evaluation.choose_samples(
        valid_data, args.valid_per_cohort, args.valid_per_cohort, args.seed + 1, excluded)
    tsel, tstats, tpaths = evaluation.choose_samples(
        test_data, args.test_per_cohort, args.test_per_cohort, args.seed + 2, excluded | vpaths)
    unseen_data = evaluation.load_split(args.unseen_manifest, "test")
    order = list(range(len(unseen_data[1])))
    rng.shuffle(order)
    usel, upaths = [], set()
    unseen_excluded = excluded | vpaths | tpaths
    for i in order:
        row = unseen_data[1][i]
        path = str(unseen_data[0] / row[0])
        if path in unseen_excluded or not 1.5 * 16000 <= int(row[1]) <= 40 * 16000:
            continue
        tag, canon = unseen_data[2][i].split(), unseen_data[3][i].split()
        if not tag or not canon or any(t not in vocab for t in tag + canon):
            continue
        if not previous.audio_check(previous.read_wav(path))["pass"]:
            continue
        usel.append((i, "normal" if tag == canon else "error"))
        upaths.add(path)
        if len(usel) == args.unseen_n:
            break
    if len(usel) != args.unseen_n:
        raise RuntimeError("Insufficient unseen paragraph clips")
    # Short utterance canons must never be called unseen for this checkpoint.
    nroot, nrows = previous.read_rows(args.noise_manifest, "train")
    nvalid_root, nvalid_rows = previous.read_rows(args.noise_manifest, "valid")
    target_paths = train_paths | vpaths | tpaths | upaths
    noise_train_candidates = [i for i, row in enumerate(nrows)
                              if int(row[1]) >= 16000 and str(nroot / row[0]) not in target_paths]
    noise_train = rng.sample(noise_train_candidates, args.noise_n)
    unique_valid = list(dict.fromkeys(str(nvalid_root / row[0]) for row in nvalid_rows))
    if len(unique_valid) != len(nvalid_rows):
        raise RuntimeError("Duplicate validation noise source")
    norder = list(range(len(nvalid_rows)))
    rng.shuffle(norder)
    half = len(norder) // 2
    noise_indices = {"train": noise_train, "valid": norder[:half], "test": norder[half:]}
    noise_sets = {"train": {str(nroot / nrows[i][0]) for i in noise_train},
                  "valid": {str(nvalid_root / nvalid_rows[i][0]) for i in noise_indices["valid"]},
                  "test": {str(nvalid_root / nvalid_rows[i][0]) for i in noise_indices["test"]}}
    audio_sets = {"train": train_paths, "valid": vpaths, "test": tpaths, "unseen": upaths}
    for mapping in (audio_sets, noise_sets):
        names = list(mapping)
        for i, name in enumerate(names):
            for other in names[:i]:
                if mapping[name] & mapping[other]:
                    raise RuntimeError("Audio split overlap: " + name + "/" + other)
    if set.union(*noise_sets.values()) & set.union(*audio_sets.values()):
        raise RuntimeError("Noise contains selected target recordings")
    checkpoints = sorted(set(args.checkpoints))
    plan = {"seed": args.seed, "ckpt": args.ckpt, "train_manifest": args.train_manifest,
            "manifest": args.manifest, "unseen_manifest": args.unseen_manifest,
            "noise_manifest": args.noise_manifest, "train_indices": selected,
            "valid_selection": vsel, "test_selection": tsel, "unseen_selection": usel,
            "valid_selection_stats": vstats, "test_selection_stats": tstats,
            "noise_indices": noise_indices, "checkpoints": checkpoints,
            "training": {"max_updates": max(checkpoints), "lr": args.lr,
                         "batch_frames": args.batch_frames, "accum": args.accum,
                         "noise_probability": 0.5, "spec_augment": False,
                         "trainable": "all encoder layers and CTC head",
                         "train_normal_error_ratio": "60:40; human TAG only",
                         "native_replay": False,
                         "augment_distribution_when_noisy": "40% babble 5..20dB; 30% rumble 0..15dB; 20% reverb+babble; 10% reverb"},
            "selection_policy": {
                "objective": "maximize mean babble10/5 actual error preservation, valid only",
                "clean_recall_preservation_loss_max_pp": 2.0,
                "clean_frr_increase_max_pp": 0.3, "clean_jer_increase_max_pp": 1.0,
                "primary_frr_increase_max_pp": 0.5,
                "fallback": "if no eligible checkpoint, diagnose final checkpoint without adoption recommendation",
                "control": "identical clean-only continuation, same update count as selected noisy checkpoint"},
            "test_policy": "Frozen checkpoint and calibration; full paired scoring; no test-driven retuning",
            "unseen_scope": "Existing RP002-004 actual-phone holdout, full audio <=40s. Mostly beyond production 20s limit; secondary diagnostic, not API deployment validation.",
            "noise_snr_convention": "whole-utterance RMS; no speed/pitch changes",
            "norms": str(MDD_OUTPUTS / "score_phoneme/norms.json"),
            "calib": str(MDD_OUTPUTS / "score_phoneme/calibration.json"),
            "source_hashes": {str(p.relative_to(_root)): digest(p) for p in (
                _pl.Path(__file__).resolve(), _root / "tasks/pronunciation-assessment/src/mdd/noise_augment.py")},
            "input_hashes": {str(p): digest(p) for p in (
                manifest / "train.tsv", manifest / "train.ltr", manifest / "train.canon",
                production / "valid.tsv", production / "valid.ltr", production / "valid.canon",
                production / "test.tsv", production / "test.ltr", production / "test.canon",
                _pl.Path(args.ckpt) / "jamo_vocab.json", _pl.Path(args.ckpt) / "config.json")}}
    write_json(out / "plan.json", plan)
    write_json(out / "audit.json", {
        "selected_targets": {k: len(v) for k, v in audio_sets.items()},
        "noise_files": {k: len(v) for k, v in noise_sets.items()},
        "target_path_overlap": 0, "noise_path_overlap": 0, "target_noise_path_overlap": 0,
        "train_belongs_to_original_stage2_train": True,
        "speaker_overlap": "not audited; no speaker metadata read",
        "training_targets": "human actual-phone .ltr only; .canon used for strata/evaluation only",
        "labels_sha256": hashlib.sha256(json.dumps(train_labels, ensure_ascii=False).encode()).hexdigest(),
        "no_waveform_outputs": True, "production_changed": False})
    print("[prepared] train=%d valid=%d test=%d unseen=%d" %
          (len(selected), len(vsel), len(tsel), len(usel)), flush=True)


def train(args):
    import torch
    from torch.cuda.amp import autocast, GradScaler
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC

    out = _pl.Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    save = out / "checkpoints" / args.mode
    if (save / "complete.json").exists() or list(save.glob("step_*")):
        raise RuntimeError("Refusing to overwrite training checkpoints")
    save.mkdir(parents=True, exist_ok=True)
    settings = plan["training"]
    max_updates = args.updates or settings["max_updates"]
    seed = plan["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_info = json.loads((_pl.Path(plan["ckpt"]) / "jamo_vocab.json").read_text())
    vocab = vocab_info["vocab"]
    items, labels = selected_data(_pl.Path(plan["train_manifest"]), "train", plan["train_indices"])
    targets = [[vocab[t] for t in y] for y in labels]
    # No silent OOV removal, waveform truncation, canonical targets or label edits.
    if any(not y for y in targets):
        raise ValueError("Empty target")
    model = Wav2Vec2BertForCTC.from_pretrained(plan["ckpt"]).to(device)
    model.config.ctc_zero_infinity = True
    model.config.apply_spec_augment = False
    model.train()
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(plan["ckpt"])
    augmenter = NoiseAugmenter(noise_paths(plan, "train"),
                               probability=settings["noise_probability"] if args.mode == "noise" else 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.98),
                                  eps=1e-8, weight_decay=0.0)
    scaler = GradScaler(enabled=device.type == "cuda", init_scale=1024.0)

    class Batches(torch.utils.data.Dataset):
        def __init__(self, epoch):
            self.epoch = epoch
            self.batches = batches_by_frames(items, settings["batch_frames"], shuffle_seed=seed + epoch)
        def __len__(self):
            return len(self.batches)
        def __getitem__(self, k):
            indices = self.batches[k]
            waves, conditions = [], []
            for i in indices:
                x = read_pcm(items[i][0])
                if len(x) != items[i][1]:
                    raise ValueError("Manifest frame count mismatch")
                x, condition = augmenter(x, seed * 1000003 + self.epoch * 100003 + i)
                waves.append(x)
                conditions.append(condition)
            features = extractor(waves, sampling_rate=16000, return_tensors="pt", padding=True)
            ys = [targets[i] for i in indices]
            lab = torch.full((len(indices), max(map(len, ys))), -100, dtype=torch.long)
            for j, y in enumerate(ys):
                lab[j, :len(y)] = torch.tensor(y)
            return features["input_features"], features["attention_mask"], lab, conditions

    started = time.monotonic()
    step = micro = skipped = 0
    losses = []
    counts = collections.Counter()
    epoch = 0
    checkpoints = set(plan["checkpoints"] if not args.updates else [args.updates])
    log_path = save / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        while step < max_updates:
            kwargs = {"num_workers": args.workers, "batch_size": None, "pin_memory": device.type == "cuda"}
            if args.workers:
                kwargs["prefetch_factor"] = 2
            loader = torch.utils.data.DataLoader(Batches(epoch), **kwargs)
            for feats, mask, labels, conditions in loader:
                counts.update(conditions)
                feats = feats.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=device.type == "cuda"):
                    loss = model(input_features=feats, attention_mask=mask, labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite CTC loss")
                losses.append(float(loss.detach()))
                scaler.scale(loss / settings["accum"]).backward()
                micro += 1
                if micro % settings["accum"]:
                    continue
                lr = tri_stage(step, settings["max_updates"], settings["lr"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() < old_scale:
                    skipped += 1
                    if skipped > 20:
                        raise RuntimeError("Repeated nonfinite gradients")
                    continue
                step += 1
                if step == 1 or step % 10 == 0 or step == max_updates:
                    elapsed = time.monotonic() - started
                    record = {"mode": args.mode, "step": step, "max_updates": max_updates,
                              "loss": float(np.mean(losses[-40:])), "lr": lr,
                              "grad_norm": float(grad), "skipped_updates": skipped,
                              "elapsed_sec": round(elapsed, 1),
                              "eta_sec": round(elapsed / step * (max_updates - step), 1),
                              "augmented_counts": dict(counts)}
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if step in checkpoints:
                    path = save / ("step_%d" % step)
                    model.save_pretrained(path)
                    extractor.save_pretrained(path)
                    write_json(path / "jamo_vocab.json", vocab_info)
                    write_json(path / "pilot.json", {"step": step, "mode": args.mode,
                               "plan_sha256": digest(out / "plan.json"), "seed": seed,
                               "actual_ltr_targets": True, "skipped_updates": skipped})
                    # Preserve an optimizer state to make an interrupted extension possible.
                    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                                "epoch": epoch, "micro": micro}, save / "optimizer_last.pt")
                    print("[saved] %s step=%d" % (args.mode, step), flush=True)
                if step == max_updates:
                    break
            epoch += 1
    write_json(save / "complete.json", {"step": step, "mode": args.mode,
               "elapsed_sec": time.monotonic() - started, "counts": dict(counts),
               "skipped_updates": skipped, "training_examples_seen": sum(counts.values()),
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
    print("[training complete] " + args.mode, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "train"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("clean", "noise"), default="noise")
    ap.add_argument("--manifest", default=str(MDD_OUTPUTS / "manifest_full_tag_replay10"))
    ap.add_argument("--train-manifest", default=str(MDD_OUTPUTS / "manifest_foreign_clean_v2_full"))
    ap.add_argument("--unseen-manifest", default=str(MDD_OUTPUTS / "manifest_unseen_prompt_rp"))
    ap.add_argument("--noise-manifest", default=str(MDD_OUTPUTS / "manifest_kspon_stage1"))
    ap.add_argument("--ckpt", default=str(DATA_ROOT / "checkpoints/w2vbert_stage2/step_38000"))
    ap.add_argument("--exclude-prior", nargs="*", default=[])
    ap.add_argument("--train-n", type=int, default=12000)
    ap.add_argument("--noise-n", type=int, default=2000)
    ap.add_argument("--valid-per-cohort", type=int, default=150)
    ap.add_argument("--test-per-cohort", type=int, default=300)
    ap.add_argument("--unseen-n", type=int, default=32)
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[200, 400, 600])
    ap.add_argument("--batch-frames", type=int, default=512000)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--updates", type=int, default=0, help="Stop at this update for a smoke or matched control; retain full-pilot LR schedule")
    args = ap.parse_args()
    if min(args.train_n, args.noise_n, args.valid_per_cohort, args.test_per_cohort,
           args.unseen_n, args.batch_frames, args.accum, *args.checkpoints) <= 0 or args.lr <= 0:
        ap.error("Counts/rate must be positive")
    prepare(args) if args.action == "prepare" else train(args)


if __name__ == "__main__":
    main()

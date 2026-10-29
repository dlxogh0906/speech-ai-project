"""Isolated paired CTC versus reference-guided error-preservation pilot."""

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
import shutil
import time
from types import SimpleNamespace

import numpy as np

import train_noise_pilot as prior
from shared.paths import MDD_OUTPUTS
from mdd.noise_augment import NoiseAugmenter, read_pcm
from mdd.phone_consistency import actual_error_mask, reference_anchors, sparse_pair_losses
from train_w2vbert_ctc import batches_by_frames, tri_stage


def prepare(args):
    """Freeze protocol and exclude every previously observed evaluation clip."""
    import compare_noise_preprocess as prep
    out = _pl.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise RuntimeError("Refusing to overwrite frozen protocol")
    base = json.loads(_pl.Path(args.previous_plan).read_text())
    exclusions = list(args.exclude_prior)
    # Adapt prior plan selections into the existing path-exclusion reader.
    for split in ("valid", "test"):
        p = out / ("prior_" + split + "_exclusions.json")
        prior.write_json(p, {"split": split, "args": {"split": split},
                            "source_row_indices": [i for i, _ in base[split + "_selection"]]})
        exclusions.append(str(p))
    # Reconstruct the original 2026-10-08 sample from its recorded seed/count.
    if args.exclude_robustness:
        _, rows = prep.read_rows(base["manifest"], "test")
        eligible = [i for i, r in enumerate(rows) if 16000 <= int(r[1]) <= 160000]
        p = out / "robustness_exclusions.json"
        prior.write_json(p, {"split": "test", "args": {"split": "test"},
                            "source_row_indices": sorted(random.Random(7).sample(eligible, 1200)),
                            "reconstructed_from": "noise_exp.py seed=7 n=1200 test length 1..10s"})
        exclusions.append(str(p))
    cfg = SimpleNamespace(out=str(out), ckpt=base["ckpt"], train_manifest=base["train_manifest"],
        manifest=base["manifest"], unseen_manifest=base["unseen_manifest"],
        noise_manifest=base["noise_manifest"], exclude_prior=exclusions, seed=args.seed,
        train_n=args.train_n, valid_per_cohort=args.valid_per_cohort,
        test_per_cohort=args.test_per_cohort, unseen_n=1, noise_n=2000,
        checkpoints=args.checkpoints, batch_frames=args.batch_frames, accum=2, lr=5e-6)
    prior.prepare(cfg)
    plan = json.loads((out / "plan.json").read_text())
    plan["experiment"] = "paired_ctc_vs_reference_guided_phone_preservation"
    plan["eval_conditions"] = ["clean", "babble20", "babble10", "babble5", "rumble0", "reverb_babble5"]
    plan["unseen_scope"] = "One RP clip reserved by shared preparation; not evaluated or used for selection in this pilot."
    plan["training"].update(noise_probability=1.0, paired_inputs=True,
        precision="CUDA BF16 autocast with FP32 master weights and CTC; CPU FP32; no loss scaling or skipped updates",
        clean_noisy_ctc="0.5 * (clean CTC + noisy CTC)",
        error_ce_weight=0.5, consistency_kl_weight=0.25,
        consistency_error_weight=4.0, teacher_correct_confidence=0.6,
        reference_timing="Frozen original model CTC forced alignment to human actual .ltr; one probability peak per token",
        hard_supervision="Actual substitution/insertion tokens on clean and noisy peaks; deleted canonical tokens handled by sequence CTC",
        consistency="KL only where clean frozen teacher top1 equals human actual token with p>=0.6; incorrect teacher predictions excluded",
        matched_control="Same paired clean/noisy inputs, seed, dropout, order, optimizer and updates; auxiliary weights zero")
    plan["selection_policy"].update(
        objective="maximize mean babble10/5 API recall subject to actual preservation and clean/FPR safety; valid only",
        primary_frr_increase_max_pp=0.0,
        control="paired CTC continuation, same successful updates as chosen candidate",
        supports_use="positive API recall and actual preservation gains; no primary FPR increase")
    plan["prior_evaluation_exclusions"] = exclusions
    plan["production_hashes"] = {str(p.relative_to(_root)): prior.digest(p) for p in (
        _root / "deploy/pronunciation-demo/app.py",
        _root / "tasks/pronunciation-assessment/src/mdd/speechsuper.py")}
    plan["fixed_asset_hashes"] = {str(p): prior.digest(p) for p in (
        _pl.Path(plan["norms"]), _pl.Path(plan["calib"]),
        *sorted(_pl.Path(plan["ckpt"]).glob("*.safetensors")))}
    sources = [_pl.Path(__file__).resolve(), _root / "tasks/pronunciation-assessment/src/mdd/phone_consistency.py",
               _root / "tasks/pronunciation-assessment/scripts/eval/eval_phone_consistency.py",
               _pl.Path(prior.__file__).resolve(),
               _root / "tasks/pronunciation-assessment/scripts/train/run_phone_consistency.sh",
               _root / "tasks/pronunciation-assessment/scripts/eval/eval_noise_training.py",
               _root / "tasks/pronunciation-assessment/scripts/eval/compare_noise_mixing.py",
               _root / "tasks/pronunciation-assessment/scripts/eval/compare_noise_preprocess.py",
               _root / "tasks/pronunciation-assessment/scripts/eval/summarize_noise_preprocess.py",
               _root / "tasks/pronunciation-assessment/src/mdd/gop_feats_v2.py",
               _root / "tasks/pronunciation-assessment/src/mdd/vad.py"]
    plan["source_hashes"].update({str(p.relative_to(_root)): prior.digest(p) for p in sources})
    prior.write_json(out / "plan.json", plan)
    snapshots = out / "source_snapshots"
    for rel in plan["source_hashes"]:
        dst = snapshots / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_root / rel, dst)
    print("[protocol frozen] " + prior.digest(out / "plan.json"), flush=True)


def training_inputs(plan):
    items, tokens = prior.selected_data(_pl.Path(plan["train_manifest"]), "train", plan["train_indices"])
    vi = json.loads((_pl.Path(plan["ckpt"]) / "jamo_vocab.json").read_text())
    targets = [[vi["vocab"][t] for t in seq] for seq in tokens]
    if any(not t for t in targets):
        raise ValueError("Empty training target")
    return items, tokens, targets, vi


def anchors(args):
    import torch
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    out = _pl.Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    cache = out / "reference_anchors.pt"
    if cache.exists():
        raise RuntimeError("Refusing to overwrite reference targets")
    items, tokens, targets, vi = training_inputs(plan)
    wanted = set(plan["train_indices"])
    canons = {}
    with (_pl.Path(plan["train_manifest"]) / "train.canon").open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                canons[i] = line.split()
    errors, deletions = [], 0
    for i, actual in zip(plan["train_indices"], tokens):
        flags, missing = actual_error_mask(canons[i], actual)
        errors.append(flags)
        deletions += missing
    decoder = HFDecoder(plan["ckpt"])
    torch.set_num_threads(4)
    batches = batches_by_frames(items, plan["training"]["batch_frames"])
    records = [None] * len(items)
    started = time.monotonic()
    for k, indices in enumerate(batches):
        waves = [read_pcm(items[i][0]) for i in indices]
        if any(len(x) != items[i][1] for i, x in zip(indices, waves)):
            raise RuntimeError("Manifest duration mismatch")
        lp, lengths = forward_lprobs(decoder, waves)
        for j, i in enumerate(indices):
            records[i] = reference_anchors(lp[j, :int(lengths[j])].numpy(), targets[i], errors[i],
                plan["training"]["teacher_correct_confidence"], vi["blank"])
        if k == 0 or (k + 1) % 50 == 0 or k + 1 == len(batches):
            elapsed = time.monotonic() - started
            print("[reference anchors] %d/%d batches %.1fs eta=%.1fs" %
                  (k + 1, len(batches), elapsed, elapsed / (k + 1) * (len(batches) - k - 1)), flush=True)
    if any(a is None for a in records):
        raise RuntimeError("Incomplete reference cache")
    torch.save({"plan_sha256": prior.digest(out / "plan.json"), "anchors": records}, cache)
    prior.write_json(out / "anchor_audit.json", {
        "utterances": len(records), "tokens": sum(len(a["targets"]) for a in records),
        "actual_error_tokens": sum(int(a["errors"].sum()) for a in records),
        "canonical_deletions_ctc_only": deletions,
        "gated_consistency_tokens": sum(int(a["teacher_correct"].sum()) for a in records),
        "gated_error_consistency_tokens": sum(int((a["teacher_correct"] & a["errors"]).sum()) for a in records),
        "cache_sha256": prior.digest(cache), "elapsed_sec": time.monotonic() - started,
        "timing_is_model_derived_not_manual": True, "no_test_predictions_read": True})


def train(args):
    import torch
    from torch.cuda.amp import autocast, GradScaler
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC
    out = _pl.Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    settings = plan["training"]
    save = out / "checkpoints" / args.mode
    if save.exists():
        raise RuntimeError("Refusing to overwrite paired training")
    save.mkdir(parents=True)
    seed = plan["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This frozen protocol requires CUDA BF16 support")
    items, _, targets, vi = training_inputs(plan)
    data = torch.load(out / "reference_anchors.pt", map_location="cpu")
    if data["plan_sha256"] != prior.digest(out / "plan.json"):
        raise RuntimeError("Reference cache belongs to another protocol")
    refs = data["anchors"]
    model = Wav2Vec2BertForCTC.from_pretrained(plan["ckpt"]).to(device)
    model.config.ctc_zero_infinity = True
    model.config.apply_spec_augment = False
    model.train()
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(plan["ckpt"])
    augmenter = NoiseAugmenter(prior.noise_paths(plan, "train"), probability=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.98), eps=1e-8, weight_decay=0)
    scaler = GradScaler(enabled=False)

    class Batches(torch.utils.data.Dataset):
        def __init__(self, epoch):
            self.epoch = epoch
            self.batches = batches_by_frames(items, settings["batch_frames"], seed + epoch)
        def __len__(self):
            return len(self.batches)
        def __getitem__(self, k):
            indices = self.batches[k]
            clean, noisy, names = [], [], []
            for i in indices:
                x = read_pcm(items[i][0])
                if len(x) != items[i][1]:
                    raise RuntimeError("Manifest duration mismatch")
                y, name = augmenter(x, seed * 1000003 + self.epoch * 100003 + i)
                clean.append(x)
                noisy.append(y)
                names.append(name)
            features = extractor(clean + noisy, sampling_rate=16000, return_tensors="pt", padding=True)
            lengths = features["attention_mask"].sum(-1).tolist()
            for j, i in enumerate(indices):
                if lengths[j] != refs[i]["output_length"] or lengths[j + len(indices)] != lengths[j]:
                    raise RuntimeError("Augmentation changed reference timing")
            ys = [targets[i] for i in indices] * 2
            labels = torch.full((len(ys), max(map(len, ys))), -100, dtype=torch.long)
            for j, y in enumerate(ys):
                labels[j, :len(y)] = torch.tensor(y)
            return features["input_features"], features["attention_mask"], labels, indices, names

    total = args.updates or settings["max_updates"]
    checkpoints = {total} if args.updates else set(plan["checkpoints"])
    started = time.monotonic()
    step = micro = skipped = epoch = 0
    counts = collections.Counter()
    exposure = []
    history = []
    # Verify trainability with small parameter slices, without copying weights.
    probes = {n: p.detach().flatten()[:32].cpu().clone() for n, p in model.named_parameters()
              if n == "lm_head.weight" or ("encoder.layers." in n and n.endswith("self_attn.linear_q.weight"))}
    if len(probes) != model.config.num_hidden_layers + 1:
        raise RuntimeError("Encoder/head trainability probes incomplete")
    with (save / "train.jsonl").open("w", encoding="utf-8") as log:
        while step < total:
            kw = {"num_workers": args.workers, "batch_size": None, "pin_memory": device.type == "cuda"}
            if args.workers:
                kw["prefetch_factor"] = 2
            for feats, mask, labels, indices, names in torch.utils.data.DataLoader(Batches(epoch), **kw):
                counts.update(names)
                exposure.extend(indices)
                feats = feats.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
                    output = model(input_features=feats, attention_mask=mask, labels=labels)
                    ctc = output.loss
                    if args.mode == "targeted":
                        b = len(indices)
                        ce, kl = sparse_pair_losses(output.logits[:b], output.logits[b:],
                            [refs[i] for i in indices], settings["consistency_error_weight"])
                        loss = ctc + settings["error_ce_weight"] * ce + settings["consistency_kl_weight"] * kl
                    else:
                        ce = kl = ctc.detach() * 0
                        loss = ctc
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite paired training loss")
                history.append((float(ctc.detach()), float(ce.detach()), float(kl.detach()), float(loss.detach())))
                scaler.scale(loss / settings["accum"]).backward()
                micro += 1
                if micro % settings["accum"]:
                    continue
                lr = tri_stage(step, settings["max_updates"], settings["lr"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not torch.isfinite(grad):
                    raise RuntimeError("Nonfinite gradient; refusing unmatched updates")
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
                if step == 1 or step % 20 == 0 or step == total:
                    elapsed = time.monotonic() - started
                    mean = np.mean(history[-40:], axis=0)
                    record = {"mode": args.mode, "step": step, "ctc": mean[0], "error_ce": mean[1],
                        "consistency_kl": mean[2], "total_loss": mean[3], "lr": lr, "grad_norm": float(grad),
                        "skipped_updates": skipped, "elapsed_sec": round(elapsed, 1),
                        "eta_sec": round(elapsed / step * (total - step), 1)}
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if step in checkpoints:
                    p = save / ("step_%d" % step)
                    model.save_pretrained(p)
                    extractor.save_pretrained(p)
                    prior.write_json(p / "jamo_vocab.json", vi)
                    prior.write_json(p / "experiment.json", {"mode": args.mode, "step": step,
                        "plan_sha256": prior.digest(out / "plan.json"), "actual_ltr_targets": True,
                        "paired_examples_seen": len(exposure), "micro_batches": micro,
                        "skipped_updates": skipped,
                        "exposure_sha256": hashlib.sha256(np.asarray(exposure, dtype=np.int64).tobytes()).hexdigest()})
                    print("[saved] %s %d" % (args.mode, step), flush=True)
                if step == total:
                    break
            epoch += 1
    changed = {n: bool(not torch.equal(p.detach().flatten()[:32].cpu(), probes[n]))
               for n, p in model.named_parameters() if n in probes}
    if not all(changed.values()):
        raise RuntimeError("Some encoder/head probes did not update")
    prior.write_json(save / "complete.json", {"mode": args.mode, "step": step,
        "elapsed_sec": time.monotonic() - started, "skipped_updates": skipped,
        "paired_examples_seen": len(exposure), "counts": dict(counts),
        "exposure_sha256": hashlib.sha256(np.asarray(exposure, dtype=np.int64).tobytes()).hexdigest(),
        "encoder_layers_and_head_updated": changed,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "anchors", "train"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--previous-plan", default=str(MDD_OUTPUTS / "analysis/noise_training_2026-10-09/pilot/plan.json"))
    ap.add_argument("--exclude-prior", nargs="*", default=[])
    ap.add_argument("--exclude-robustness", action="store_true")
    ap.add_argument("--train-n", type=int, default=12000)
    ap.add_argument("--valid-per-cohort", type=int, default=150)
    ap.add_argument("--test-per-cohort", type=int, default=400)
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[200, 400, 600])
    ap.add_argument("--batch-frames", type=int, default=512000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--mode", choices=("targeted", "paired_control"), default="targeted")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--updates", type=int, default=0)
    args = ap.parse_args()
    if min(args.train_n, args.valid_per_cohort, args.test_per_cohort, args.batch_frames, *args.checkpoints) <= 0:
        ap.error("Counts must be positive")
    {"prepare": prepare, "anchors": anchors, "train": train}[args.action](args)


if __name__ == "__main__":
    main()

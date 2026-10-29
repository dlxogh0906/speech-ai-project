"""Matched ㅆ-only ranking pilot with independently weighted normal preservation.

Both arms start from the ORIGINAL checkpoint and use identical train data,
batch order, optimizer, 300 updates, and error rank weight 1. Only normal rank
weight differs (1 versus 4). No hard-negative mining or validation selection.
Mixed-error utterances retain the complete human TAG transcript for CTC.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"), str(Path(__file__).parent)]

import argparse
import collections
import hashlib
import json
import math
import random
import time

import numpy as np

# These frozen pilot helpers remain byte-for-byte unchanged. In particular,
# CTC is FP32, zero_infinity=False, and both rank branches receive gradients.
import train_sibilant_rank_pilot as core
from train_sibilant_rank_pilot import counterfactual_loss, digest, write_json

VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def select_training_sites(rows, vocab, blank):
    """Validate full transcripts, selecting only unambiguous onset ㅆ sites.

    Canonical and actual indices are separate: insertions/deletions elsewhere
    must not shift a rank candidate onto the wrong actual token.
    """
    targets, errors, normals = [], [], []
    counts = collections.Counter()
    excluded = collections.Counter()
    by_role = collections.defaultdict(collections.Counter)
    if not rows or any(r.get("source_split") != "train" for r in rows):
        raise ValueError("Training must contain official train only")
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training utterance")
    for row in rows:
        actual, canonical = row.get("actual"), row.get("canonical")
        if not actual or not canonical or any(not isinstance(t, str) or len(t) != 1
                                             for t in actual + canonical):
            raise ValueError("Missing or invalid human/canonical transcript: " + row["uid"])
        try:
            target = [vocab[t] for t in actual]
        except KeyError as ex:
            raise ValueError("Human transcript token absent from checkpoint vocabulary") from ex
        if blank in target:
            raise ValueError("Blank token in human transcript")
        if not isinstance(row.get("frames"), int) or row["frames"] <= 0:
            raise ValueError("Invalid audio frames: " + row["uid"])
        ep, np_, qs, actual_indices = [], [], set(), set()
        role = row.get("role", "unspecified")
        for site in row.get("excluded_sites", []):
            excluded[site.get("exclusion_reason") or site.get("label_kind") or "unspecified"] += 1
        for site in row.get("sites", []):
            q, a = site.get("q"), site.get("a")
            if not isinstance(q, int) or not 0 <= q < len(canonical) or canonical[q] != site.get("canon"):
                raise ValueError("Canonical site index/label mismatch: " + row["uid"])
            if q in qs:
                raise ValueError("Duplicate canonical site: " + row["uid"])
            qs.add(q)
            if site.get("alignment_ambiguous"):
                reason = "skip_ambiguous"
            elif a is None:
                if site.get("actual") is not None:
                    raise ValueError("Unambiguous actual token has no index: " + row["uid"])
                reason = "skip_deletion"
            else:
                if not isinstance(a, int) or not 0 <= a < len(actual) or actual[a] != site.get("actual"):
                    raise ValueError("Actual site index/label mismatch: " + row["uid"])
                if a in actual_indices:
                    raise ValueError("Duplicate unambiguous actual site: " + row["uid"])
                actual_indices.add(a)
                if site["canon"] != "ㅆ":
                    reason = "skip_other_canonical"
                elif q + 1 >= len(canonical) or canonical[q + 1] not in VOWELS:
                    reason = "skip_canonical_not_onset"
                elif q > 0 and canonical[q - 1] not in VOWELS:
                    reason = "skip_post_coda"
                elif a + 1 >= len(actual) or actual[a + 1] not in VOWELS:
                    reason = "skip_actual_not_onset"
                elif site["actual"] == "ㅅ":
                    reason = "error_rank"
                    ep.append((a, vocab["ㅆ"]))
                elif site["actual"] == "ㅆ":
                    reason = "normal_rank"
                    np_.append((a, vocab["ㅅ"]))
                else:
                    reason = "skip_other_actual"
            counts[reason] += 1
            by_role[role][reason] += 1
        counts["error_rank_utterances"] += bool(ep)
        counts["normal_rank_utterances"] += bool(np_)
        counts["both_rank_utterances"] += bool(ep) and bool(np_)
        counts["ctc_only_utterances"] += not ep and not np_
        counts["nonidentical_full_transcript_utterances"] += actual != canonical
        targets.append(target)
        errors.append(ep)
        normals.append(np_)
    return targets, errors, normals, {"counts": dict(counts), "by_role": {k: dict(v) for k, v in by_role.items()},
                                     "dataset_excluded_sites": dict(excluded)}


def weighted_loss(lp, lengths, targets, errors, normals, blank, margin, normal_weight):
    """Independent per-batch means: adding normals cannot dilute error rank."""
    ctc, error_rank, em = counterfactual_loss(lp, lengths, targets, errors, blank, margin)
    _, normal_rank, nm = counterfactual_loss(lp, lengths, targets, normals, blank, margin)
    return ctc + error_rank + normal_weight * normal_rank, ctc, error_rank, normal_rank, em, nm


def load_training(args):
    data = Path(args.data) / "train.jsonl"
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    ckpt = Path(args.ckpt)
    vi = json.loads((ckpt / "jamo_vocab.json").read_text(encoding="utf-8"))
    targets, errors, normals, selection = select_training_sites(rows, vi["vocab"], vi["blank"])
    if not sum(map(len, errors)) or not sum(map(len, normals)):
        raise ValueError("Both actual ㅆ→ㅅ errors and correct onset ㅆ examples are required")
    return data, rows, ckpt, vi, targets, errors, normals, selection


def train(args):
    import torch
    from torch.cuda.amp import autocast
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC
    from mdd.noise_augment import read_pcm
    from train_w2vbert_ctc import batches_by_frames

    data, rows, ckpt, vi, targets, errors, normals, selection = load_training(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    items = [(r["wav"], r["frames"]) for r in rows]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA experiment requires BF16")
    model = Wav2Vec2BertForCTC.from_pretrained(str(ckpt)).to(device)
    if model.config.pad_token_id != vi["blank"]:
        raise ValueError("Checkpoint blank and vocabulary disagree")
    model.config.apply_spec_augment = False
    model.config.layerdrop = 0.0
    for name in ("activation_dropout", "attention_dropout", "conformer_conv_dropout",
                 "feat_proj_dropout", "final_dropout", "hidden_dropout"):
        setattr(model.config, name, 0.0)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.requires_grad_(True)
    model.train()
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(str(ckpt))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0)
    plan = {"settings": vars(args), "train_sha256": digest(data),
            "initial_weights_sha256": digest(ckpt / "model.safetensors"),
            "initial_vocab_sha256": digest(ckpt / "jamo_vocab.json"),
            "source_sha256": digest(__file__), "frozen_core_sha256": digest(core.__file__),
            "train_utterances": len(rows), "roles": dict(collections.Counter(r.get("role", "unspecified") for r in rows)),
            "site_selection": selection, "rank_error_sites": sum(map(len, errors)),
            "rank_normal_sites": sum(map(len, normals)), "rank_error_weight": 1.0,
            "rank_normal_weight": args.normal_rank_weight,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "precision": "FP32 master weights, CUDA BF16 forward, FP32 logsoftmax and CTC",
            "all_dropout_and_layerdrop_zero": True, "spec_augment": False,
            "loss": "mean_batch(NLLactual/L) + mean_batch(mean_error_sites(hinge/L)) + normal_weight*mean_batch(mean_normal_sites(hinge/L)); hinge=relu(margin+NLLactual-NLLcf)",
            "candidate": "Change exactly one eligible actual token; errors s→ss, correct ss→s. Onset required on BOTH canonical and actual axes; canonical q==0 or previous vowel (conservative deployed context). Post-coda contributes full CTC only.",
            "full_actual_CTC": True, "hard_negative_mining": False,
            "no_calibration_or_test_used_for_training": True,
            "comparison": "New same-data ss-only weight1 versus weight4; previous bidirectional weight1 is not the matched control."}
    write_json(out / "plan.json", plan)
    (out / Path(__file__).name).write_text(Path(__file__).read_text(), encoding="utf-8")
    probes = {n: p.detach().flatten()[:64].cpu().clone() for n, p in model.named_parameters()
              if n == "lm_head.weight" or ("encoder.layers." in n and n.endswith("self_attn.linear_q.weight"))}
    if len(probes) != model.config.num_hidden_layers + 1:
        raise RuntimeError("Incomplete encoder probes")

    class Batches(torch.utils.data.Dataset):
        def __init__(self, epoch):
            self.batches = batches_by_frames(items, args.batch_frames, args.seed + epoch)

        def __len__(self):
            return len(self.batches)

        def __getitem__(self, index):
            indices = self.batches[index]
            xs = [read_pcm(items[i][0]) for i in indices]
            if any(len(x) != items[i][1] for x, i in zip(xs, indices)):
                raise RuntimeError("Audio duration differs from frozen manifest")
            f = extractor(xs, sampling_rate=16000, return_tensors="pt", padding=True)
            return f["input_features"], f["attention_mask"], indices

    start = time.monotonic()
    epoch = step = micro = 0
    exposure, history, gradients = [], [], []
    with (out / "train.jsonl").open("w", encoding="utf-8") as log, \
            (out / "batches.jsonl").open("w", encoding="utf-8") as batchlog, \
            (out / "gradients.jsonl").open("w", encoding="utf-8") as gradlog:
        while step < args.steps:
            kw = {"batch_size": None, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
            if args.workers:
                kw["prefetch_factor"] = 2
            for feats, mask, indices in torch.utils.data.DataLoader(Batches(epoch), **kw):
                feats, mask = feats.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                ys = [targets[i] for i in indices]
                ep = [(j, q, phone) for j, i in enumerate(indices) for q, phone in errors[i]]
                np_ = [(j, q, phone) for j, i in enumerate(indices) for q, phone in normals[i]]
                with autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
                    logits = model(input_features=feats, attention_mask=mask).logits
                lp = logits.float().log_softmax(-1)
                lengths = model._get_feat_extract_output_lengths(mask.sum(-1)).to(torch.long)
                loss, ctc, error_rank, normal_rank, em, nm = weighted_loss(
                    lp, lengths, ys, ep, np_, vi["blank"], args.margin, args.normal_rank_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                (loss / args.accum).backward()
                exposure.extend(indices)
                micro += 1
                h = {"ctc": float(ctc.detach()), "error_rank": float(error_rank.detach()),
                     "normal_rank": float(normal_rank.detach()), "loss": float(loss.detach()),
                     "error_candidates": em["candidate_count"], "error_active": em["active_count"],
                     "normal_candidates": nm["candidate_count"], "normal_active": nm["active_count"]}
                history.append(h)
                batchlog.write(json.dumps({"micro_batch": micro, "epoch": epoch, "next_update": step + 1,
                                           "indices": [int(i) for i in indices],
                                           "uids": [rows[i]["uid"] for i in indices],
                                           "error_rank_sites": len(ep), "normal_rank_sites": len(np_)}) + "\n")
                if micro % args.accum:
                    continue
                step += 1
                if step <= args.warmup:
                    scale = step / max(1, args.warmup)
                else:
                    scale = 1 - 0.9 * (step - args.warmup) / max(1, args.steps - args.warmup)
                lr = args.lr * scale
                for group in optimizer.param_groups:
                    group["lr"] = lr
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(grad):
                    raise RuntimeError("Nonfinite gradient; no skipped steps allowed")
                gradients.append(float(grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gradlog.write(json.dumps({"step": step, "lr": lr, "grad_norm_before_clip": float(grad),
                                          "clipped": float(grad) > 5.0}) + "\n")
                gradlog.flush()
                batchlog.flush()
                if step == 1 or step % 20 == 0 or step == args.steps:
                    recent = history[-40:]
                    elapsed = time.monotonic() - start
                    rec = {"step": step, "normal_rank_weight": args.normal_rank_weight,
                           "lr": lr, "grad_norm": float(grad), "elapsed_sec": round(elapsed, 1),
                           "eta_sec": round(elapsed / step * (args.steps - step), 1)}
                    for key in ("ctc", "error_rank", "normal_rank", "loss"):
                        rec[key] = float(np.mean([h[key] for h in recent]))
                    for key in ("error_candidates", "error_active", "normal_candidates", "normal_active"):
                        rec["recent_" + key] = sum(h[key] for h in recent)
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                if step == args.steps:
                    break
            epoch += 1
    changed = {n: bool(not torch.equal(p.detach().flatten()[:64].cpu(), probes[n]))
               for n, p in model.named_parameters() if n in probes}
    if not all(changed.values()):
        raise RuntimeError("Encoder or head failed to update")
    final = out / ("step_%d" % args.steps)
    model.save_pretrained(str(final))
    extractor.save_pretrained(str(final))
    write_json(final / "jamo_vocab.json", vi)
    counts = collections.Counter(exposure)
    write_json(out / "utterance_exposure.json", [{"index": i, "uid": r["uid"], "role": r.get("role"),
                                                "count": counts[i], "error_sites": len(errors[i]),
                                                "normal_sites": len(normals[i])} for i, r in enumerate(rows)])
    complete = {"normal_rank_weight": args.normal_rank_weight, "steps": step, "micro_batches": micro,
                "seen_utterances": len(exposure), "unique_seen": len(counts),
                "exposure_sha256": hashlib.sha256(np.asarray(exposure, dtype=np.int64).tobytes()).hexdigest(),
                "batches_sha256": digest(out / "batches.jsonl"),
                "seen_roles": dict(collections.Counter(rows[i].get("role", "unspecified") for i in exposure)),
                "seen_error_rank_sites": sum(len(errors[i]) for i in exposure),
                "seen_normal_rank_sites": sum(len(normals[i]) for i in exposure),
                "encoder_and_head_changed": changed, "elapsed_sec": time.monotonic() - start,
                "gradient_updates": len(gradients), "gradient_norm_min": min(gradients),
                "gradient_norm_max": max(gradients), "gradient_clipped_updates": sum(g > 5 for g in gradients),
                "max_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "checkpoint": str(final), "checkpoint_weights_sha256": digest(final / "model.safetensors"),
                "plan_sha256": digest(out / "plan.json")}
    write_json(out / "complete.json", complete)
    print(json.dumps({k: v for k, v in complete.items() if k != "encoder_and_head_changed"}), flush=True)


def self_test():
    import copy
    import torch
    torch.set_num_threads(2)
    core.self_test()
    vocab = {"<blank>": 0, "ㅅ": 1, "ㅆ": 2, "ㅏ": 3, "ㄱ": 4, "ㅓ": 5}
    # A deletion elsewhere makes q != a. Two target sites include one error and
    # one correct ㅆ in the SAME otherwise-incorrect utterance.
    row = {"uid": "mixed", "source_split": "train", "role": "mixed", "frames": 16000,
           "canonical": ["ㄱ", "ㅏ", "ㅆ", "ㅏ", "ㅆ", "ㅓ"],
           "actual": ["ㅏ", "ㅅ", "ㅏ", "ㅆ", "ㅓ"],
           "sites": [{"q": 2, "a": 1, "canon": "ㅆ", "actual": "ㅅ"},
                     {"q": 4, "a": 3, "canon": "ㅆ", "actual": "ㅆ"}]}
    ys, er, no, stats = select_training_sites([row], vocab, 0)
    assert ys == [[3, 1, 3, 2, 5]] and er == [[(1, 2)]] and no == [[(3, 1)]]
    assert stats["counts"]["both_rank_utterances"] == 1
    for field, value in (("actual", "ㄱ"), ("a", 99), ("q", 1)):
        bad = copy.deepcopy(row)
        bad["sites"][0][field] = value
        try:
            select_training_sites([bad], vocab, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Inconsistent unambiguous site accepted")
    reverse = {"uid": "reverse", "source_split": "train", "role": "replay", "frames": 16000,
               "canonical": ["ㅅ", "ㅏ"], "actual": ["ㅆ", "ㅏ"],
               "sites": [{"q": 0, "a": 0, "canon": "ㅅ", "actual": "ㅆ"}]}
    _, er, no, stats = select_training_sites([reverse], vocab, 0)
    assert er == [[]] and no == [[]] and stats["counts"]["skip_other_canonical"] == 1
    missing = copy.deepcopy(row)
    missing["sites"][0].update(a=None, actual=None, alignment_ambiguous=True)
    missing["sites"][1].update(a=None, actual=None)
    ys, er, no, stats = select_training_sites([missing], vocab, 0)
    assert ys[0] == [vocab[t] for t in row["actual"]] and not er[0] and not no[0]
    assert stats["counts"]["skip_ambiguous"] == stats["counts"]["skip_deletion"] == 1
    no_vowel = copy.deepcopy(row)
    no_vowel["actual"][2] = "ㄱ"
    _, er, no, stats = select_training_sites([no_vowel], vocab, 0)
    assert er == [[]] and no == [[(3, 1)]] and stats["counts"]["skip_actual_not_onset"] == 1
    post_coda = {"uid": "post_coda", "source_split": "train", "role": "error", "frames": 16000,
                 "canonical": ["ㄱ", "ㅆ", "ㅏ"], "actual": ["ㄱ", "ㅅ", "ㅏ"],
                 "sites": [{"q": 1, "a": 1, "canon": "ㅆ", "actual": "ㅅ"}]}
    ys, er, no, stats = select_training_sites([post_coda], vocab, 0)
    assert ys == [[4, 1, 3]] and er == [[]] and no == [[]] and stats["counts"]["skip_post_coda"] == 1
    # Increasing normal weight must leave error-example gradient identical;
    # after removing CTC its normal-example gradient must be exactly fourfold.
    grads, cgrads = [], []
    for weight in (1.0, 4.0):
        z = torch.tensor([[[0.0, -1.0, 1.0]], [[0.0, 1.0, -1.0]]], requires_grad=True)
        loss, ctc, erank, nrank, _, _ = weighted_loss(z.log_softmax(-1), torch.tensor([1, 1]),
                                                      [[1], [2]], [(0, 0, 2)], [(1, 0, 1)], 0, 2.0, weight)
        cgrad = torch.autograd.grad(ctc, z, retain_graph=True)[0]
        grad = torch.autograd.grad(loss, z)[0]
        grads.append(grad)
        cgrads.append(cgrad)
        assert torch.allclose(erank, torch.tensor(2.0)) and torch.allclose(nrank, torch.tensor(2.0))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-6)
    assert torch.allclose(grads[1][1] - cgrads[1][1], 4 * (grads[0][1] - cgrads[0][1]), atol=1e-6)
    print("Mixed transcripts, distinct q/a, ss-only selection, skipped ambiguity/deletion/onset, label rejection, independent weight gradients: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-frames", type=int, default=384000)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--normal-rank-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.data or not args.ckpt or (not args.validate_only and not args.out):
        ap.error("--data, --ckpt, and (unless validate-only) --out are required")
    positive = (args.steps, args.batch_frames, args.accum, args.lr, args.margin, args.normal_rank_weight, args.threads)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or args.warmup < 0 or args.workers < 0:
        ap.error("Training counts/weights must be positive finite values")
    if args.validate_only:
        data, rows, _ck, _vi, _ys, er, no, stats = load_training(args)
        print(json.dumps({"train_sha256": digest(data), "utterances": len(rows),
                          "rank_error_sites": sum(map(len, er)), "rank_normal_sites": sum(map(len, no)),
                          "selection": stats}, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()

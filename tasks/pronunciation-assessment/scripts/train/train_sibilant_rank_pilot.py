"""Matched ordinary-CTC / one-phone counterfactual-CTC encoder pilot.

Uses only train.jsonl. Evaluation and production assets are never modified.
Default counterfactual restores one erroneous sibilant in the human sequence.
With --include-normal-rank, correct sibilants also compete against their flip.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
sys.path[:0] = [str(_root), str(_root / "tasks/pronunciation-assessment/src")]

import argparse
import collections
import hashlib
import json
import random
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sequence_nll(log_probs, lengths, sequences, blank):
    """Unnormalized differentiable CTC NLL. Impossible paths are errors."""
    import torch
    target_lengths = torch.tensor([len(y) for y in sequences], device=log_probs.device)
    minimum = [len(y) + sum(a == b for a, b in zip(y, y[1:])) for y in sequences]
    if any(n > int(t) for n, t in zip(minimum, lengths)):
        raise ValueError("CTC target has no possible path")
    flat = torch.tensor([t for y in sequences for t in y], device=log_probs.device)
    with torch.backends.cudnn.flags(enabled=False):
        nll = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1), flat, lengths, target_lengths,
            blank=blank, reduction="none", zero_infinity=False)
    if not torch.isfinite(nll).all():
        raise RuntimeError("Nonfinite CTC NLL; refusing to mask impossible targets")
    return nll


def counterfactual_loss(log_probs, lengths, targets, candidates, blank, margin):
    """Mean CTC plus per-utterance rank, both normalized by target length.

    Candidates are (batch index, actual-token index, competing token ID).
Rank = relu(margin + NLL(actual) - NLL(counterfactual)); neither branch detaches.
Zero-error utterances contribute zero rank loss, retaining replay's weight.
"""
    import torch
    actual = sequence_nll(log_probs, lengths, targets, blank)
    sizes = torch.tensor([len(y) for y in targets], device=log_probs.device, dtype=actual.dtype)
    ctc = (actual / sizes).mean()
    if not candidates:
        return ctc, log_probs.sum() * 0, {"candidate_count": 0, "active_count": 0, "mean_gap": None}
    indices, cfs = [], []
    for b, q, phone in candidates:
        y = list(targets[b])
        if y[q] == phone:
            raise ValueError("Counterfactual did not change the actual phone")
        y[q] = phone
        indices.append(b)
        cfs.append(y)
    take = torch.tensor(indices, device=log_probs.device, dtype=torch.long)
    counter = sequence_nll(log_probs.index_select(0, take), lengths.index_select(0, take), cfs, blank)
    # Positive gap means actual is more likely than the restored canonical.
    gap = counter - actual.index_select(0, take)
    hinges = torch.relu(margin - gap) / sizes.index_select(0, take)
    sums = torch.zeros_like(actual).scatter_add(0, take, hinges)
    counts = torch.zeros_like(actual).scatter_add(0, take, torch.ones_like(hinges))
    rank = (sums / counts.clamp(min=1)).mean()
    return ctc, rank, {"candidate_count": len(candidates),
                       "active_count": int((gap.detach() < margin).sum()),
                       "mean_gap": float(gap.detach().mean())}


def self_test():
    import torch
    torch.set_num_threads(2)
    # One frame has one valid CTC path: analytical gradient is exact.
    z = torch.tensor([[[0.0, -1.0, 1.0]]], requires_grad=True)
    lp = z.log_softmax(-1)
    ctc, rank, meta = counterfactual_loss(lp, torch.tensor([1]), [[1]], [(0, 0, 2)], 0, 2.0)
    expected = -lp[0, 0, 1]
    assert torch.allclose(ctc, expected)
    assert torch.allclose(rank, torch.tensor(4.0))
    rank.backward()
    assert torch.allclose(z.grad, torch.tensor([[[0.0, -1.0, 1.0]]]), atol=1e-6), z.grad
    z2 = torch.tensor([[[0.0, 5.0, -5.0]]], requires_grad=True)
    _, rank2, _ = counterfactual_loss(z2.log_softmax(-1), torch.tensor([1]), [[1]], [(0, 0, 2)], 0, 2.0)
    rank2.backward()
    assert float(rank2) == 0 and bool((z2.grad == 0).all())
    # A repeat needs an intervening blank, even if target lengths match.
    try:
        sequence_nll(torch.randn(1, 2, 3).log_softmax(-1), torch.tensor([2]), [[1, 1]], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Impossible CTC path not rejected")
    z3 = torch.tensor([[[0.0, -1.0, 1.0]], [[0.0, 1.0, -1.0]]])
    _, rank3, _ = counterfactual_loss(z3.log_softmax(-1), torch.tensor([1, 1]),
                                     [[1], [1]], [(0, 0, 2)], 0, 2.0)
    assert torch.allclose(rank3, torch.tensor(2.0)), rank3
    print("CTC equivalence, rank sign/gradient, inactive hinge, replay weighting, impossible paths: PASS")


def train(args):
    import torch
    from torch.cuda.amp import autocast
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC
    from mdd.noise_augment import read_pcm
    from train_w2vbert_ctc import batches_by_frames

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = Path(args.data) / "train.jsonl"
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    if not rows or any(r["source_split"] != "train" for r in rows):
        raise ValueError("Training must contain official train only")
    if len(set(r["uid"] for r in rows)) != len(rows):
        raise ValueError("Duplicate training utterance")
    ckpt = Path(args.ckpt)
    vi = json.loads((ckpt / "jamo_vocab.json").read_text(encoding="utf-8"))
    vocab, blank = vi["vocab"], vi["blank"]
    targets = [[vocab[t] for t in r["actual"]] for r in rows]
    substitutions, normal_substitutions = [], []
    error_sites = error_utterances = 0
    for r, target in zip(rows, targets):
        if not target or blank in target:
            raise ValueError("Empty target or blank in human transcript")
        pairs, normal_pairs = [], []
        utterance_errors = 0
        for site in r["sites"]:
            if r["actual"][site["a"]] != site["actual"] or r["canonical"][site["q"]] != site["canon"]:
                raise ValueError("Site label disagrees with utterance transcript")
            if site["actual"] != site["canon"]:
                if set((site["actual"], site["canon"])) != {"ㅅ", "ㅆ"}:
                    raise ValueError("Non-sibilant candidate")
                pairs.append((site["a"], vocab[site["canon"]]))
                utterance_errors += 1
            elif args.include_normal_rank:
                if site["actual"] not in ("ㅅ", "ㅆ"):
                    raise ValueError("Non-sibilant normal candidate")
                opposite = "ㅆ" if site["actual"] == "ㅅ" else "ㅅ"
                normal_pairs.append((site["a"], vocab[opposite]))
        substitutions.append(pairs)
        normal_substitutions.append(normal_pairs)
        error_sites += utterance_errors
        error_utterances += bool(utterance_errors)
    if not error_sites:
        raise ValueError("No actual sibilant error training examples")
    items = [(r["wav"], r["frames"]) for r in rows]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA experiment requires BF16")
    model = Wav2Vec2BertForCTC.from_pretrained(str(ckpt)).to(device)
    if model.config.pad_token_id != blank:
        raise ValueError("Checkpoint blank and vocabulary disagree")
    model.config.apply_spec_augment = False
    model.config.layerdrop = 0.0
    # Make actual and counterfactual compare identical logits, and keep the
    # matched continuation deterministic with all 24 encoder layers active.
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
    settings = vars(args).copy()
    plan = {"settings": settings, "train_sha256": digest(data),
            "initial_weights_sha256": digest(ckpt / "model.safetensors"),
            "source_sha256": digest(__file__), "train_utterances": len(rows),
            "error_utterances": error_utterances, "error_sites": error_sites,
            "rank_sites": sum(len(s) for s in substitutions),
            "normal_rank_sites": sum(len(s) for s in normal_substitutions),
            "roles": dict(collections.Counter(r["role"] for r in rows)),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "precision": "FP32 master weights, CUDA BF16 forward, FP32 logsoftmax and CTC",
            "all_dropout_and_layerdrop_zero": True, "spec_augment": False,
            "candidate": ("human actual sequence with exactly one normal OR erroneous sibilant flipped"
                          if args.include_normal_rank else
                          "human actual sequence with exactly one target error restored to canonical"),
            "loss": "mean(NLLactual/L) + weight*mean_utterance(mean_errors(hinge/L)) + optional weight*mean_utterance(mean_normals(hinge/L)); hinge=relu(m+NLLactual-NLLcf)",
            "no_calibration_or_test_used_for_training": True}
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
    exposure = []
    history = []
    with (out / "train.jsonl").open("w", encoding="utf-8") as log:
        while step < args.steps:
            kw = {"batch_size": None, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
            if args.workers:
                kw["prefetch_factor"] = 2
            for feats, mask, indices in torch.utils.data.DataLoader(Batches(epoch), **kw):
                feats, mask = feats.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                ys = [targets[i] for i in indices]
                candidates = [(j, q, phone) for j, i in enumerate(indices) for q, phone in substitutions[i]]
                with autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
                    logits = model(input_features=feats, attention_mask=mask).logits
                lp = logits.float().log_softmax(-1)
                lengths = model._get_feat_extract_output_lengths(mask.sum(-1)).to(torch.long)
                ctc, rank, meta = counterfactual_loss(lp, lengths, ys, candidates, blank, args.margin)
                if args.include_normal_rank:
                    normals = [(j, q, phone) for j, i in enumerate(indices) for q, phone in normal_substitutions[i]]
                    _, normal_rank, normal_meta = counterfactual_loss(lp, lengths, ys, normals, blank, args.margin)
                else:
                    normal_rank = rank * 0
                    normal_meta = {"candidate_count": 0, "active_count": 0}
                meta["normal_rank"] = float(normal_rank.detach())
                meta["normal_candidate_count"] = normal_meta["candidate_count"]
                meta["normal_active_count"] = normal_meta["active_count"]
                loss = ctc + (args.rank_weight * (rank + normal_rank) if args.mode == "rank" else rank * 0)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                (loss / args.accum).backward()
                exposure.extend(indices)
                micro += 1
                history.append((float(ctc.detach()), float(rank.detach()), float(loss.detach()), meta))
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
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 20 == 0 or step == args.steps:
                    recent = history[-40:]
                    means = np.mean([h[:3] for h in recent], axis=0)
                    elapsed = time.monotonic() - start
                    rec = {"mode": args.mode, "step": step, "ctc": float(means[0]),
                           "rank": float(means[1]), "loss": float(means[2]), "lr": lr,
                           "grad_norm": float(grad), "elapsed_sec": round(elapsed, 1),
                           "eta_sec": round(elapsed / step * (args.steps - step), 1),
                           "recent_candidates": sum(h[3]["candidate_count"] for h in recent),
                           "recent_active": sum(h[3]["active_count"] for h in recent),
                           "normal_rank": float(np.mean([h[3]["normal_rank"] for h in recent])),
                           "recent_normal_candidates": sum(h[3]["normal_candidate_count"] for h in recent),
                           "recent_normal_active": sum(h[3]["normal_active_count"] for h in recent)}
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
    complete = {"mode": args.mode, "steps": step, "micro_batches": micro,
                "seen_utterances": len(exposure), "unique_seen": len(set(exposure)),
                "exposure_sha256": hashlib.sha256(np.asarray(exposure, dtype=np.int64).tobytes()).hexdigest(),
                "encoder_and_head_changed": changed, "elapsed_sec": time.monotonic() - start,
                "max_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "checkpoint": str(final), "plan_sha256": digest(out / "plan.json")}
    write_json(out / "complete.json", complete)
    print(json.dumps({k: v for k, v in complete.items() if k != "encoder_and_head_changed"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=("control", "rank"), default="rank")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-frames", type=int, default=384000)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--include-normal-rank", action="store_true",
                    help="Also contrast correct sibilants with the opposite phone (follow-up experiment)")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.data, args.ckpt, args.out)):
        ap.error("--data, --ckpt and --out are required")
    if min(args.steps, args.batch_frames, args.accum, args.lr, args.margin, args.rank_weight) <= 0:
        ap.error("Training counts and weights must be positive")
    train(args)


if __name__ == "__main__":
    main()

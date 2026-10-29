"""One-factor real-recording pair-rank loss on the frozen paired batch plan."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import train_sibilant_cv_support as trainer
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, phase_loss
from mdd.sibilant_context_batches import make_plan
from mdd.sibilant_context_rank import context_rank_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    p, root = json.loads(Path(a.protocol).read_text()), Path(a.root)
    for path, digest in p["sha256"].items():
        if trainer.service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    torch.set_num_threads(2)
    out = root / "heads"
    out.mkdir(parents=True, exist_ok=False)
    train = dict(np.load(root / "features/train.npz"))
    labels = trainer.service.read_jsonl(root / "features/train.labels.jsonl")
    features = trainer.service.read_jsonl(root / "features/train.features.jsonl")
    plan, contexts, sampling = make_plan(train["y"], labels, features, p["seed"], p["steps"], "paired_context")
    np.testing.assert_array_equal(plan, np.load(p["control_batch_plan"]))
    np.save(out / "batch_plan.npy", plan)
    write(out / "selected_contexts.json", contexts)
    write(out / "sampling.json", sampling)
    encoder = load_encoder(p, True)
    torch.manual_seed(p["seed"])
    device = next(encoder.parameters()).device
    head = SupportReadout(encoder.config.hidden_size, True).to(device)
    initial = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in head.parameters())).hexdigest()
    control = json.loads(Path(p["control_fit"]).read_text())["arms"]["learned_support"]
    if initial != control["initial_head_sha256"]:
        raise AssertionError("initial head changed")
    optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": .001},
                                  {"params": [v for v in encoder.parameters() if v.requires_grad], "lr": .00001}], weight_decay=.01)
    losses, phases, ranks, started = [], [], [], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step, indices in enumerate(plan, 1):
        optimizer.zero_grad(set_to_none=True)
        x = torch.as_tensor(train["x"][indices], device=device)
        lengths = torch.as_tensor(train["lengths"][indices], device=device)
        mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        actual = torch.as_tensor(train["y"][indices], device=device)
        logits, role = head(hidden(encoder, x, mask), mask, torch.as_tensor(train["relative"][indices], device=device))
        ce = torch.nn.functional.cross_entropy(logits, actual)
        auxiliary = phase_loss(role, torch.as_tensor(train["phase"][indices], device=device))
        ranking = context_rank_loss(logits, actual)
        loss = ce + p["phase_weight"] * auxiliary + p["rank_weight"] * ranking
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if step == 1:
            if any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):
                raise AssertionError("earlier encoder gradient")
            if not any(v.grad is not None and bool(v.grad.abs().sum()) for layer in encoder.encoder.layers[-2:] for v in layer.parameters()):
                raise AssertionError("no encoder update")
        torch.nn.utils.clip_grad_norm_([v for group in optimizer.param_groups for v in group["params"]], 1.)
        optimizer.step()
        losses.append(float(ce.detach()))
        phases.append(float(auxiliary.detach()))
        ranks.append(float(ranking.detach()))
        if step % 50 == 0:
            print(step, "actual", np.mean(losses[-50:]), "phase", np.mean(phases[-50:]),
                  "real_pair_rank", np.mean(ranks[-50:]), "seconds", round(time.monotonic() - started, 1), flush=True)
    torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                "learned": True, "thresholds": None}, out / "uncalibrated.pt")
    report = {"sampling": sampling, "initial_head_sha256": initial,
              "batch_plan_identical_to_control": True, "rank_weight": p["rank_weight"],
              "loss_first_last50": {k: [float(np.mean(v[:50])), float(np.mean(v[-50:]))]
                                    for k, v in (("ce", losses), ("phase", phases), ("rank", ranks))},
              "seconds": time.monotonic() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "artifact_sha256": trainer.service.sha(out / "uncalibrated.pt"),
              "test_read": False, "calibration_read": False}
    write(out / "fit.json", report)
    write(out / "fit.finished.json", {"sha256": trainer.service.sha(out / "fit.json")})


if __name__ == "__main__":
    main()

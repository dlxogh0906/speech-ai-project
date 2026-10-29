"""Hold C/V learning fixed and vary actual-class/context-paired batch sampling."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
import hashlib
import json
import time
import numpy as np
import torch
import train_sibilant_cv_support as trainer
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, phase_loss
from mdd.sibilant_context_batches import make_plan


def fit(p, root):
    out = root / "heads"
    out.mkdir(parents=True, exist_ok=False)
    train, cal = dict(np.load(root / "features/train.npz")), dict(np.load(root / "features/cal.npz"))
    train_labels = trainer.service.read_jsonl(root / "features/train.labels.jsonl")
    train_features = trainer.service.read_jsonl(root / "features/train.features.jsonl")
    plan, context_keys, sampling = make_plan(train["y"], train_labels, train_features, p["seed"], p["steps"], p["sampling"])
    np.save(out / "batch_plan.npy", plan)
    write(out / "sampling.json", sampling)
    write(out / "selected_contexts.json", context_keys)
    if p["sampling"] == "all_actual":
        np.testing.assert_array_equal(plan, np.load(p["original_batch_plan"]))
    uniform = Path(p["uniform_source"])
    for name in ("uniform.pt", "uniform.cal.npz"):
        (out / name).symlink_to((uniform / name).resolve())
    uniform_report = json.loads((uniform / "fit.json").read_text())["arms"]["uniform"]
    encoder = load_encoder(p, True)
    torch.manual_seed(p["seed"])
    head = SupportReadout(encoder.config.hidden_size, True).to(next(encoder.parameters()).device)
    initial = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in head.parameters())).hexdigest()
    optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": .001},
                                  {"params": [v for v in encoder.parameters() if v.requires_grad], "lr": .00001}], weight_decay=.01)
    losses, phases, started = [], [], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    device = next(encoder.parameters()).device
    for step, indices in enumerate(plan, 1):
        optimizer.zero_grad(set_to_none=True)
        x = torch.as_tensor(train["x"][indices], device=device)
        lengths = torch.as_tensor(train["lengths"][indices], device=device)
        mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        logits, role = head(hidden(encoder, x, mask), mask, torch.as_tensor(train["relative"][indices], device=device))
        actual_loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(train["y"][indices], device=device))
        auxiliary = phase_loss(role, torch.as_tensor(train["phase"][indices], device=device))
        loss = actual_loss + p["phase_weight"] * auxiliary
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
        losses.append(float(actual_loss.detach()))
        phases.append(float(auxiliary.detach()))
        if step % 50 == 0:
            print(p["sampling"], step, "actual", np.mean(losses[-50:]), "phase", np.mean(phases[-50:]), flush=True)
    probabilities, available, seconds = trainer.score(encoder, head, cal)
    if p["sampling"] == "all_actual":
        saved = np.load(p["original_cal_scores"])
        np.testing.assert_array_equal(probabilities, saved["probabilities"])
        np.testing.assert_array_equal(available, saved["available"])
    labels = trainer.service.read_jsonl(root / "features/cal.labels.jsonl")
    features = trainer.service.read_jsonl(root / "features/cal.features.jsonl")
    thresholds, audit = trainer.calibrate(labels, features, probabilities, available)
    np.savez(out / "learned_support.cal.npz", probabilities=probabilities, available=available)
    torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                "thresholds": thresholds, "learned": True}, out / "learned_support.pt")
    report = {"sampling": sampling, "class_counts": {i: int((train["y"] == i).sum()) for i in range(4)},
              "arms": {"uniform": uniform_report, "learned_support": {
                  "initial_head_sha256": initial, "calibration": audit,
                  "class_loss_first_last50": [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
                  "phase_loss_first_last50": [float(np.mean(phases[:50])), float(np.mean(phases[-50:]))],
                  "cal_support_available": int(available.sum()), "cal_features": len(available),
                  "seconds": time.monotonic() - started, "cal_score_seconds": seconds,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()}}}
    if initial != uniform_report["initial_head_sha256"]:
        raise AssertionError("head initialization changed")
    write(out / "fit.json", report)
    write(out / "fit.finished.json", {"sha256": trainer.service.sha(out / "fit.json")})


def main():
    trainer.fit = fit
    trainer.main()


if __name__ == "__main__":
    main()

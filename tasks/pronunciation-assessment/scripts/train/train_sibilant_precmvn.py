"""Matched CAL-only pre-CMVN relative spectrum test, no test-set access.

Original normalized encoder input, initial weights, next300 batches and losses
are fixed. Reuse the exactly reproduced unchanged control. Compare normalized
relative spectrum, pre-CMVN C mean, and pre-CMVN relative spectrum branches.
"""
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
import json
import shutil
import time
import numpy as np
import torch
import train_sibilant_cv_support as trainer
from train_sibilant_c_trajectory import aucs
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_context_batches import make_plan
from mdd.sibilant_cv_support import phase_loss, reliable_support
from mdd.sibilant_relative_spectrum import RelativeSpectrumReadout


def score(encoder, head, inputs):
    if set(inputs) != {"x", "lengths", "relative", "acoustic"}:
        raise ValueError("inference accepts only observed acoustic inputs and runtime positions")
    probabilities, available = [], []
    device, started = next(encoder.parameters()).device, time.monotonic()
    with torch.no_grad():
        for start in range(0, len(inputs["x"]), 64):
            select = slice(start, start + 64)
            x = torch.as_tensor(inputs["x"][select], device=device)
            lengths = torch.as_tensor(inputs["lengths"][select], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(inputs["relative"][select], device=device)
            acoustic = torch.as_tensor(inputs["acoustic"][select], device=device)
            logits, phase = head(hidden(encoder, x, mask), mask, relative, acoustic)
            probabilities.extend(logits.softmax(-1).cpu().numpy())
            available.extend(reliable_support(phase, mask).cpu().tolist())
    torch.cuda.synchronize()
    return np.asarray(probabilities), np.asarray(available), time.monotonic() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p["sha256"].items():
        if trainer.service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    train = dict(np.load(p["train_prefix"] + ".npz"))
    cal = dict(np.load(p["cal_prefix"] + ".npz"))
    if set(cal) != {"x", "lengths", "relative"}:
        raise ValueError("CAL inputs contain labels")
    raw = {name: dict(np.load(path)) for name, path in p["acoustic"].items()}
    for name, original in (("train", train), ("cal", cal)):
        np.testing.assert_array_equal(raw[name]["lengths"], original["lengths"])
        if raw[name]["acoustic"].shape != original["x"].shape:
            raise ValueError("observed acoustic grid differs")
    observed = np.arange(train["x"].shape[1])[None] < train["lengths"][:, None]
    frames = raw["train"]["acoustic"][observed].astype(np.float64)
    mean, scale = frames.mean(0).astype(np.float32), np.maximum(frames.std(0), 1e-5).astype(np.float32)
    del frames
    np.savez(out / "training_acoustic_stats.npz", mean=mean, scale=scale)
    train_labels = trainer.service.read_jsonl(p["train_prefix"] + ".labels.jsonl")
    train_features = trainer.service.read_jsonl(p["train_prefix"] + ".features.jsonl")
    plan, _, _ = make_plan(train["y"], train_labels, train_features, p["seed"],
                           p["prior_updates"] + p["steps"], "paired_context")
    np.testing.assert_array_equal(plan[:p["prior_updates"]], np.load(p["prior_batch_plan"]))
    plan = plan[p["prior_updates"]:]
    np.testing.assert_array_equal(plan, np.load(p["unchanged_root"] + "/batch_plan.npy"))
    np.save(out / "batch_plan.npy", plan)
    source = Path(p["unchanged_root"])
    control_report = json.loads((source / "report.json").read_text())["arms"]["unchanged"]
    reference = dict(np.load(source / "initial_guard.npz"))
    for suffix in (".pt", ".cal_scores.npz"):
        (out / ("unchanged" + suffix)).symlink_to((source / ("unchanged" + suffix)).resolve())
    artifact = torch.load(p["artifact"], map_location="cpu")
    report = {"scope": p["scope"], "inference_batch": 64, "test_scored": False,
              "service_improvement_proven": False, "arms": {"unchanged": control_report},
              "criterion": p["criterion"], "acoustic_stats_observed_train_frames": int(observed.sum())}
    scores = {"unchanged": dict(np.load(out / "unchanged.cal_scores.npz"))}
    for arm in ("normalized_relative", "pre_mean", "pre_relative"):
        encoder = load_encoder(p, True)
        encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
        torch.manual_seed(p["seed"])
        normalized = arm == "normalized_relative"
        branch = "mean" if arm == "pre_mean" else "relative"
        head = RelativeSpectrumReadout(encoder.config.hidden_size, branch,
                                      np.zeros_like(mean) if normalized else mean,
                                      np.ones_like(scale) if normalized else scale).to(next(encoder.parameters()).device)
        head.load_control(artifact["head"])
        acoustic_train = train["x"] if normalized else raw["train"]["acoustic"]
        acoustic_cal = cal["x"] if normalized else raw["cal"]["acoustic"]
        device = next(encoder.parameters()).device
        indices = plan[0]
        x = torch.as_tensor(train["x"][indices], device=device)
        lengths = torch.as_tensor(train["lengths"][indices], device=device)
        mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        relative = torch.as_tensor(train["relative"][indices], device=device)
        acoustic = torch.as_tensor(acoustic_train[indices], device=device)
        with torch.no_grad():
            logits, phase = head(hidden(encoder, x, mask), mask, relative, acoustic)
            np.testing.assert_array_equal(logits.cpu().numpy(), reference["actual"])
            np.testing.assert_array_equal(phase.cpu().numpy(), reference["roles"])
        optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": .001},
                                      {"params": [v for v in encoder.parameters() if v.requires_grad], "lr": .00001}],
                                     weight_decay=.01)
        losses, started = [], time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step, indices in enumerate(plan, 1):
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(train["x"][indices], device=device)
            lengths = torch.as_tensor(train["lengths"][indices], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(train["relative"][indices], device=device)
            acoustic = torch.as_tensor(acoustic_train[indices], device=device)
            logits, phase = head(hidden(encoder, x, mask), mask, relative, acoustic)
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(train["y"][indices], device=device))
            loss = loss + p["phase_weight"] * phase_loss(phase, torch.as_tensor(train["phase"][indices], device=device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if step == 1 and any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):
                raise AssertionError("earlier encoder layers changed")
            torch.nn.utils.clip_grad_norm_([v for group in optimizer.param_groups for v in group["params"]], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(arm, step, "loss", float(np.mean(losses[-50:])), flush=True)
        train_seconds = time.monotonic() - started
        ps, av, seconds = score(encoder, head, dict(cal, acoustic=acoustic_cal))
        np.savez(out / (arm + ".cal_scores.npz"), probabilities=ps, available=av)
        labels = trainer.service.read_jsonl(p["cal_prefix"] + ".labels.jsonl")
        features = trainer.service.read_jsonl(p["cal_prefix"] + ".features.jsonl")
        thresholds, audit = trainer.calibrate(labels, features, ps, av)
        for key in thresholds:
            if key.endswith("/affricate"):
                thresholds[key] = None
        torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                    "readout": "relative_spectrum", "branch": branch, "acoustic_source": "normalized" if normalized else "precmvn",
                    "thresholds": thresholds, "learned": True}, out / (arm + ".pt"))
        scores[arm] = {"probabilities": ps, "available": av}
        report["arms"][arm] = {"thresholds": thresholds, "calibration": audit, "auc": aucs(labels, ps, av),
                               "support_available": int(av.sum()), "features": len(av), "initial_logits_and_roles_exact": True,
                               "parameters": sum(v.numel() for v in head.parameters()), "training_seconds": train_seconds,
                               "score_seconds": seconds, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                               "artifact_sha256": trainer.service.sha(out / (arm + ".pt"))}
        write(out / "report.json", report)
        print(arm, json.dumps(report["arms"][arm], ensure_ascii=False), flush=True)
        del optimizer, encoder, head
        torch.cuda.empty_cache()
    common = np.stack([v["available"] for v in scores.values()]).all(0)
    for arm, value in scores.items():
        report["arms"][arm]["common_support_auc"] = aucs(labels, value["probabilities"], common)
    candidate = report["arms"]["pre_relative"]
    gain = sum(v["cal_additional_exact"] for k, v in candidate["calibration"].items() if k.endswith("/pair"))
    report["hypothesis_screen_passed"] = bool(gain >= p["criterion"]["minimum_exact_cal_gain"]
        and all(candidate["common_support_auc"][phone]["pair_error"] >= p["criterion"]["minimum_pair_errors"]
                and all(candidate["common_support_auc"][phone]["auc"] is not None
                        and report["arms"][arm]["common_support_auc"][phone]["auc"] is not None
                        and candidate["common_support_auc"][phone]["auc"] >= report["arms"][arm]["common_support_auc"][phone]["auc"] + p["criterion"]["minimum_auc_gain"]
                        for arm in ("unchanged", "normalized_relative", "pre_mean")) for phone in ("ㅅ", "ㅆ")))
    write(out / "report.json", report)
    write(out / "finished.json", {"sha256": trainer.service.sha(out / "report.json")})
    print("hypothesis_screen_passed", report["hypothesis_screen_passed"], flush=True)


if __name__ == "__main__":
    main()

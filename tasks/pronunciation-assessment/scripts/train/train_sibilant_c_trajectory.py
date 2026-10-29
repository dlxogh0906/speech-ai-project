"""Matched small CAL-only test of within-C change versus extra capacity.

Continue the same frozen full-data model with identical next300 batches. Compare
unchanged readout, additional C mean, and additional earlier/later C change.
Batch64 CAL is only a cheap hypothesis screen, never a service evaluation.
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
import hashlib
import json
import shutil
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import train_sibilant_cv_support as trainer
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, phase_loss
from mdd.sibilant_c_trajectory import CTrajectoryReadout
from mdd.sibilant_context_batches import make_plan


def aucs(labels, scores, available):
    result = {}
    for expected in ("ㅅ", "ㅆ"):
        rows = [r for r in labels if r["expected"] == expected and r["actual"] in ("ㅅ", "ㅆ")
                and r["label_kind"] in ("correct", "pair_error") and r.get("feature_index") is not None
                and available[r["feature_index"]] and r["existing"]["phone"] is not None
                and not r["existing"]["phone"].get("errorType")
                and r["existing"]["phone"].get("alpha_jamo") == expected]
        target = [int(r["actual"] != expected) for r in rows]
        values = [float(scores[r["feature_index"], int(expected == "ㅅ")]) for r in rows]
        result[expected] = {"normal": target.count(0), "pair_error": target.count(1),
                            "auc": float(roc_auc_score(target, values)) if len(set(target)) == 2 else None}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p["sha256"].items():
        if trainer.service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    train = dict(np.load(p["train_prefix"] + ".npz"))
    train_labels = trainer.service.read_jsonl(p["train_prefix"] + ".labels.jsonl")
    train_features = trainer.service.read_jsonl(p["train_prefix"] + ".features.jsonl")
    cal = dict(np.load(p["cal_prefix"] + ".npz"))
    if set(cal) != {"x", "lengths", "relative"}:
        raise ValueError("CAL inference must contain acoustic inputs only")
    plan, _, _ = make_plan(train["y"], train_labels, train_features, p["seed"],
                           p["prior_updates"] + p["steps"], "paired_context")
    np.testing.assert_array_equal(plan[:p["prior_updates"]], np.load(p["prior_batch_plan"]))
    plan = plan[p["prior_updates"]:]
    np.save(out / "batch_plan.npy", plan)
    artifact = torch.load(p["artifact"], map_location="cpu")
    report = {"scope": p["scope"], "inference_batch": 64, "test_scored": False,
              "service_improvement_proven": False, "arms": {}, "criterion": p["criterion"]}
    initial_guard = None
    for arm in ("unchanged", "mean", "drift"):
        encoder = load_encoder(p, True)
        encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
        torch.manual_seed(p["seed"])
        head = (SupportReadout(encoder.config.hidden_size, True) if arm == "unchanged"
                else CTrajectoryReadout(encoder.config.hidden_size, arm)).to(next(encoder.parameters()).device)
        if arm == "unchanged":
            head.load_state_dict(artifact["head"])
        else:
            head.load_control(artifact["head"])
        device = next(encoder.parameters()).device
        index = plan[0]
        x = torch.as_tensor(train["x"][index], device=device)
        lengths = torch.as_tensor(train["lengths"][index], device=device)
        mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        relative = torch.as_tensor(train["relative"][index], device=device)
        with torch.no_grad():
            outputs = head(hidden(encoder, x, mask), mask, relative)
            guard = [v.detach().cpu().numpy() for v in outputs]
        if initial_guard is None:
            initial_guard = guard
            np.savez(out / "initial_guard.npz", actual=guard[0], roles=guard[1])
        else:
            for v, expected in zip(guard, initial_guard):
                np.testing.assert_array_equal(v, expected)
        optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": .001},
                                      {"params": [v for v in encoder.parameters() if v.requires_grad], "lr": .00001}],
                                     weight_decay=.01)
        losses, started = [], time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step, index in enumerate(plan, 1):
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(train["x"][index], device=device)
            lengths = torch.as_tensor(train["lengths"][index], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(train["relative"][index], device=device)
            logits, phase = head(hidden(encoder, x, mask), mask, relative)
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(train["y"][index], device=device))
            loss = loss + p["phase_weight"] * phase_loss(phase, torch.as_tensor(train["phase"][index], device=device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if step == 1 and any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):
                raise AssertionError("earlier encoder layers changed")
            torch.nn.utils.clip_grad_norm_([v for group in optimizer.param_groups for v in group["params"]], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(arm, "step", step, "loss", float(np.mean(losses[-50:])), flush=True)
        train_seconds = time.monotonic() - started
        ps, av, seconds = trainer.score(encoder, head, cal)
        np.savez(out / (arm + ".cal_scores.npz"), probabilities=ps, available=av)
        # CAL labels are never passed to the encoder/head or score function.
        labels = trainer.service.read_jsonl(p["cal_prefix"] + ".labels.jsonl")
        features = trainer.service.read_jsonl(p["cal_prefix"] + ".features.jsonl")
        thresholds, audit = trainer.calibrate(labels, features, ps, av)
        for key in thresholds:
            if key.endswith("/affricate"):
                thresholds[key] = None
        torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                    "readout": "existing" if arm == "unchanged" else "c_trajectory_" + arm,
                    "thresholds": thresholds, "learned": True}, out / (arm + ".pt"))
        report["arms"][arm] = {"thresholds": thresholds, "calibration": audit, "auc": aucs(labels, ps, av),
                               "support_available": int(av.sum()), "features": len(av),
                               "parameters": sum(v.numel() for v in head.parameters()),
                               "initial_logits_and_roles_exact": True, "training_seconds": train_seconds,
                               "score_seconds": seconds, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                               "first_last50_loss": [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))]}
        write(out / "report.json", report)
        print(arm, json.dumps(report["arms"][arm], ensure_ascii=False), flush=True)
        del optimizer, encoder, head
        torch.cuda.empty_cache()
    drift = report["arms"]["drift"]
    controls = [report["arms"][name] for name in ("unchanged", "mean")]
    gain = sum(v["cal_additional_exact"] for k, v in drift["calibration"].items() if k.endswith("/pair"))
    report["hypothesis_screen_passed"] = bool(gain >= p["criterion"]["minimum_exact_cal_gain"]
        and all(drift["auc"][phone]["pair_error"] >= p["criterion"]["minimum_pair_errors"]
                and drift["auc"][phone]["auc"] is not None
                and all(c["auc"][phone]["auc"] is not None
                        and drift["auc"][phone]["auc"] >= c["auc"][phone]["auc"] + p["criterion"]["minimum_auc_gain"]
                        for c in controls) for phone in ("ㅅ", "ㅆ")))
    write(out / "report.json", report)
    write(out / "finished.json", {"sha256": trainer.service.sha(out / "report.json")})
    print("hypothesis_screen_passed", report["hypothesis_screen_passed"], flush=True)


if __name__ == "__main__":
    main()

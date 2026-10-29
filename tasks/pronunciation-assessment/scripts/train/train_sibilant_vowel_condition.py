"""CAL-only test of known following-vowel conditioning with a matched null.

Runtime vowel IDs come from cached canonical sequences, never actual TAG vowels.
The parameter-matched null uses deterministic clip-hash groups, not linguistic
context; it is a capacity diagnostic, not a proposed service policy.
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
import collections
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
from mdd.sibilant_vowel_condition import VowelConditionReadout, VOWEL_IDS


def context_inputs(prefix):
    records = {r["row"]["uid"]: r["existing"]["raw"] for r in trainer.service.read_jsonl(prefix + ".responses.jsonl")}
    features = trainer.service.read_jsonl(prefix + ".features.jsonl")
    runtime, null = [], []
    for f in features:
        raw, q = records[f["uid"]], f["q_runtime"]
        if raw is None or q + 1 >= len(raw["canon"]) or raw["canon"][q] != f["canonical"]:
            raise ValueError("runtime context identity differs")
        vowel = raw["canon"][q + 1]
        if vowel not in VOWEL_IDS:
            raise ValueError("eligible clip does not have a runtime following vowel")
        runtime.append(VOWEL_IDS[vowel])
        null.append(1 + int(f["clip_sha256"][:16], 16) % len(VOWEL_IDS))
    return np.asarray(runtime, np.int64), np.asarray(null, np.int64)


def forward(encoder, head, observed, ids, indices):
    if set(observed) != {"x", "lengths", "relative"}:
        raise ValueError("only observed acoustics and runtime positions accepted")
    device = next(encoder.parameters()).device
    x = torch.as_tensor(observed["x"][indices], device=device)
    lengths = torch.as_tensor(observed["lengths"][indices], device=device)
    mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
    relative = torch.as_tensor(observed["relative"][indices], device=device)
    logits, phase = head(hidden(encoder, x, mask), mask, relative, torch.as_tensor(ids[indices], device=device))
    return logits, phase, mask


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
    observed = {k: train[k] for k in ("x", "lengths", "relative")}
    cal = dict(np.load(p["cal_prefix"] + ".npz"))
    if set(cal) != set(observed):
        raise ValueError("CAL inputs contain labels")
    contexts = {name: context_inputs(p[name + "_prefix"]) for name in ("train", "cal")}
    np.savez(out / "context_ids.npz", **{name + "_" + mode: contexts[name][i]
             for name in contexts for i, mode in enumerate(("runtime", "null"))})
    labels = trainer.service.read_jsonl(p["train_prefix"] + ".labels.jsonl")
    features = trainer.service.read_jsonl(p["train_prefix"] + ".features.jsonl")
    plan, _, _ = make_plan(train["y"], labels, features, p["seed"], p["prior_updates"] + p["steps"], "paired_context")
    np.testing.assert_array_equal(plan[:p["prior_updates"]], np.load(p["prior_batch_plan"]))
    plan = plan[p["prior_updates"]:]
    control = Path(p["unchanged_root"])
    np.testing.assert_array_equal(plan, np.load(control / "batch_plan.npy"))
    np.save(out / "batch_plan.npy", plan)
    artifact = torch.load(p["artifact"], map_location="cpu")
    initial = dict(np.load(control / "initial_guard.npz"))
    control_report = json.loads((control / "report.json").read_text())["arms"]["unchanged"]
    scores = {"unchanged": dict(np.load(control / "unchanged.cal_scores.npz"))}
    report = {"scope": p["scope"], "test_scored": False, "inference_batch": 64,
              "service_improvement_proven": False, "arms": {"unchanged": control_report},
              "runtime_vowel_counts": {n: dict(collections.Counter(v[0].tolist())) for n, v in contexts.items()}}
    for arm, mode in (("null_group", 1), ("runtime_vowel", 0)):
        encoder = load_encoder(p, True)
        encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
        torch.manual_seed(p["seed"])
        head = VowelConditionReadout(encoder.config.hidden_size).to(next(encoder.parameters()).device)
        head.load_control(artifact["head"])
        with torch.no_grad():
            logits, phase, _ = forward(encoder, head, observed, contexts["train"][mode], plan[0])
            np.testing.assert_array_equal(logits.cpu().numpy(), initial["actual"])
            np.testing.assert_array_equal(phase.cpu().numpy(), initial["roles"])
        optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": .001},
                                      {"params": [v for v in encoder.parameters() if v.requires_grad], "lr": .00001}], weight_decay=.01)
        device, started, losses = next(encoder.parameters()).device, time.monotonic(), []
        torch.cuda.reset_peak_memory_stats()
        for step, indices in enumerate(plan, 1):
            optimizer.zero_grad(set_to_none=True)
            logits, phase, _ = forward(encoder, head, observed, contexts["train"][mode], indices)
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(train["y"][indices], device=device))
            loss = loss + p["phase_weight"] * phase_loss(phase, torch.as_tensor(train["phase"][indices], device=device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for g in optimizer.param_groups for v in g["params"]], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(arm, step, float(np.mean(losses[-50:])), flush=True)
        training_seconds = time.monotonic() - started
        ps, av, started = [], [], time.monotonic()
        with torch.no_grad():
            for start in range(0, len(cal["x"]), 64):
                logits, phase, mask = forward(encoder, head, cal, contexts["cal"][mode], slice(start, start + 64))
                ps.extend(logits.softmax(-1).cpu().numpy())
                av.extend(reliable_support(phase, mask).cpu().tolist())
        ps, av = np.asarray(ps), np.asarray(av)
        seconds = time.monotonic() - started
        np.savez(out / (arm + ".cal_scores.npz"), probabilities=ps, available=av)
        labels = trainer.service.read_jsonl(p["cal_prefix"] + ".labels.jsonl")
        features = trainer.service.read_jsonl(p["cal_prefix"] + ".features.jsonl")
        thresholds, audit = trainer.calibrate(labels, features, ps, av)
        for key in thresholds:
            if key.endswith("/affricate"):
                thresholds[key] = None
        torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                    "thresholds": thresholds, "readout": "vowel_condition", "context": arm,
                    "vowel_ids": VOWEL_IDS, "learned": True}, out / (arm + ".pt"))
        scores[arm] = {"probabilities": ps, "available": av}
        report["arms"][arm] = {"thresholds": thresholds, "calibration": audit, "auc": aucs(labels, ps, av),
                               "support_available": int(av.sum()), "features": len(av), "initial_logits_and_roles_exact": True,
                               "training_seconds": training_seconds, "score_seconds": seconds,
                               "parameters": sum(v.numel() for v in head.parameters()), "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        write(out / "report.json", report)
        print(arm, json.dumps(report["arms"][arm], ensure_ascii=False), flush=True)
        del optimizer, encoder, head
        torch.cuda.empty_cache()
    common = np.stack([v["available"] for v in scores.values()]).all(0)
    for arm, value in scores.items():
        report["arms"][arm]["common_support_auc"] = aucs(labels, value["probabilities"], common)
    candidate = report["arms"]["runtime_vowel"]
    gain = sum(v["cal_additional_exact"] for k, v in candidate["calibration"].items() if k.endswith("/pair"))
    report["hypothesis_screen_passed"] = bool(gain >= p["criterion"]["minimum_exact_cal_gain"]
        and all(candidate["common_support_auc"][phone]["pair_error"] >= p["criterion"]["minimum_pair_errors"]
                and all(candidate["common_support_auc"][phone]["auc"] >= report["arms"][arm]["common_support_auc"][phone]["auc"] + p["criterion"]["minimum_auc_gain"]
                        for arm in ("unchanged", "null_group")) for phone in ("ㅅ", "ㅆ")))
    write(out / "report.json", report)
    write(out / "finished.json", {"sha256": trainer.service.sha(out / "report.json")})
    print("hypothesis_screen_passed", report["hypothesis_screen_passed"], flush=True)


if __name__ == '__main__':
    main()

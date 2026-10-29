"""Matched frozen/last-two-layer local encoder experiment, no TAG boundaries.

Prepare reconstructs original VAD PCM and verifies its hash, then extracts the
31-frame-equivalent runtime context. Fit sees train and calibration only;
evaluate loads frozen artifacts and retains all original response denominators.
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
import hashlib
import json
import shutil
import time
import numpy as np
import torch

import eval_sibilant_directional as service
from train_sibilant_transition import render, metrics
from audit_sibilant_service_coverage import audit as coverage_audit
from mdd.sibilant_hidden import zero_extra_thresholds
from mdd.sibilant_local_encoder import TemporalReadout


def write(path, value):
    path.write_text(json.dumps(service.jsonable(value), ensure_ascii=False, indent=2) + "\n")


def prepare(p, out):
    from transformers import SeamlessM4TFeatureExtractor
    from mdd.vad import trim
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for split, folder in p["sources"].items():
        source = Path(folder)
        features = service.read_jsonl(source / (split + ".features.jsonl"))
        records = service.read_jsonl(source / (split + ".responses.jsonl"))
        xs, audits = [None] * len(features), [None] * len(features)
        for record in records:
            if record["raw"] is None:
                continue
            wav, _ = trim(service.load_audio(record["row"])[:320000], 16000)
            digest = hashlib.sha256(wav.tobytes()).hexdigest()
            for q, ix in record["feature_indices"].items():
                f = features[ix]
                if f["uid"] != record["uid"] or f["q_runtime"] != int(q) or f["vad_pcm_sha256"] != digest:
                    raise AssertionError("cached feature/VAD identity changed")
                a = max(0, int((f["anchor_frame"] - 15) * f["frame_seconds"] * 16000))
                b = min(len(wav), int((f["anchor_frame"] + 16) * f["frame_seconds"] * 16000))
                clip = wav[a:b]
                value = fe([clip], sampling_rate=16000, return_tensors="np", padding=True)
                length = int(value["attention_mask"][0].sum())
                if length < 3:
                    raise ValueError("insufficient runtime support")
                xs[ix] = value["input_features"][0, :length].astype(np.float32)
                audits[ix] = {"uid": f["uid"], "q_runtime": int(q), "start": a, "end": b,
                              "samples": len(clip), "vad_pcm_sha256": digest,
                              "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest(), "input_frames": length}
        if any(x is None for x in xs):
            raise AssertionError("unfilled local features")
        lengths = np.array([len(x) for x in xs], dtype=np.int64)
        x = np.zeros((len(xs), int(lengths.max()), xs[0].shape[1]), np.float32)
        for i, value in enumerate(xs):
            x[i, :len(value)] = value
        np.savez(out / (split + ".npz"), x=x, lengths=lengths)
        with (out / (split + ".audit.jsonl")).open("w") as stream:
            for row in audits:
                service.write_line(stream, row)
        summary[split] = {"sites": len(xs), "input_shape": list(x.shape), "length_min_max": [int(lengths.min()), int(lengths.max())]}
        print("prepared", split, summary[split], flush=True)
    write(out / "finished.json", summary)


def load_encoder(p, tuned):
    from transformers import Wav2Vec2BertForCTC
    # Load the exact CTC checkpoint, then retain its encoder; no random encoder.
    whole = Wav2Vec2BertForCTC.from_pretrained(p["checkpoint"])
    encoder = whole.wav2vec2_bert
    del whole
    encoder.requires_grad_(False)
    if tuned:
        for layer in encoder.encoder.layers[-2:]:
            layer.requires_grad_(True)
    encoder.eval()  # Keep all dropout and running state identical in both arms.
    return encoder.to("cuda")


def hidden(encoder, x, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = encoder(input_features=x, attention_mask=mask).last_hidden_state
    if value.shape[:2] != mask.shape:
        raise AssertionError("local frame grid changed")
    return value


def score(encoder, head, data, batch=64):
    output = []
    tick = time.monotonic()
    with torch.no_grad():
        for a in range(0, len(data["x"]), batch):
            x = torch.as_tensor(data["x"][a:a + batch], device="cuda")
            lengths = torch.as_tensor(data["lengths"][a:a + batch], device="cuda")
            mask = torch.arange(x.shape[1], device="cuda")[None, :] < lengths[:, None]
            output.extend(head(hidden(encoder, x, mask), mask).softmax(-1)[:, 1].cpu().tolist())
    torch.cuda.synchronize()
    return np.array(output), time.monotonic() - tick


def fit(p, root):
    out = root / "heads"
    out.mkdir(parents=True, exist_ok=False)
    sources = {s: Path(f) for s, f in p["sources"].items()}
    features = service.read_jsonl(sources["train"] / "train.features.jsonl")
    labels = service.read_jsonl(sources["train"] / "train.labels.jsonl")
    groups = collections.defaultdict(list)
    targets = np.full(len(features), -1, np.int64)
    for row in labels:
        ix = row.get("feature_index")
        if row["state"] != "available" or ix is None or row["actual"] not in ("ㅅ", "ㅆ"):
            continue
        if row["uid"] != features[ix]["uid"] or row["q_runtime"] != features[ix]["q_runtime"]:
            raise AssertionError("training label identity changed")
        targets[ix] = int(row["actual"] == "ㅆ")
        groups[row["expected"] + row["actual"]].append(ix)
    if len(groups) != 4:
        raise ValueError("four train-only expected/actual groups required")
    rng = np.random.RandomState(p["seed"])
    plan = np.array([np.concatenate([rng.choice(groups[k], p["per_group"], replace=True)
                                    for k in sorted(groups)]) for _ in range(p["steps"])])
    np.save(out / "batch_plan.npy", plan)
    train = dict(np.load(root / "features/train.npz"))
    cal = dict(np.load(root / "features/cal.npz"))
    cal_labels = service.read_jsonl(sources["cal"] / "cal.labels.jsonl")
    cal_features = service.read_jsonl(sources["cal"] / "cal.features.jsonl")
    report = {"training_groups": {k: len(v) for k, v in groups.items()}, "arms": {}}
    for arm in ("frozen_local", "tuned_last2_local"):
        tuned = arm.startswith("tuned")
        torch.cuda.reset_peak_memory_stats()
        encoder = load_encoder(p, tuned)
        torch.manual_seed(p["seed"])
        head = TemporalReadout(encoder.config.hidden_size).to("cuda")
        initial = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in head.parameters())).hexdigest()
        optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": p["head_lr"]}] +
                    ([{"params": [v for v in encoder.parameters() if v.requires_grad], "lr": p["encoder_lr"]}] if tuned else []), weight_decay=.01)
        started, losses, gradient_audit = time.monotonic(), [], None
        for step, indices in enumerate(plan):
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(train["x"][indices], device="cuda")
            lengths = torch.as_tensor(train["lengths"][indices], device="cuda")
            mask = torch.arange(x.shape[1], device="cuda")[None, :] < lengths[:, None]
            y = torch.as_tensor(targets[indices], device="cuda")
            logits = head(hidden(encoder, x, mask), mask)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if step == 0:
                gradient_audit = {"encoder_parameter_tensors_with_gradient": sum(v.grad is not None for v in encoder.parameters()),
                                  "earlier_layers_have_gradient": any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()),
                                  "head_nonzero_gradient": any(v.grad is not None and bool(v.grad.abs().sum()) for v in head.parameters())}
                if gradient_audit["earlier_layers_have_gradient"] or not gradient_audit["head_nonzero_gradient"]:
                    raise AssertionError("gradient scope incorrect")
                if tuned and not gradient_audit["encoder_parameter_tensors_with_gradient"]:
                    raise AssertionError("fine-tuning did not reach encoder")
                np.save(out / (arm + ".initial_logits.npy"), logits.detach().cpu().numpy())
            torch.nn.utils.clip_grad_norm_([v for group in optimizer.param_groups for v in group["params"]], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 50 == 0:
                print(arm, step + 1, "loss", float(np.mean(losses[-50:])), "seconds", round(time.monotonic() - started, 1), flush=True)
        scores, seconds = score(encoder, head, cal)
        thresholds, audit = zero_extra_thresholds(cal_labels, cal_features, scores)
        np.save(out / (arm + ".cal_scores.npy"), scores)
        artifact = {"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()} if tuned else None,
                    "thresholds": thresholds, "protocol": p, "head_initial_sha256": initial}
        torch.save(artifact, out / (arm + ".pt"))
        report["arms"][arm] = {"thresholds": thresholds, "calibration": audit, "training_seconds": time.monotonic() - started,
                               "cal_score_seconds": seconds, "head_initial_sha256": initial,
                               "loss_first_last50": [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
                               "gradient_audit": gradient_audit, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        write(out / "fit.json", report)
        del optimizer, head, encoder
        torch.cuda.empty_cache()
    np.testing.assert_array_equal(np.load(out / "frozen_local.initial_logits.npy"), np.load(out / "tuned_last2_local.initial_logits.npy"))
    write(out / "fit.finished.json", {"sha256": service.sha(out / "fit.json"), "initial_logits_exact": True})


def evaluate(p, root):
    head_dir = root / "heads"
    if not (head_dir / "fit.finished.json").exists():
        raise ValueError("fit incomplete")
    meta = json.loads((Path(p["sources"]["cal"]) / "metadata.json").read_text())["service_metadata"]
    paths = {Path(k).name: k for k in meta["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"], utt_pct=paths["utt_percentile.json"])
    # Formatter initialization also loads the untouched original model; release
    # its GPU weights while using cached raw responses for API formatting.
    app = service.app_module(config, meta["base_ckpt"])
    app.S["dec"].model.to("cpu")
    torch.cuda.empty_cache()
    report = {"scope": p["scope"], "arms": {}, "deployment": False}
    for arm in ("frozen_local", "tuned_last2_local"):
        artifact = torch.load(head_dir / (arm + ".pt"), map_location="cpu")
        encoder = load_encoder(p, False)
        if artifact["last2"] is not None:
            encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
        head = TemporalReadout(encoder.config.hidden_size).to("cuda")
        head.load_state_dict(artifact["head"])
        results = {}
        for split in ("cal", "test"):
            source = Path(p["sources"][split])
            scores, seconds = score(encoder, head, dict(np.load(root / ("features/" + split + ".npz"))))
            if split == "cal":
                np.testing.assert_array_equal(scores, np.load(head_dir / (arm + ".cal_scores.npy")))
            np.save(head_dir / (arm + "." + split + ".scores.npy"), scores)
            records = service.read_jsonl(source / (split + ".responses.jsonl"))
            output, phones = render(app, records, scores, artifact["thresholds"])
            labels = service.read_jsonl(source / (split + ".labels.jsonl"))
            rows = [dict(r, candidate_phone=phones[r["uid"]][r["q_runtime"]] if r["q_runtime"] is not None and phones[r["uid"]] else None) for r in labels]
            for suffix, values in (("responses", output), ("sites", rows)):
                with (head_dir / (arm + "." + split + "." + suffix + ".jsonl")).open("w") as stream:
                    for row in values:
                        service.write_line(stream, row)
            detail = {"overall": metrics(rows), "service_coverage": coverage_audit(rows),
                      "extra_encoder_score_seconds": seconds, "feature_sites": len(scores)}
            for field in ("following", "position", "item", "language"):
                detail[field] = {str(v): metrics([r for r in rows if r[field] == v]) for v in sorted({r[field] for r in rows}, key=str)}
            results[split] = detail
            print(arm, split, json.dumps(detail["overall"], ensure_ascii=False), flush=True)
        report["arms"][arm] = results
        del head, encoder
        torch.cuda.empty_cache()
        write(head_dir / "report.json", report)
    write(head_dir / "finished.json", {"report_sha256": service.sha(head_dir / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", required=True, choices=("prepare", "fit", "evaluate"))
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    torch.set_num_threads(2)
    root = Path(args.out)
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.protocol, root / "protocol.json")
        shutil.copy2(__file__, root / Path(__file__).name)
        prepare(p, root / "features")
    elif args.stage == "fit":
        fit(p, root)
    else:
        evaluate(p, root)


if __name__ == "__main__":
    main()

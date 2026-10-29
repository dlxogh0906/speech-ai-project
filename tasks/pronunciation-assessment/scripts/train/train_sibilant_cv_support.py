"""Matched four-class local encoder heads with fixed versus learned C/V support."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"), str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
import argparse
import collections
import copy
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render, comparison
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, phase_loss, reliable_support
from mdd.sibilant_transition import PAIR


def score(encoder, head, data):
    scores, availability = [], []
    tick = time.monotonic()
    if "phase" in data:
        raise ValueError("evaluation must not have TAG phase input")
    with torch.no_grad():
        for a in range(0, len(data["x"]), 64):
            x = torch.as_tensor(data["x"][a:a + 64], device="cuda")
            lengths = torch.as_tensor(data["lengths"][a:a + 64], device="cuda")
            mask = torch.arange(x.shape[1], device="cuda")[None] < lengths[:, None]
            relative = torch.as_tensor(data["relative"][a:a + 64], device="cuda")
            logits, phase = head(hidden(encoder, x, mask), mask, relative)
            scores.extend(logits.softmax(-1).cpu().numpy())
            available = reliable_support(phase, mask) if head.learned else torch.ones(len(x), dtype=torch.bool, device="cuda")
            availability.extend(available.cpu().tolist())
    torch.cuda.synchronize()
    return np.array(scores), np.array(availability), time.monotonic() - tick


def proposal(feature, probabilities, available):
    if not available:
        return None
    expected, actual = feature["canonical"], feature["baseline_pred"]
    winner = int(np.argmax(probabilities))
    if actual in PAIR and winner == int(PAIR[expected] == "ㅆ"):
        return "pair", PAIR[expected], float(probabilities[winner])
    if actual in ("ㅈ", "ㅉ", "ㅊ") and winner == 2:
        return "affricate", actual, float(probabilities[2])
    return None


def calibrate(labels, features, probabilities, available):
    thresholds, audit = {}, {}
    for expected in PAIR:
        normals = [r for r in labels if r["expected"] == r["actual"] == expected and r["existing"]["phone"] is not None]
        for kind in ("pair", "affricate"):
            positive, negative = [], []
            for r in labels:
                ix, phone = r.get("feature_index"), r["existing"]["phone"]
                if r["expected"] != expected or ix is None or phone is None or phone.get("errorType"):
                    continue
                candidate = proposal(features[ix], probabilities[ix], available[ix])
                if candidate is None or candidate[0] != kind:
                    continue
                correct = (r["label_kind"] not in ("correct", "ambiguous") and phone.get("alpha_jamo") == expected and r["actual"] == candidate[1])
                (positive if correct else negative).append(candidate[2])
            threshold = None
            if len(normals) >= 100 and negative:
                threshold = max(.5, float(np.nextafter(max(negative), np.inf)))
                if threshold > 1:
                    threshold = None
            key = expected + "/" + kind
            thresholds[key] = threshold
            audit[key] = {"normals_known": len(normals), "negative": len(negative), "positive": len(positive), "threshold": threshold,
                          "cal_additional_exact": sum(v >= threshold for v in positive) if threshold is not None else 0}
    return thresholds, audit


def fit(p, root):
    out = root / "heads"
    out.mkdir(parents=True, exist_ok=False)
    train = dict(np.load(root / "features/train.npz"))
    cal = dict(np.load(root / "features/cal.npz"))
    groups = {i: np.flatnonzero(train["y"] == i) for i in range(4)}
    if not all(len(v) for v in groups.values()):
        raise ValueError("all four actual classes required")
    rng = np.random.RandomState(p["seed"])
    plan = np.array([np.concatenate([rng.choice(groups[i], 4, replace=True) for i in range(4)]) for _ in range(p["steps"])])
    np.save(out / "batch_plan.npy", plan)
    labels = service.read_jsonl(root / "features/cal.labels.jsonl")
    features = service.read_jsonl(root / "features/cal.features.jsonl")
    report = {"class_counts": {k: len(v) for k, v in groups.items()}, "arms": {}}
    for arm in ("uniform", "learned_support"):
        encoder = load_encoder(p, True)
        torch.manual_seed(p["seed"])
        head = SupportReadout(encoder.config.hidden_size, arm == "learned_support").to("cuda")
        initial = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in head.parameters())).hexdigest()
        optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": .001},
                                      {"params": [v for v in encoder.parameters() if v.requires_grad], "lr": .00001}], weight_decay=.01)
        losses, phases, started = [], [], time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step, indices in enumerate(plan, 1):
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(train["x"][indices], device="cuda")
            lengths = torch.as_tensor(train["lengths"][indices], device="cuda")
            mask = torch.arange(x.shape[1], device="cuda")[None] < lengths[:, None]
            logits, role = head(hidden(encoder, x, mask), mask, torch.as_tensor(train["relative"][indices], device="cuda"))
            actual_loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(train["y"][indices], device="cuda"))
            auxiliary = phase_loss(role, torch.as_tensor(train["phase"][indices], device="cuda"))
            loss = actual_loss + (p["phase_weight"] if head.learned else 0.) * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if step == 1:
                if any(v.grad is not None for layer in encoder.encoder.layers[:-2] for v in layer.parameters()):
                    raise AssertionError("earlier encoder gradient")
                if not any(v.grad is not None and bool(v.grad.abs().sum()) for layer in encoder.encoder.layers[-2:] for v in layer.parameters()):
                    raise AssertionError("no encoder update")
            torch.nn.utils.clip_grad_norm_([v for g in optimizer.param_groups for v in g["params"]], 1.)
            optimizer.step()
            losses.append(float(actual_loss.detach()))
            phases.append(float(auxiliary.detach()))
            if step % 50 == 0:
                print(arm, step, "actual", np.mean(losses[-50:]), "phase", np.mean(phases[-50:]), "seconds", round(time.monotonic() - started, 1), flush=True)
        probabilities, available, seconds = score(encoder, head, cal)
        thresholds, audit = calibrate(labels, features, probabilities, available)
        np.savez(out / (arm + ".cal.npz"), probabilities=probabilities, available=available)
        torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "last2": {k: v.detach().cpu() for k, v in encoder.encoder.layers[-2:].state_dict().items()},
                    "thresholds": thresholds, "learned": head.learned}, out / (arm + ".pt"))
        report["arms"][arm] = {"initial_head_sha256": initial, "calibration": audit,
                               "class_loss_first_last50": [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))],
                               "phase_loss_first_last50": [float(np.mean(phases[:50])), float(np.mean(phases[-50:]))],
                               "cal_support_available": int(available.sum()), "cal_features": len(available),
                               "seconds": time.monotonic() - started, "cal_score_seconds": seconds,
                               "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        write(out / "fit.json", report)
        del optimizer, head, encoder
        torch.cuda.empty_cache()
    if len({v["initial_head_sha256"] for v in report["arms"].values()}) != 1:
        raise AssertionError("head initialization differs")
    write(out / "fit.finished.json", {"sha256": service.sha(out / "fit.json")})


def evaluate(p, root):
    out = root / "heads"
    if not (out / "fit.finished.json").exists():
        raise ValueError("fit incomplete")
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "arms": {}, "deployment": False}
    for arm in ("uniform", "learned_support"):
        artifact = torch.load(out / (arm + ".pt"), map_location="cpu")
        encoder = load_encoder(p, False)
        encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
        head = SupportReadout(encoder.config.hidden_size, artifact["learned"]).to("cuda")
        head.load_state_dict(artifact["head"])
        results = {}
        for split in ("cal", "test"):
            probabilities, available, seconds = score(encoder, head, dict(np.load(root / ("features/" + split + ".npz"))))
            if split == "cal":
                saved = np.load(out / (arm + ".cal.npz"))
                np.testing.assert_array_equal(probabilities, saved["probabilities"])
                np.testing.assert_array_equal(available, saved["available"])
            np.savez(out / (arm + "." + split + ".scores.npz"), probabilities=probabilities, available=available)
            features = service.read_jsonl(root / ("features/" + split + ".features.jsonl"))
            labels = service.read_jsonl(root / ("features/" + split + ".labels.jsonl"))
            records = service.read_jsonl(root / ("features/" + split + ".responses.jsonl"))
            by_uid = collections.defaultdict(list)
            for i, feature in enumerate(features):
                by_uid[feature["uid"]].append(i)
            rows, counts = [], collections.Counter()
            by_label = collections.defaultdict(list)
            for label in labels:
                by_label[label["uid"]].append(label)
            with (out / (arm + "." + split + ".responses.jsonl")).open("x") as stream:
                for record in records:
                    row, base = record["row"], record["existing"]
                    candidate = dict(base)
                    if base["raw"] is not None:
                        raw = base["raw"]
                        sites = copy.deepcopy(raw["sites"])
                        phones = service.phonemes(base["body"])
                        for ix in by_uid[row["uid"]]:
                            f = features[ix]
                            decision = proposal(f, probabilities[ix], available[ix])
                            counts["support_available" if available[ix] else "support_unavailable"] += 1
                            if decision is None:
                                continue
                            kind, alt, confidence = decision
                            threshold = artifact["thresholds"][f["canonical"] + "/" + kind]
                            q = f["q_runtime"]
                            if threshold is not None and confidence >= threshold and not phones[q].get("errorType"):
                                sites[q]["sz"] = {"alt": alt, "p": confidence, "th": threshold}
                        candidate["body"] = render(app, row, raw, sites)
                        new = service.phonemes(candidate["body"])
                        for q, (a, b) in enumerate(zip(phones, new)):
                            if a.get("pronunciation") != b.get("pronunciation") or a.get("errorType") and a != b or "sz" not in sites[q] and a != b:
                                raise AssertionError("existing score/alert or noncandidate phone changed")
                    views = {"original": observe(record["original"], row), "existing": observe(base, row),
                             "prior": observe(record["prior"], row), "typed": observe(candidate, row)}
                    for label in by_label[row["uid"]]:
                        value = dict(label)
                        value.update({name: view[label["q"]] for name, view in views.items()})
                        rows.append(value)
                    service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
            with (out / (arm + "." + split + ".sites.jsonl")).open("x") as stream:
                for row in rows:
                    service.write_line(stream, row)
            detail = {"versus_original": comparison(rows, "original", "typed"), "versus_reference": comparison(rows, "existing", "typed"),
                      "versus_prior": comparison(rows, "prior", "typed"), "counts": dict(counts), "score_seconds": seconds}
            for field in ("label_kind", "expected", "following", "position", "item", "language"):
                detail[field] = {str(v): comparison([r for r in rows if r[field] == v], "existing", "typed") for v in sorted({r[field] for r in rows}, key=str)}
            results[split] = detail
            print(arm, split, json.dumps(detail["versus_reference"], ensure_ascii=False), flush=True)
        report["arms"][arm] = results
        write(out / "report.json", report)
        del head, encoder
        torch.cuda.empty_cache()
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    torch.set_num_threads(2)
    (fit if a.stage == "fit" else evaluate)(p, Path(a.root))


if __name__ == "__main__":
    main()

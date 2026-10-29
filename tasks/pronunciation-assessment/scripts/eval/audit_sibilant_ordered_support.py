"""CAL-only check of ordered versus scattered phonemic role weights.

Freeze model predictions/ordered probabilities before loading audit role values.
Coarse TAG C/V labels are not physical frication/aspiration/glottal events.
This does not fit a new classifier or establish service improvement.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import json
import shutil
import numpy as np
import torch
import eval_sibilant_directional as service
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, reliable_support
from mdd.sibilant_ordered_support import ordered_phase_probabilities


def masses(probabilities, targets, mask):
    weights = probabilities[..., 1:] * mask[..., None]
    weights = weights / np.maximum(weights.sum(1, keepdims=True), 1e-30)
    own, other = [], []
    for role in (1, 2, 3):
        own.append((weights[..., role - 1] * (targets == role)).sum(1))
        other.append((weights[..., role - 1] * ((targets >= 0) & (targets != role))).sum(1))
    return np.stack(own, 1), np.stack(other, 1)


def summary(selected, a, b, ao, bo):
    result = {"sites": int(selected.sum()), "roles": {}}
    for j, role in enumerate(("C", "earlyV", "lateV")):
        result["roles"][role] = {"original_own_median": float(np.median(a[selected, j])) if selected.any() else None,
                                "ordered_own_median": float(np.median(b[selected, j])) if selected.any() else None,
                                "paired_own_change_median": float(np.median((b - a)[selected, j])) if selected.any() else None,
                                "original_other_median": float(np.median(ao[selected, j])) if selected.any() else None,
                                "ordered_other_median": float(np.median(bo[selected, j])) if selected.any() else None,
                                "paired_other_change_median": float(np.median((bo - ao)[selected, j])) if selected.any() else None}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, expected in p["sha256"].items():
        if service.sha(path) != expected:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    data = dict(np.load(p["inputs"]))
    if set(data) != {"x", "relative", "lengths"}:
        raise ValueError("runtime inputs must not contain annotation")
    artifact = torch.load(p["artifact"], map_location="cpu")
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    head = SupportReadout(encoder.config.hidden_size, True).to(next(encoder.parameters()).device)
    head.load_state_dict(artifact["head"])
    original, ordered, actual, available = [], [], [], []
    device = next(encoder.parameters()).device
    with torch.no_grad():
        for i in range(len(data["x"])):
            x = torch.as_tensor(data["x"][i:i + 1], device=device)
            length = torch.as_tensor(data["lengths"][i:i + 1], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < length[:, None]
            logits, phase = head(hidden(encoder, x, mask), mask, torch.as_tensor(data["relative"][i:i + 1], device=device))
            original.append(phase.softmax(-1)[0].cpu().numpy())
            ordered.append(ordered_phase_probabilities(phase, mask)[0].cpu().numpy())
            actual.append(logits.softmax(-1)[0].cpu().numpy())
            available.append(bool(reliable_support(phase, mask)[0]))
            if (i + 1) % 400 == 0:
                print("CAL runtime predictions frozen", i + 1, flush=True)
    original, ordered, actual, available = np.array(original), np.array(ordered), np.array(actual), np.array(available)
    reference = np.load(p["reference_scores"])
    np.testing.assert_array_equal(actual, reference["probabilities"])
    np.testing.assert_array_equal(available, reference["available"])
    np.savez(out / "inference.npz", original=original, ordered=ordered, actual=actual, available=available)
    write(out / "inference.finished.json", {"sha256": service.sha(out / "inference.npz"), "TAG_values_used_in_inference": False})
    del encoder, head
    torch.cuda.empty_cache()
    # Read human role values only AFTER inference and reproducibility checks.
    truth = np.load(p["audit_roles"])
    np.testing.assert_array_equal(truth["lengths"], data["lengths"])
    targets = truth["phase"]
    mask = np.arange(original.shape[1])[None] < data["lengths"][:, None]
    complete = np.stack([(targets == role).any(1) for role in (1, 2, 3)], 1).all(1)
    confident = complete & available
    own, other = masses(original, targets, mask)
    new_own, new_other = masses(ordered, targets, mask)
    report = {"scope": p["scope"], "all_input_features": len(original), "complete": summary(complete, own, new_own, other, new_other),
              "confident": summary(confident, own, new_own, other, new_other), "deployment": False,
              "original_probabilities_and_availability_reproduced_exactly": True, "decision_criterion": p["criterion"],
              "inference_batch": 1, "new_classifier_fitted": False, "subgroups": {}}
    roles = report["confident"]["roles"]
    report["localization_hypothesis_supported"] = bool(confident.sum() >= p["criterion"]["minimum_sites"]
        and roles["C"]["paired_own_change_median"] >= -p["criterion"]["maximum_C_mass_drop"]
        and all(roles[k]["paired_own_change_median"] >= p["criterion"]["minimum_V_mass_gain"]
                and roles[k]["paired_other_change_median"] < 0 for k in ("earlyV", "lateV")))
    labels = service.read_jsonl(p["labels"])
    for field in ("following", "item", "label_kind"):
        result = {}
        for value in sorted({row[field] for row in labels}, key=str):
            indices = {row["feature_index"] for row in labels if row[field] == value and row.get("feature_index") is not None}
            selected = np.array([i in indices for i in range(len(original))]) & confident
            result[str(value)] = summary(selected, own, new_own, other, new_other)
        report["subgroups"][field] = result
    write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})
    print(json.dumps({k: report[k] for k in ("all_input_features", "confident", "localization_hypothesis_supported")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

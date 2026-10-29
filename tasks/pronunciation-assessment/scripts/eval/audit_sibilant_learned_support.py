"""Audit inferred coarse C/V roles after freezing all inference outputs.

TAG is read only after inference. Its phone intervals are not independent
frication, aspiration, resonance or glottal-event boundaries. Nominal feature
frame centers do not delimit the encoder's receptive field.
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
import collections
import hashlib
import json
import shutil
import numpy as np
import torch
import eval_sibilant_directional as service
from audit_sibilant_anchor_coverage import retained_samples
from extract_sibilant_localization_diagnostic import collect
from train_sibilant_local_encoder import load_encoder, hidden, write
from mdd.sibilant_cv_support import SupportReadout, reliable_support
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles
from mdd.sibilant_support_audit import support_measurements
from mdd.canonical_fix import CONSONANTS
from mdd.sibilant_transition import VOWELS
from mdd.vad import trim


def summary(rows):
    valid = [r for r in rows if r["audit_state"] == "available"]
    complete = [r for r in valid if r["support"]["complete_observed_roles"]]
    confident = [r for r in complete if r["inferred_available"]]
    result = {"all_sites": len(rows), "states": dict(collections.Counter(r["audit_state"] for r in rows)),
              "known_intervals": len(valid), "all_roles_observed": len(complete), "confident_and_complete": len(confident)}
    for name, selected in (("complete", complete), ("confident", confident)):
        item = {"sites": len(selected)}
        for j, role in enumerate(("C", "earlyV", "lateV")):
            values = [r["support"] for r in selected]
            hits = [r["peak_hits"][j] for r in values if r["peak_hits"][j] is not None]
            soft = [r["soft_mass"][j]["own_known"] for r in values]
            sharp = [r["confident_mass"][j] for r in values if r["confident_mass"][j] is not None]
            squared = [r["squared_mass"][j] for r in values if r["squared_mass"][j] is not None]
            item[role] = {"peak_correct": sum(hits), "peak_known": len(hits),
                          "soft_own_mass_p10_p50_p90": np.quantile(soft, [.1, .5, .9]).tolist() if soft else None,
                          "confident_own_mass_p10_p50_p90": np.quantile(sharp, [.1, .5, .9]).tolist() if sharp else None,
                          "squared_own_mass_p10_p50_p90": np.quantile(squared, [.1, .5, .9]).tolist() if squared else None}
        result[name] = item
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out, source = Path(a.out), Path(p["model_root"])
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    artifact = torch.load(p["artifact"], map_location="cpu")
    torch.set_num_threads(2)
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    head = SupportReadout(encoder.config.hidden_size, True).to("cuda")
    head.load_state_dict(artifact["head"])
    report = {"scope": p["scope"], "deployment": False, "splits": {}}
    # Freeze BOTH splits' predictions before reading any held-out annotation.
    for split in ("cal", "test"):
        data = dict(np.load(source / ("features/" + split + ".npz")))
        if "phase" in data or "y" in data:
            raise ValueError("heldout annotation present in model input")
        actual, probabilities, availability = [], [], []
        with torch.no_grad():
            for start in range(0, len(data["x"]), 64):
                x = torch.as_tensor(data["x"][start:start + 64], device="cuda")
                lengths = torch.as_tensor(data["lengths"][start:start + 64], device="cuda")
                mask = torch.arange(x.shape[1], device="cuda")[None] < lengths[:, None]
                relative = torch.as_tensor(data["relative"][start:start + 64], device="cuda")
                logits, phase = head(hidden(encoder, x, mask), mask, relative)
                actual.extend(logits.softmax(-1).cpu().numpy())
                probabilities.extend(phase.softmax(-1).cpu().numpy())
                availability.extend(reliable_support(phase, mask).cpu().tolist())
        actual, availability = np.array(actual), np.array(availability)
        saved = np.load(source / ("heads/learned_support." + split + ".scores.npz"))
        np.testing.assert_array_equal(actual, saved["probabilities"])
        np.testing.assert_array_equal(availability, saved["available"])
        np.savez(out / (split + ".inferred.npz"), actual=actual, phase=np.array(probabilities), available=availability)
        print(split, "inference frozen, original probabilities/availability exact", len(actual), flush=True)
    write(out / "inference.finished.json", {s: service.sha(out / (s + ".inferred.npz")) for s in ("cal", "test")})
    del head, encoder
    torch.cuda.empty_cache()
    wanted = {r["row"]["uid"] for split in ("cal", "test") for r in service.read_jsonl(source / ("features/" + split + ".responses.jsonl"))}
    tags = collect(p["evaluation_annotations"], wanted)
    for split in ("cal", "test"):
        features = service.read_jsonl(source / ("features/" + split + ".features.jsonl"))
        labels = service.read_jsonl(source / ("features/" + split + ".labels.jsonl"))
        records = service.read_jsonl(source / ("features/" + split + ".responses.jsonl"))
        arrays = np.load(source / ("features/" + split + ".npz"))
        inferred = np.load(out / (split + ".inferred.npz"))
        by_uid = collections.defaultdict(list)
        for label in labels:
            by_uid[label["uid"]].append(label)
        rows, reproduced = [], 0
        for record in records:
            row, base = record["row"], record["existing"]
            meta = tags[row["uid"]]
            if meta["spk"] != row["speaker"] or meta["item"] != row["item"]:
                raise AssertionError("annotation identity mismatch")
            intervals = single_phone_intervals(meta["tag_phones"], row["actual"])
            indices, wav = None, None
            if any(r["feature_index"] is not None for r in by_uid[row["uid"]]):
                x = service.load_audio(row)[:320000]
                indices = retained_samples(x)
                wav, _ = trim(x, 16000)
                np.testing.assert_array_equal(x[indices], wav)
                digest = hashlib.sha256(wav.tobytes()).hexdigest()
                reproduced += 1
            for label in by_uid[row["uid"]]:
                ix = label["feature_index"]
                value = {k: label.get(k) for k in ("uid", "q", "expected", "actual", "label_kind", "item", "following", "position", "language")}
                value["audit_state"] = label["existing"]["state"]
                if ix is not None:
                    f, length = features[ix], int(arrays["lengths"][ix])
                    if f["vad_pcm_sha256"] != digest or hashlib.sha256(wav[f["left"]:f["right"]].tobytes()).hexdigest() != f["clip_sha256"]:
                        raise AssertionError("runtime audio/clip changed")
                    value.update(inferred_available=bool(inferred["available"][ix]), raw_expected_restored=f["baseline_pred"] == label["expected"] and label["label_kind"] not in ("correct", "ambiguous"))
                    if intervals is None:
                        value["audit_state"] = "tag_sequence_unresolved"
                    elif label["alignment_ambiguous"] or label["actual"] not in CONSONANTS:
                        value["audit_state"] = "no_single_actual_consonant"
                    else:
                        centers = f["left"] + (np.arange(length) + .5) * (f["right"] - f["left"]) / length
                        times = indices[np.minimum(centers.astype(int), len(indices) - 1)] / 16000.
                        roles = coarse_roles(times, intervals, label["a"], VOWELS, .12, .01)
                        if not (roles >= 0).any():
                            value["audit_state"] = "no_singleton_cv_intervals"
                        else:
                            value["audit_state"] = "available"
                            value["support"] = support_measurements(inferred["phase"][ix, :length], roles, np.ones(length, bool))
                            target = 0 if label["actual"] == "ㅅ" else 1 if label["actual"] == "ㅆ" else 2 if label["actual"] in ("ㅈ", "ㅉ", "ㅊ") else 3
                            value["actual_group_correct"] = int(inferred["actual"][ix].argmax()) == target
                rows.append(value)
        with (out / (split + ".sites.jsonl")).open("x") as stream:
            for row in rows:
                service.write_line(stream, row)
        result = {"overall": summary(rows), "reproduced_vad_inputs": reproduced,
                  "normal": summary([r for r in rows if r["label_kind"] == "correct"]),
                  "error": summary([r for r in rows if r["label_kind"] not in ("correct", "ambiguous")]),
                  "raw_restored": summary([r for r in rows if r.get("raw_expected_restored")])}
        for field in ("following", "position", "item", "label_kind", "actual_group_correct"):
            result[field] = {str(v): summary([r for r in rows if r.get(field) == v]) for v in sorted({r.get(field) for r in rows}, key=str)}
        report["splits"][split] = result
        write(out / "report.json", report)
        print(split, json.dumps({"overall": result["overall"], "error": result["error"]}, ensure_ascii=False), flush=True)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


if __name__ == "__main__":
    main()

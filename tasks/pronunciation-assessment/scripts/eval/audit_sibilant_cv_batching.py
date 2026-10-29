"""Check a frozen C/V confirmation policy with one local clip per forward.

Runtime eligibility is fixed from decoder/API state. Compare against saved
batched probabilities on identical acoustic input; render and score full cohorts.
Never adjust thresholds or use annotation boundaries for inference.
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
import json
import shutil
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import restored_sites, verify_phones, detail
from train_sibilant_cv_support import score
from train_sibilant_local_encoder import load_encoder, write
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_cv_policy import annotate
from mdd.sibilant_transition import PAIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    policy = json.loads(Path(p["policy"]).read_text())
    artifact = torch.load(p["artifact"], map_location="cpu")
    assert artifact["thresholds"] == policy["thresholds"]
    torch.set_num_threads(2)
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    head = SupportReadout(encoder.config.hidden_size, True).to("cuda")
    head.load_state_dict(artifact["head"])
    report = {"deployment": False, "thresholds_unchanged": True, "cohorts": {}}
    for name, c in p["batch_audit_sources"].items():
        records = service.read_jsonl(c["records"])
        features = service.read_jsonl(c["features"])
        labels = service.read_jsonl(c["labels"])
        data, old = dict(np.load(c["inputs"])), np.load(c["scores"])
        by_record = {r["row"]["uid"]: r for r in records}
        by_label = collections.defaultdict(list)
        for label in labels:
            by_label[label["uid"]].append(label)
        selected, by_uid = [], collections.defaultdict(dict)
        for i, f in enumerate(features):
            base = by_record[f["uid"]][c["base"]]
            raw, q = base["raw"], f["q_runtime"]
            if raw["sites"][q].get("pred") == PAIR[raw["canon"][q]] and not service.phonemes(base["body"])[q].get("errorType"):
                selected.append(i)
        counts, differences, timings, audits = collections.Counter(), [], [], []
        for i in selected:
            single = {k: v[i:i + 1] for k, v in data.items()}
            probabilities, available, seconds = score(encoder, head, single)
            f = features[i]
            obs = {"probabilities": probabilities[0].tolist(), "available": bool(available[0])}
            by_uid[f["uid"]][f["q_runtime"]] = obs
            difference = float(np.max(np.abs(probabilities[0] - old["probabilities"][i])))
            differences.append(difference)
            timings.append(1000 * seconds)
            audits.append(dict(f, original_probabilities=old["probabilities"][i].tolist(),
                               original_available=bool(old["available"][i]), single_request=obs, max_probability_difference=difference))
        rows = []
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for record in records:
                row, base = record["row"], record[c["base"]]
                candidate = dict(base)
                if by_uid[row["uid"]]:
                    raw, phones = base["raw"], service.phonemes(base["body"])
                    sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                    sites, states = annotate(raw["canon"], sites, phones, by_uid[row["uid"]], policy["thresholds"])
                    counts.update(states.values())
                    candidate["body"] = render(app, row, raw, sites)
                    verify_phones(phones, service.phonemes(candidate["body"]), sites)
                view = observe(candidate, row)
                for label in by_label[row["uid"]]:
                    value = dict(label)
                    value["batched"] = label["candidate"]
                    value["candidate"] = view[label["q"]]
                    old_phone = label["candidate"]["phone"]
                    new_phone = value["candidate"]["phone"]
                    counts["alert_state_flips"] += bool(old_phone and old_phone.get("errorType")) != bool(new_phone and new_phone.get("errorType"))
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
        for suffix, values in (("sites", rows), ("score_audit", audits)):
            with (out / (name + "." + suffix + ".jsonl")).open("x") as stream:
                for value in values:
                    service.write_line(stream, value)
        result = detail(rows, counts)
        result.update({"eligible_clips": len(selected), "max_probability_difference": max(differences) if differences else None,
                       "single_request_ms_p50_p95": np.quantile(timings, [.5, .95]).tolist() if timings else None})
        report["cohorts"][name] = result
        print(name, json.dumps({"counts": dict(counts), "candidate": result["versus_prior"]["candidate"],
                               "max_probability_difference": result["max_probability_difference"]}, ensure_ascii=False), flush=True)
        write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


if __name__ == "__main__":
    main()

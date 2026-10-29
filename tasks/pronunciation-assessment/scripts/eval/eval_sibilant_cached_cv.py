"""Evaluate a frozen head on full runtime clip caches, with and without agreement.

No fitting or threshold adjustment. Cal-named stress cohorts are additional
checks, not a replacement for the artifact's original calibration provenance.
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
from train_sibilant_local_encoder import load_encoder, write
from train_sibilant_cv_support import score
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_cv_policy import annotate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    cache, out = Path(a.cache), Path(a.out)
    if not (cache / "finished.json").exists():
        raise ValueError("runtime cache incomplete")
    if json.loads((cache / "finished.json").read_text())["sha256"] != service.sha(cache / "report.json"):
        raise ValueError("runtime cache report changed")
    cache_report = json.loads((cache / "report.json").read_text())
    for name in p["cohorts"]:
        for suffix, digest in cache_report["cohorts"][name]["output_sha256"].items():
            if service.sha(cache / (name + suffix)) != digest:
                raise ValueError("runtime cache output changed: " + name + suffix)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    torch.set_num_threads(2)
    artifact = torch.load(p["artifact"], map_location="cpu")
    encoder = load_encoder(p, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    head = SupportReadout(encoder.config.hidden_size, True).to(next(encoder.parameters()).device)
    head.load_state_dict(artifact["head"])
    # Inference does not read TAG label files.
    for name in p["cohorts"]:
        probabilities, available, seconds = score(encoder, head, dict(np.load(cache / (name + ".npz"))))
        np.savez(out / (name + ".scores.npz"), probabilities=probabilities, available=available)
        print(name, "frozen model scored", len(probabilities), "seconds", seconds, flush=True)
    del encoder, head
    torch.cuda.empty_cache()
    app = app_from(p)
    app.S["dec"].model.to("cpu")
    report = {"scope": p["scope"], "thresholds": artifact["thresholds"], "fitting": False, "deployment": False, "cohorts": {}}
    for name in p["cohorts"]:
        records = service.read_jsonl(cache / (name + ".responses.jsonl"))
        labels = service.read_jsonl(cache / (name + ".labels.jsonl"))
        features = service.read_jsonl(cache / (name + ".features.jsonl"))
        scores = np.load(out / (name + ".scores.npz"))
        by_label, by_uid = collections.defaultdict(list), collections.defaultdict(dict)
        for label in labels:
            by_label[label["uid"]].append(label)
        for i, f in enumerate(features):
            by_uid[f["uid"]][f["q_runtime"]] = {"probabilities": scores["probabilities"][i].tolist(), "available": bool(scores["available"][i])}
        results = {}
        for arm, agreement in (("full", False), ("agreement", True)):
            rows, counts = [], collections.Counter()
            with (out / (name + "." + arm + ".responses.jsonl")).open("x") as stream:
                for record in records:
                    row, base = record["row"], record["prior"]
                    candidate = dict(base)
                    if base["raw"] is not None and by_uid[row["uid"]]:
                        raw, phones = base["raw"], service.phonemes(base["body"])
                        sites = restored_sites(raw, base["body"], p["pair_thresholds"])
                        sites, states = annotate(raw["canon"], sites, phones, by_uid[row["uid"]], artifact["thresholds"], agreement)
                        counts.update(states.values())
                        if "confirmed" in states.values():
                            candidate["body"] = render(app, row, raw, sites)
                            verify_phones(phones, service.phonemes(candidate["body"]), sites)
                    views = {"original": observe(record["original"], row), "reference": observe(record["existing"], row),
                             "prior": observe(base, row), "candidate": observe(candidate, row)}
                    for label in by_label[row["uid"]]:
                        value = dict(label)
                        value.update({key: view[label["q"]] for key, view in views.items()})
                        rows.append(value)
                    service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
            with (out / (name + "." + arm + ".sites.jsonl")).open("x") as stream:
                for row in rows:
                    service.write_line(stream, row)
            results[arm] = detail(rows, counts)
            print(name, arm, json.dumps(results[arm]["versus_prior"], ensure_ascii=False), flush=True)
        report["cohorts"][name] = results
        write(out / "report.json", report)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


if __name__ == "__main__":
    main()

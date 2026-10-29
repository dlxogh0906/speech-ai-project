"""Prepare matched pretrained/fine-tuned projections for the existing linear runner.

PCA uses training frames only. Both encoders get a newly fitted projection so
the representation comparison is not confounded by the earlier small-data PCA.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]

import argparse
import copy
import datetime
import json
import shutil
import time

import numpy as np
import torch

import eval_sibilant_directional as service
from train_sibilant_linear import frozen_load
from mdd.sibilant_hidden import fit_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--pretrained-features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    previous = json.loads(Path(protocol["previous_linear_protocol"]).read_text())
    if previous["C"] != .1 or protocol["planned_comparison"]["layer"] != 24:
        raise ValueError("unexpected fixed comparison configuration")
    if service.sha(protocol["previous_linear_protocol"]) != protocol["source_sha256"][protocol["previous_linear_protocol"]]:
        raise ValueError("previous protocol changed")
    fine_source = Path(previous["hidden_features"])
    old_metadata = json.loads((fine_source / "metadata.json").read_text())
    pretrained = Path(args.pretrained_features)
    fresh_metadata = json.loads((pretrained / "metadata.json").read_text())
    if fresh_metadata["protocol"] != protocol or fresh_metadata["limit"] != 0:
        raise ValueError("pretrained extraction protocol mismatch or smoke extraction")
    if not (pretrained / "finished.json").exists() or set(fresh_metadata["splits"]) != {"train", "cal", "test"}:
        raise ValueError("pretrained extraction incomplete")
    for split in ("train", "cal", "test"):
        if not np.load(pretrained / (split + ".filled.npy")).all():
            raise ValueError("missing pretrained observations")
    arrays, labels, features, _ = frozen_load(fine_source, old_metadata["protocol"], previous, "train")
    train = [r for r in labels if r["feature_index"] is not None]
    index = np.array([r["feature_index"] for r in train])
    if len(set(index.tolist())) != len(index):
        raise ValueError("duplicate training site observation")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(args.protocol, out / "comparison_protocol.json")
    report = {"test_read_for_pca": False, "train_sites": len(index), "encoders": {},
              "source_sha256": service.sha(__file__), "comparison_protocol_sha256": service.sha(args.protocol)}
    for encoder, source in [("finetuned_runtime", fine_source), ("pretrained_runtime", pretrained)]:
        tick = time.monotonic()
        output = out / encoder
        output.mkdir()
        cache = output / "hidden"
        cache.mkdir()
        metadata = copy.deepcopy(old_metadata)
        metadata["protocol"]["layers"] = [24]
        metadata["representation"] = encoder
        metadata["runtime_inputs"] = protocol["planned_comparison"]["localization"] + "; " + encoder
        metadata["limit"] = 0
        metadata["representation_source_metadata_sha256"] = service.sha(source / "metadata.json")
        for split in ("train", "cal", "test"):
            for suffix in (".h24.npy", ".filled.npy"):
                (cache / (split + suffix)).symlink_to(source / (split + suffix))
        (cache / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        shutil.copy2(source / "finished.json", cache / "finished.json")
        hidden = np.load(cache / "train.h24.npy", mmap_mode="r")
        if len(hidden) != len(features):
            raise ValueError("training hidden count mismatch")
        projection = fit_projection(hidden[index], arrays["mask"][index], components=64, seed=20261024)
        artifact = output / "projection.pt"
        torch.save({"projection": projection, "encoder": encoder, "training_sites": len(index)}, artifact)
        linear = copy.deepcopy(previous)
        linear.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      hidden_features=str(cache), projection_artifact=str(artifact),
                      projection_artifact_sha256=service.sha(artifact),
                      arms=["h24_temporal_linear"], primary_arm="h24_temporal_linear",
                      encoder=encoder, C=.1,
                      hypothesis=protocol["hypothesis"],
                      comparison_protocol=str(Path(args.protocol).resolve()),
                      comparison_protocol_sha256=service.sha(args.protocol),
                      projection_fit="PCA64 newly fit on same expanded train sites for each encoder, not old773-site projection",
                      limits=protocol["limits"] + ["reused exploratory calibration/test; no independent human validation"])
        linear["hidden_sha256"] = {split: {str(cache / (split + suffix)): service.sha(cache / (split + suffix))
                                           for suffix in (".h24.npy", ".filled.npy")} for split in ("train", "cal", "test")}
        linear["adoption"] = "Pretrained representation must improve correct-direction API error detection without added normal alerts; confirm beyond this corpus and measure extra encoder cost before adoption."
        (output / "linear_protocol.json").write_text(json.dumps(linear, ensure_ascii=False, indent=2) + "\n")
        report["encoders"][encoder] = {"training_frames": int(projection["training_frames"]),
                                         "explained_variance_sum": float(projection["explained_variance_ratio"].sum()),
                                         "linear_protocol_sha256": service.sha(output / "linear_protocol.json"),
                                         "seconds": time.monotonic() - tick}
        print(encoder, report["encoders"][encoder], flush=True)
    (out / "finished.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

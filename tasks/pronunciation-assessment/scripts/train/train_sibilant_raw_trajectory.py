"""Reuse the exact matched C-trajectory experiment with observed fbank branches.

Adapters are local to this process. Both the guard/training path and the shared
batched scorer receive (encoder representation, observed fbank), never labels.
The original readout control discards the fbank member. No service is modified.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import train_sibilant_c_trajectory as experiment
import argparse
import json
import shutil
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_raw_trajectory import RawTrajectoryReadout


class ExistingControl(SupportReadout):
    def forward(self, observed, mask, relative):
        return super().forward(observed[0], mask, relative)


class RawBranch(RawTrajectoryReadout):
    def forward(self, observed, mask, relative):
        return super().forward(observed[0], mask, relative, observed[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    original_hidden = experiment.hidden

    def observed_hidden(encoder, x, mask):
        return original_hidden(encoder, x, mask), x

    experiment.SupportReadout = ExistingControl
    experiment.CTrajectoryReadout = RawBranch
    experiment.hidden = observed_hidden
    experiment.trainer.hidden = observed_hidden
    experiment.main()
    # The shared experiment's numeric layout is reused, but raw-fbank artifacts
    # must never claim to be encoder-trajectory artifacts.
    out = Path(args.out)
    shutil.copy2(__file__, out / Path(__file__).name)
    report = json.loads((out / "report.json").read_text())
    report["branch_input"] = "observed normalized160-dimensional fbank, not encoder representation"
    for arm in ("mean", "drift"):
        artifact = torch.load(out / (arm + ".pt"), map_location="cpu")
        artifact["readout"] = "raw_c_trajectory_" + arm
        torch.save(artifact, out / (arm + ".pt"))
        report["arms"][arm]["artifact_sha256"] = experiment.trainer.service.sha(out / (arm + ".pt"))
    experiment.write(out / "report.json", report)
    experiment.write(out / "finished.json", {"sha256": experiment.trainer.service.sha(out / "report.json")})


if __name__ == "__main__":
    main()

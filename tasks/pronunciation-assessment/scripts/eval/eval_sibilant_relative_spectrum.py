"""Run the unchanged API evaluator with a typed normalized-C/V readout.

Model/score adapters are process-local. Acoustic inputs are exactly x, observed
lengths and runtime relative position. The normalized skip branch sees x itself,
not TAG boundaries, actual labels or a different audio window.
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
import shutil
import time
import numpy as np
import torch
import eval_sibilant_context_rank as evaluator
from train_sibilant_local_encoder import load_encoder, hidden
from mdd.sibilant_cv_support import reliable_support
from mdd.sibilant_relative_spectrum import RelativeSpectrumReadout

original_model = evaluator.model


def model(protocol, artifact):
    if artifact.get("readout") == "existing":
        return original_model(protocol, artifact)
    if artifact.get("readout") != "relative_spectrum" or artifact.get("acoustic_source") != "normalized":
        raise ValueError("only typed existing or normalized-relative models are supported")
    encoder = load_encoder(protocol, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact["last2"])
    state = artifact["head"]
    head = RelativeSpectrumReadout(encoder.config.hidden_size, artifact["branch"],
                                  state["acoustic_mean"], state["acoustic_scale"]).to(next(encoder.parameters()).device)
    head.load_state_dict(state)
    if not torch.equal(head.acoustic_mean, torch.zeros_like(head.acoustic_mean)) or not torch.equal(head.acoustic_scale, torch.ones_like(head.acoustic_scale)):
        raise ValueError("normalized input branch must use identity acoustic scaling")
    return encoder, head


def score_single(encoder, head, inputs):
    if set(inputs) != {"x", "lengths", "relative"}:
        raise ValueError("inference accepts observed acoustics and runtime position only")
    probabilities, available = [], []
    device, started = next(encoder.parameters()).device, time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(len(inputs["x"])):
            x = torch.as_tensor(inputs["x"][i:i + 1], device=device)
            lengths = torch.as_tensor(inputs["lengths"][i:i + 1], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(inputs["relative"][i:i + 1], device=device)
            h = hidden(encoder, x, mask)
            logits, roles = (head(h, mask, relative, x) if isinstance(head, RelativeSpectrumReadout)
                             else head(h, mask, relative))
            probabilities.append(logits.softmax(-1)[0].cpu().numpy())
            available.append(bool(reliable_support(roles, mask)[0]))
            if (i + 1) % 1000 == 0:
                print("single relative requests", i + 1, "seconds", round(time.monotonic() - started, 1), flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    print("single relative complete", len(probabilities), "seconds", seconds,
          "peak_gpu_bytes", torch.cuda.max_memory_allocated(), flush=True)
    return np.asarray(probabilities), np.asarray(available), seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    ap.add_argument("--calibration")
    args = ap.parse_args()
    evaluator.model = model
    evaluator.score_single = score_single
    evaluator.main()
    shutil.copy2(__file__, Path(args.out) / Path(__file__).name)


if __name__ == "__main__":
    main()

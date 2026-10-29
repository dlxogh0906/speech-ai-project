"""Transfer changed encoder tensors and verify the reconstructed full state.

Portable utility for isolated NPU experiment directories. A tensor fingerprint
avoids depending on torch.save container differences across torch versions.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists() and _r.parent != _r:
    _r = _r.parent
if (_r / "pyproject.toml").exists():
    _sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("export", "materialize"))
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--delta", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--unfreeze-from", type=int, default=16)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    base = torch.load(args.base, map_location="cpu")
    if args.mode == "export":
        if args.candidate is None:
            ap.error("export requires --candidate")
        candidate = torch.load(args.candidate, map_location="cpu")
        if base.keys() != candidate.keys():
            raise ValueError("Encoder keys differ")
        changed = {}
        for key in base:
            if not torch.equal(base[key], candidate[key]):
                if not key.startswith("encoder.layers.") or int(key.split(".")[2]) < args.unfreeze_from:
                    raise ValueError("Frozen tensor changed: " + key)
                changed[key] = candidate[key]
        payload = {"format": "encoder_tensor_delta_v1", "base_fingerprint": fingerprint(base),
                   "candidate_fingerprint": fingerprint(candidate), "tensors": changed}
        torch.save(payload, args.delta)
        print("[export] {} tensors, {:.1f} MB".format(len(changed), args.delta.stat().st_size / 1e6), flush=True)
    else:
        if args.output is None:
            ap.error("materialize requires --output")
        payload = torch.load(args.delta, map_location="cpu", weights_only=False)
        if payload["format"] != "encoder_tensor_delta_v1" or fingerprint(base) != payload["base_fingerprint"]:
            raise ValueError("Delta source encoder does not match")
        if set(payload["tensors"]) - set(base):
            raise ValueError("Unknown encoder tensor")
        base.update(payload["tensors"])
        if fingerprint(base) != payload["candidate_fingerprint"]:
            raise ValueError("Reconstructed encoder differs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".materializing.pt")
        torch.save(base, temporary)
        temporary.replace(args.output)
        args.output.with_suffix(".tensor_fingerprint.json").write_text(json.dumps(
            {k: payload[k] for k in ("format", "base_fingerprint", "candidate_fingerprint")}, indent=2))
        print("[materialized] " + str(args.output), flush=True)


if __name__ == "__main__":
    main()

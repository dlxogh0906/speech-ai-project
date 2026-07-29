#!/usr/bin/env python3
"""Create a compact fairseq XLS-R checkpoint for inference.

The original training checkpoint contains roughly twice as much optimizer state
as model weights. This keeps the small fairseq compatibility metadata while
removing the large last_optimizer_state tensor tree.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MODEL_ROOT


import argparse
import json
import time
from pathlib import Path

import torch


REQUIRED_KEYS = ("cfg", "model", "task_state")
REMOVE_KEYS = ("last_optimizer_state",)


def tensor_bytes(value, seen=None):
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(tensor_bytes(item, seen) for item in value)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(f"{MODEL_ROOT}/xlsr2_2B_1000k.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"{DATA_ROOT}/models/xlsr2_2B_1000k_inference.pt"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists; pass --force to replace it")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    checkpoint = torch.load(
        str(args.input),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    load_sec = time.perf_counter() - load_started

    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(f"checkpoint is missing required keys: {missing}")

    inference_checkpoint = {
        key: value for key, value in checkpoint.items() if key not in REMOVE_KEYS
    }
    save_started = time.perf_counter()
    torch.save(inference_checkpoint, args.output)
    save_sec = time.perf_counter() - save_started

    source_size = args.input.stat().st_size
    output_size = args.output.stat().st_size
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_size_gib": source_size / 1024**3,
        "output_size_gib": output_size / 1024**3,
        "size_reduction_percent": 100.0 * (1.0 - output_size / source_size),
        "kept_keys": list(inference_checkpoint),
        "removed_keys": sorted(set(checkpoint) - set(inference_checkpoint)),
        "model_tensor_gib": tensor_bytes(checkpoint["model"]) / 1024**3,
        "load_sec": load_sec,
        "save_sec": save_sec,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

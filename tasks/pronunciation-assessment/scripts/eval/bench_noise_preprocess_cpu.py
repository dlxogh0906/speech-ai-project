# -*- coding: utf-8 -*-
"""CPU latency only, using synthetic audio; can run in an isolated directory."""

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--helper", required=True)
    ap.add_argument("--silero-model", required=True)
    ap.add_argument("--rnnoise-library", required=True)
    ap.add_argument("--repeat", type=int, default=15)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("noise_preprocess", Path(args.helper))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    processors = {"silero": helper.SileroEdges(args.silero_model),
                  "rnnoise": helper.RNNoise16k(args.rnnoise_library)}
    results = []
    for seconds in (2, 5, 10):
        t = np.arange(seconds * 16000, dtype=np.float32) / 16000
        envelope = np.maximum(0, np.sin(2 * np.pi * 1.5 * t))
        wav = (0.10 * envelope * (np.sin(2 * np.pi * 145 * t)
               + 0.5 * np.sin(2 * np.pi * 290 * t))
               + 0.004 * np.random.RandomState(seconds).randn(len(t))).astype(np.float32)
        for name, processor in processors.items():
            processor(wav)
            timings = []
            for _ in range(args.repeat):
                out, info = processor(wav)
                assert np.isfinite(out).all() and 0 < len(out) <= len(wav)
                timings.append(info["preprocess_ms"])
            results.append({"method": name, "seconds": seconds,
                            "p50_ms": float(np.median(timings)),
                            "p95_ms": float(np.percentile(timings, 95))})
    print(json.dumps({"host": platform.node(), "python": platform.python_version(),
                      "scope": "synthetic audio, CPU preprocessor only, not accuracy/end-to-end latency",
                      "repeat": args.repeat, "results": results}, indent=2))


if __name__ == "__main__":
    main()

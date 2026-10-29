"""Inspect the current model's validation manifests and prediction caches."""

import argparse
import collections
import csv
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    if args.runtime:
        import onnxruntime as ort
        import torch
        print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "cuda_available": torch.cuda.is_available(), "onnxruntime": ort.__version__,
                          "providers": ort.get_available_providers(), "build": ort.get_build_info()}))
        session = ort.InferenceSession(str(args.root / "run_zeroth/deploy/encoder_kd_l17.onnx"),
                                       providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                                                  "CPUExecutionProvider"])
        print(json.dumps({"active_session_providers": session.get_providers()}))
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv"])
        return
    root = args.root
    history = json.loads((root / "run_zeroth/history.json").read_text())
    data_dir = Path(history["cfg"]["data_dir"])
    print(json.dumps({"training_config": history["cfg"],
                      "last_validation": history["evals"][-1]}, ensure_ascii=False))
    for split in ("train", "valid", "test", "test_koraug_zeroth"):
        path = data_dir / (split + ".csv")
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        print(json.dumps({"manifest": str(path), "n": len(rows),
                          "fields": list(rows[0]), "example": rows[0],
                          "domains": dict(collections.Counter(r["domain"] for r in rows)),
                          "speakers": len(set(r["speaker"] for r in rows)),
                          "countries": len(set(r["label_en"] for r in rows))}, ensure_ascii=False))
    for path in sorted(root.glob("eval*/report.json")):
        report = json.loads(path.read_text())
        if "valid" in report.get("test", "") or path.parent.name == "eval_zeroth":
            print(json.dumps({"report": str(path), "input": report.get("test"),
                              "n": report.get("n"), "models": report.get("models"),
                              "overview": {name: {key: value for key, value in metrics.items()
                                                 if key in ("web_all", "web_sent", "web_word", "korean", "p505_labeled")}
                                           for name, metrics in report.get("results", {}).items()}}, ensure_ascii=False))
    for path in sorted((root / "run_zeroth").glob("*")):
        print(json.dumps({"file": str(path), "bytes": path.stat().st_size}))
    for path in sorted(root.glob("**/*valid*.npz")):
        arrays = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _, dtype = np.lib.format._read_array_header(stream, version)
                    arrays[name] = {"shape": list(shape), "dtype": str(dtype)}
        print(json.dumps({"cache": str(path), "arrays": arrays}))


if __name__ == "__main__":
    main()

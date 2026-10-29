# -*- coding: utf-8 -*-
"""Run the existing ONNX evaluator with this interpreter's CUDA libraries."""

import os
from pathlib import Path
import re
import runpy
import sys
import sysconfig
import tempfile


def main():
    if "--help" not in sys.argv and not os.environ.get("NAT_VALIDATION_CUDA_LIBS_READY"):
        packages = Path(sysconfig.get_paths()["purelib"])
        torch_lib = packages / "torch/lib"
        library_dirs = [torch_lib] + sorted((packages / "nvidia").glob("*/lib"))
        aliases = Path(tempfile.mkdtemp(prefix="nat_validation_cuda_"))
        for source in torch_lib.glob("*.so*"):
            name = re.sub(r"-[a-f0-9]+(?=\.so)", "", source.name)
            if name != source.name:
                (aliases / name).symlink_to(source)
                unversioned = name.split(".so")[0] + ".so"
                if not (aliases / unversioned).exists():
                    (aliases / unversioned).symlink_to(source)
        environment = os.environ.copy()
        environment["NAT_VALIDATION_CUDA_LIBS_READY"] = "1"
        environment["LD_LIBRARY_PATH"] = ":".join(
            [str(aliases)] + [str(path) for path in library_dirs if path.exists()]
            + ([environment["LD_LIBRARY_PATH"]] if environment.get("LD_LIBRARY_PATH") else []))
        os.execve(sys.executable, [sys.executable, "-u", str(Path(__file__).resolve())] + sys.argv[1:], environment)
    if "--check-runtime" in sys.argv:
        import onnxruntime as ort
        import numpy as np
        model = sys.argv[sys.argv.index("--check-runtime") + 1]
        session = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print("Active providers:", session.get_providers(), flush=True)
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("GPU provider did not load")
        output = session.run(["pooled"], {"source": np.zeros((1, 16000), np.float32)})[0]
        print("GPU inference output shape:", output.shape, flush=True)
        return
    target = Path(__file__).resolve().parents[2] / "experiments/webmic_kd_eval.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

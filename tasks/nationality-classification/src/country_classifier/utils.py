"""Shared helpers for reproducible training and JSON metadata."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_label_map(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "id_to_label" in data:
        id_to_label = data["id_to_label"]
        labels = [id_to_label[str(idx)] for idx in range(len(id_to_label))]
    elif "label_to_id" in data:
        label_to_id = data["label_to_id"]
        labels = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    else:
        raise ValueError(f"{path} does not contain id_to_label or label_to_id")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return labels, label_to_id, data

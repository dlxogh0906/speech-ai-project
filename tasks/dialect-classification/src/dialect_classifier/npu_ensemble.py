# -*- coding: utf-8 -*-
"""NPU 번들에서 사투리 앙상블을 돌린다 (2026-10-28) — `ensemble.DialectEnsemble` 과 같은 `.score()` 계약.

번들 모델 폴더(`scripts/export/export_npu_bundle.py` 가 만든다):
  meta.json                     규칙 · 라벨 · 파일 이름 · 만든 근거
  <kind>/pure_fp16.pt           순수 인코더 가중치 (npu_models.build(kind) 와 같은 이름)
  <kind>/head.npz               weight · bias · mu · sd (+ eat_large 는 mel_banks)
  <kind>/<kind>_static12s.rbln  NPU 컴파일 결과 (`scripts/export/compile_rbln.py`)

backend='rbln' 은 NPU(rebel 런타임), backend='torch' 는 같은 순수 모듈을 CPU/GPU 에서 돌린다(대조용).
NPU 호출은 한 프로세스 안에서 스레드끼리 병렬로 돌지 않으므로 잠금 하나로 차례대로 부른다.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np

from .ensemble import KINDS, LABELS, MAX_SEC, MIN_SEC, SR


def softmax(logits):
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def combine(rule, a, b):
    """scripts/eval/score_sealed_test.candidates() 의 후보 중 `rule` 하나만 계산한다(번들에 scripts/ 가 없어서 옮겨 적었다)."""
    if rule == "eat_large":
        return a
    if rule == "xlsr2b":
        return b
    head, alpha = rule.rsplit("_", 1)
    alpha = float(alpha)
    if head == "probability_eat":
        return alpha * a + (1 - alpha) * b
    assert head == "eat_standard_gate_regional_eat", rule
    out = np.zeros_like(a)
    out[:, :5] = alpha * a[:, :5] / a[:, :5].sum(1, keepdims=True) + (1 - alpha) * b[:, :5] / b[:, :5].sum(1, keepdims=True)
    gate = a.argmax(1) == LABELS.index("Standard")
    out[gate, :5] = 0
    out[gate, 5] = 1
    return out


class NpuEnsemble:
    def __init__(self, root, backend="rbln", devices=None, torch_device=None):
        from .npu_models import Head
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        assert self.meta["labels"] == LABELS, "번들 라벨 순서가 다르다"
        self.rule, self.backend = self.meta["rule"], backend
        self.lock = threading.Lock()
        self.heads = {k: Head(np.load(str(self.root / k / "head.npz"))) for k in KINDS}
        self.mel_banks = np.load(str(self.root / "eat_large" / "head.npz"))["mel_banks"]
        self.run = {}
        if backend == "rbln":
            import rebel
            devices = list(devices) if devices else [0, 0]
            for kind, dev in zip(KINDS, devices):
                rt = rebel.Runtime(str(self.root / kind / self.meta["files"][kind]["rbln"]), device=int(dev), tensor_type="np")
                self.run[kind] = lambda a, b, rt=rt: np.asarray(rt.run(a, b), np.float32)
            self.device = "rbln:" + ",".join(str(d) for d in devices)
        else:
            import torch
            from .npu_models import load_pure
            dev = torch.device(torch_device or ("cuda" if torch.cuda.is_available() else "cpu"))
            for kind in KINDS:
                model = load_pure(kind, self.root / kind / self.meta["files"][kind]["state"]).to(dev)

                def run(a, b, model=model):
                    with torch.no_grad():
                        return model(torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)).float().cpu().numpy()
                self.run[kind] = run
            self.device = str(dev)

    def pooled(self, kind, wav):
        from .npu_models import inputs
        a, b = inputs(kind, wav, self.mel_banks)
        return self.run[kind](a, b)

    def score(self, windows):
        """windows: 16 kHz mono float32 구간 목록(1~12초) → 모델별 · 규칙의 구간별 확률."""
        for w in windows:
            if not MIN_SEC <= len(w) / SR <= MAX_SEC + 1e-3:
                raise ValueError("window must be %.0f-%.0f s, got %.2f s" % (MIN_SEC, MAX_SEC, len(w) / SR))
        t0 = time.perf_counter()
        with self.lock:
            logits = {k: np.concatenate([self.heads[k](self.pooled(k, w)) for w in windows]) for k in KINDS}
        probs = {k: softmax(logits[k]) for k in KINDS}
        probs["selected"] = combine(self.rule, probs["eat_large"], probs["xlsr2b"])
        probs["encode_ms"] = (time.perf_counter() - t0) * 1000
        return probs

"""Classification metrics with exact sample-level top-k accounting."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import torch


class TopKAccuracy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total = 0
        self.top1_correct = 0
        self.top5_correct = 0

    @torch.no_grad()
    def update(self, logits, labels):
        if labels.numel() == 0:
            return
        k = min(5, self.num_classes)
        topk = logits.topk(k=k, dim=-1).indices
        labels_col = labels.view(-1, 1)
        self.top1_correct += logits.argmax(dim=-1).eq(labels).sum().item()
        self.top5_correct += topk.eq(labels_col).any(dim=1).sum().item()
        self.total += labels.numel()

    def compute(self):
        if self.total == 0:
            return {"top1_acc": 0.0, "top5_acc": 0.0, "num_samples": 0}
        return {
            "top1_acc": self.top1_correct / self.total,
            "top5_acc": self.top5_correct / self.total,
            "num_samples": self.total,
        }


@torch.no_grad()
def metrics_from_logits(logits, labels):
    meter = TopKAccuracy(logits.size(-1))
    meter.update(logits, labels)
    return meter.compute()

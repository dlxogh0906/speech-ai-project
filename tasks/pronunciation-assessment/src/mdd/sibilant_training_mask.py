"""Train-only classification weights; no audio, TAG label or service change."""
import torch


def masked_actual_loss(logits, labels, valid):
    if valid.shape != labels.shape or valid.dtype != torch.bool:
        raise ValueError('one boolean train supervision mask per class label required')
    if bool(valid.all()):
        # Keep the original operation exactly when every label is supported.
        return torch.nn.functional.cross_entropy(logits, labels)
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    terms = []
    for k in range(4):
        selected = (labels == k) & valid
        terms.append(losses[selected].mean() if bool(selected.any()) else logits.sum() * 0)
    return torch.stack(terms).mean()

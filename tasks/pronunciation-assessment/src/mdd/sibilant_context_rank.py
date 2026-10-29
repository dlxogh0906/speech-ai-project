"""Compare two real training recordings at the same phonemic item position.

The sampler orders four actual-S recordings, then their four actual-SS mates,
then affricate/other recordings. Context is used only to construct training
pairs; inference continues to predict actual phone from acoustic inputs alone.
"""
import torch
import torch.nn.functional as F


def context_rank_loss(logits, actual):
    if logits.shape != (16, 4) or actual.shape != (16,):
        raise ValueError("expected the fixed sixteen-example four-class batch")
    if not (bool((actual[:4] == 0).all()) and bool((actual[4:8] == 1).all())
            and bool((actual[8:12] == 2).all()) and bool((actual[12:] == 3).all())):
        raise ValueError("paired actual-S/SS batch order changed")
    evidence = logits[:, 1] - logits[:, 0]
    return F.softplus(-(evidence[4:8] - evidence[:4])).mean()

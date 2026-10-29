"""Small frozen-backbone temporal head; offline pilot, no serving registration."""
import numpy as np
import torch
from torch import nn

from mdd.sibilant_transition import normalize_acoustics


class TransitionHead(nn.Module):
    def __init__(self, input_dim, context_dim, width=48):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv1d(input_dim, width, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.classifier = nn.Linear(width * 3 + context_dim, 2)

    def forward(self, sequence, mask, context):
        valid = mask[:, None, :].to(sequence.dtype)
        hidden = self.temporal(sequence.transpose(1, 2) * valid)
        bounds = (0, 10, 21, 31)
        pooled = [(hidden[:, :, a:b] * valid[:, :, a:b]).sum(-1) /
                  valid[:, :, a:b].sum(-1).clamp_min(1)
                  for a, b in zip(bounds[:-1], bounds[1:])]
        return self.classifier(torch.cat(pooled + [context], dim=1))


def input_tensors(arrays, normalization, acoustic):
    sequence = arrays["posterior"].astype(np.float32)
    if acoustic:
        sequence = np.concatenate([sequence, normalize_acoustics(arrays["acoustic"], normalization)], axis=-1)
    return (torch.from_numpy(sequence), torch.from_numpy(arrays["mask"].astype(bool)),
            torch.from_numpy(arrays["context"].astype(np.float32)))


@torch.inference_mode()
def predict(model, tensors, batch_size=256):
    model.eval()
    return np.concatenate([model(*(v[start:start + batch_size] for v in tensors)).softmax(-1)[:, 1].numpy()
                           for start in range(0, len(tensors[0]), batch_size)])

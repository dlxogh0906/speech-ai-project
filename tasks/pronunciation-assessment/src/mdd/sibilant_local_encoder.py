"""Temporal readout for an inactive local-context encoder candidate."""
import torch


class TemporalReadout(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3 * hidden_size, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2))

    def forward(self, hidden, mask):
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.any(1).all():
            raise ValueError("nonempty observed-frame masks required")
        position = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        lengths = mask.long().sum(1, keepdim=True)
        bins = (3 * position // lengths).clamp(max=2)
        values = []
        for phase in range(3):
            valid = mask.bool() & (bins == phase)
            if not valid.any(1).all():
                raise ValueError("three observed temporal bins required")
            values.append((hidden.float() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True))
        return self.net(torch.cat(values, dim=-1))

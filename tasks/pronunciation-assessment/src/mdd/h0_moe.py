"""Dense, audio-conditioned residual experts for a bounded H0 MoE pilot.

All experts execute. This is not sparse top-k routing or a compute-saving claim.
The forward interface accepts acoustic representations and lengths only.
"""
import torch


class ResidualExperts(torch.nn.Module):
    def __init__(self, width=1024, bottleneck=256, experts=4, routed=True):
        super().__init__()
        self.routed = routed
        self.experts_count = experts
        self.middle_norm = torch.nn.LayerNorm(width)
        self.final_norm = torch.nn.LayerNorm(width)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(width, bottleneck), torch.nn.GELU(),
                                torch.nn.Linear(bottleneck, width)) for _ in range(experts)])
        self.router = torch.nn.Linear(2 * width, experts)
        torch.nn.init.zeros_(self.router.weight)
        torch.nn.init.zeros_(self.router.bias)
        for expert in self.experts:
            torch.nn.init.zeros_(expert[-1].weight)
            torch.nn.init.zeros_(expert[-1].bias)
        self.router.requires_grad_(routed)
        self.final_norm.requires_grad_(routed)

    def forward(self, middle, final, lengths, force_uniform=False):
        z = self.middle_norm(middle)
        if self.routed and not force_uniform:
            logits = self.router(torch.cat((z, self.final_norm(final)), dim=-1))
            gates = logits.float().softmax(-1)
        else:
            gates = final.new_full((*final.shape[:2], self.experts_count),
                                   1. / self.experts_count, dtype=torch.float32)
        residual = torch.zeros_like(final, dtype=torch.float32)
        for k, expert in enumerate(self.experts):
            residual = residual + gates[..., k:k+1] * expert(z).float()
        mask = torch.arange(final.shape[1], device=final.device)[None] < lengths[:, None]
        valid = gates[mask]
        if not valid.numel():
            raise ValueError('no valid acoustic frames')
        # CV squared of mean soft routing mass; padding contributes nothing.
        mean = valid.sum(0) / valid.shape[0]
        # Equivalent CV squared, avoiding cancellation around the uniform point.
        balance = self.experts_count * (mean - 1. / self.experts_count).square().sum()
        return final + residual.to(final.dtype), balance, gates


def routing_counts(gates, lengths, predictions=None):
    """Additive diagnostics; tie-broken top1 at uniform initialization is arbitrary."""
    with torch.no_grad():
        mask = torch.arange(gates.shape[1], device=gates.device)[None] < lengths[:, None]
        if predictions is not None:
            mask = mask & (predictions != 0)
        g = gates[mask].float()
        return dict(frames=len(g), mass=g.sum(0).cpu().tolist(),
                    square_mass=g.square().sum(0).cpu().tolist(),
                    top1=torch.bincount(g.argmax(-1), minlength=gates.shape[-1]).cpu().tolist(),
                    entropy_sum=float(-(g * g.clamp_min(1e-12).log()).sum()))

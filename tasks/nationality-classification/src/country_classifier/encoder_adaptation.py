"""Small adapters and frozen-prefix inference for the deployed pure encoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear projection plus a trainable, initially zero update."""

    def __init__(self, base, rank=8, alpha=8.0):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, value):
        return self.base(value) + F.linear(F.linear(value, self.A), self.B) * self.scale

    def merged_weight(self):
        return self.base.weight + (self.B @ self.A) * self.scale


def attach_lora(encoder, from_layer=16, rank=8, alpha=8.0):
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    paths = []
    for index in range(from_layer, len(encoder.encoder.layers)):
        attention = encoder.encoder.layers[index].self_attn
        for name in ("q_proj", "v_proj"):
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise ValueError("LoRA requires an ordinary linear projection")
            setattr(attention, name, LoRALinear(base, rank, alpha))
            paths.append("encoder.layers.{}.self_attn.{}".format(index, name))
    return paths


def merged_encoder_state(encoder, adapter_paths):
    """Export ordinary PureEncoder keys; no adapter branch is needed at runtime."""
    state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    for path in adapter_paths:
        index, name = int(path.split(".")[2]), path.rsplit(".", 1)[1]
        adapter = getattr(encoder.encoder.layers[index].self_attn, name)
        for key in ("A", "B", "base.weight", "base.bias"):
            state.pop(path + "." + key, None)
        state[path + ".weight"] = adapter.merged_weight().detach().cpu().clone()
        if adapter.base.bias is not None:
            state[path + ".bias"] = adapter.base.bias.detach().cpu().clone()
    return state


@torch.no_grad()
def frozen_prefix(encoder, source, valid, from_layer=16):
    value = source.unsqueeze(1)
    for sequence in encoder.feature_extractor.conv_layers:
        value = F.gelu(sequence[2][1](sequence[0](value).transpose(1, 2)).transpose(1, 2))
    value = encoder.post_extract_proj(encoder.layer_norm(value.transpose(1, 2)))
    value = value * valid.unsqueeze(-1)
    position = encoder.encoder.pos_conv[0](value.transpose(1, 2))[:, :, :-1]
    value = value + F.gelu(position).transpose(1, 2)
    bias = ((valid - 1) * 1e4)[:, None, None, :]
    for layer in list(encoder.encoder.layers)[:from_layer]:
        value = layer(value, bias)
    return value


def suffix_features(encoder, value, valid, mu, sd, from_layer=16):
    bias = ((valid - 1) * 1e4)[:, None, None, :]
    for layer in list(encoder.encoder.layers)[from_layer:]:
        value = layer(value, bias)
    mask = valid.unsqueeze(-1)
    pooled = (value * mask).sum(1) / mask.sum(1).clamp_min(1)
    return (pooled.clamp(-50, 50) - mu) / sd


def probability_consistency(clean_logits, altered_logits):
    """Symmetric Jensen-Shannon loss; both views retain student gradients."""
    log_clean = F.log_softmax(clean_logits, dim=-1)
    log_altered = F.log_softmax(altered_logits, dim=-1)
    clean, altered = log_clean.exp(), log_altered.exp()
    log_mean = ((clean + altered) * 0.5).clamp_min(1e-8).log()
    return ((clean * (log_clean - log_mean)).sum(-1).mean()
            + (altered * (log_altered - log_mean)).sum(-1).mean()) * 0.5

"""TRAIN-only interventions for separating context and normalization sensitivity.

Annotation-selected protected spans are diagnostic/training controls, never a
service localization method or physical frication/glottal boundary truth.
"""
from contextlib import contextmanager
import numpy as np
import torch


def attenuate_outside(waveform, lo, hi, gain=0., fade=80):
    """Keep [lo, hi) bitwise unchanged; fade only outside that protected span."""
    x = np.asarray(waveform)
    if (x.ndim != 1 or x.dtype != np.float32 or not np.isfinite(x).all()
            or not 0 <= lo < hi <= len(x) or not 0 <= gain <= 1 or fade < 0):
        raise ValueError('finite float32 waveform and valid protected span/gain required')
    envelope = np.full(len(x), gain, np.float32)
    envelope[lo:hi] = 1
    n = min(fade, lo)
    if n:
        envelope[lo-n:lo] = gain + (1-gain) * (.5-.5*np.cos(np.pi*np.arange(n)/n))
    n = min(fade, len(x)-hi)
    if n:
        envelope[hi:hi+n] = gain + (1-gain) * (.5+.5*np.cos(np.pi*np.arange(n)/n))
    result = x * envelope
    if not np.array_equal(result[lo:hi], x[lo:hi]):
        raise ValueError('protected samples changed')
    return result


class OriginalGroupNorm:
    """Hold original group statistics, anchored to native output exactly.

    Reference input/output are captured on the unmodified waveform. Replaying
    output + (new_input-reference_input)*inverse_std*affine_scale keeps native
    output bitwise for identical input, without relying on matching CUDA
    reduction arithmetic in a separately computed group_norm implementation.
    This intervention does not alter any model weights or running state.
    """
    def __init__(self, module, value, output):
        if not isinstance(module, torch.nn.GroupNorm) or value.ndim != 3:
            raise ValueError('temporal GroupNorm required')
        self.module = module
        self.value, self.output = value.detach().clone(), output.detach().clone()
        b, c, t = value.shape
        grouped = value.detach().reshape(b, module.num_groups, -1)
        inverse = torch.rsqrt(grouped.var(-1, unbiased=False, keepdim=True) + module.eps)
        self.scale = inverse.expand(-1, -1, c // module.num_groups).reshape(b, c, 1)
        if module.affine:
            self.scale = self.scale * module.weight.detach()[None, :, None]

    def replace(self, module, args, output):
        if module is not self.module or args[0].shape != self.value.shape:
            raise ValueError('normalization reference module/coordinates differ')
        return self.output + (args[0] - self.value) * self.scale

    @contextmanager
    def hold(self):
        handle = self.module.register_forward_hook(self.replace)
        try:
            yield
        finally:
            handle.remove()


def capture_native_context(encoder, value):
    """Capture one native GroupNorm and convolution output on original audio."""
    normalizers = [m for m in encoder.modules() if isinstance(m, torch.nn.GroupNorm)]
    if len(normalizers) != 1:
        raise ValueError('expected exactly one native feature GroupNorm')
    capture = {}
    def norm_hook(module, args, output):
        capture['normalization'] = OriginalGroupNorm(module, args[0], output)
    def conv_hook(module, args, output):
        capture['convolution'] = output.detach().clone()
    handles = [normalizers[0].register_forward_hook(norm_hook),
               encoder.feature_extractor.register_forward_hook(conv_hook)]
    try:
        hidden = encoder(value, output_hidden_states=True).hidden_states
    finally:
        for handle in handles:
            handle.remove()
    return hidden, capture['normalization'], capture['convolution']


def forward_context_variant(encoder, value, normalization=None):
    """Return native hidden/convolution tracks, optionally with held statistics."""
    capture = {}
    def hook(module, args, output):
        capture['convolution'] = output.detach().clone()
    handle = encoder.feature_extractor.register_forward_hook(hook)
    try:
        if normalization is None:
            hidden = encoder(value, output_hidden_states=True).hidden_states
        else:
            with normalization.hold():
                hidden = encoder(value, output_hidden_states=True).hidden_states
    finally:
        handle.remove()
    return hidden, capture['convolution']

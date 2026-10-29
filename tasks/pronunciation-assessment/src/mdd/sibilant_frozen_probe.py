"""Capacity-matched frozen-feature probes with explicit backbone/projection binding."""
import hashlib
import numpy as np
import torch
from mdd.sibilant_vowel_presence import PresenceReadout

READOUT = 'frozen_projection_cv_v1'


def sampled_frames(features, lengths, maximum):
    """Hash observed frame identities only, without labels or padding."""
    choices = []
    for i, (feature, length) in enumerate(zip(features, lengths)):
        for frame in range(int(length)):
            identity = feature['uid'] + '/' + str(feature['q_runtime']) + '/' + str(frame)
            digest = hashlib.sha256(identity.encode()).digest()
            choices.append((digest, i, frame))
    choices.sort()
    return np.asarray([(i, frame) for _, i, frame in choices[:maximum]], np.int64)


def fit_projection(sample, dimensions, device):
    x = torch.as_tensor(sample, dtype=torch.float64, device=device)
    if x.ndim != 2 or not 0 < dimensions <= min(x.shape) or not bool(torch.isfinite(x).all()):
        raise ValueError('finite observed TRAIN frames and supported projection size required')
    mean = x.mean(0)
    scale = x.std(0, unbiased=False)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    z = (x - mean) / scale
    covariance = z.T @ z / max(1, len(z) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    components = vectors[:, -dimensions:].flip(1).contiguous()
    peaks = components.abs().argmax(0)
    signs = components[peaks, torch.arange(dimensions, device=device)].sign()
    components = components * torch.where(signs == 0, torch.ones_like(signs), signs)
    ratio = float(values[-dimensions:].clamp(min=0).sum() / values.clamp(min=0).sum().clamp(min=1e-20))
    return {'mean': mean.float().cpu(), 'scale': scale.float().cpu(),
            'components': components.float().cpu(), 'explained_variance_ratio': ratio}


def projection_digest(projection):
    digest = hashlib.sha256()
    for name in ('mean', 'scale', 'components'):
        a = torch.as_tensor(projection[name]).detach().cpu().float().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def project_observed(hidden, mask, projection):
    if hidden.shape[:2] != mask.shape or hidden.shape[-1] != len(projection['mean']):
        raise ValueError('projection/input coordinate or dimension mismatch')
    mean, scale, components = [projection[k].to(hidden.device, dtype=torch.float32)
                               for k in ('mean', 'scale', 'components')]
    result = torch.zeros((*hidden.shape[:2], components.shape[1]), device=hidden.device)
    # Do not multiply padding by zero: poisoned padding must never enter a
    # projection or subsequent temporal pooling.
    result[mask] = ((hidden[mask].float() - mean) / scale) @ components
    return result


def validate_binding(artifact, expected):
    if artifact.get('readout') != READOUT:
        raise ValueError('typed frozen-feature readout required')
    for key, value in expected.items():
        if artifact['binding'].get(key) != value:
            raise ValueError('frozen-feature binding mismatch: ' + key)
    if artifact['binding']['projection_sha256'] != projection_digest(artifact['projection']):
        raise ValueError('projection tensors do not match the trained head')


class FrozenProbe(torch.nn.Module):
    def __init__(self, projection):
        super().__init__()
        for key in ('mean', 'scale', 'components'):
            self.register_buffer(key, projection[key].clone())
        self.head = PresenceReadout(projection['components'].shape[1], True)

    def forward(self, hidden, mask, relative):
        projection = {key: getattr(self, key) for key in ('mean', 'scale', 'components')}
        return self.head(project_observed(hidden, mask, projection), mask, relative)


def load_probe(artifact, expected, device):
    validate_binding(artifact, expected)
    model = FrozenProbe(artifact['projection']).to(device).eval()
    model.head.load_state_dict(artifact['head'])
    return model

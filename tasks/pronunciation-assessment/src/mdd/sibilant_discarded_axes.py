"""Small residual observation model, preserving original paired probability mass.

Axes are orthogonal to the frozen PCA basis. Weak role means are observations,
not physical frication or glottal boundaries. Runtime never takes TAG or L1.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES

SCHEMA = 'native_discarded_C_earlyVcontrast_axes4_each_runtimeV4_v1'
READOUT = 'discarded_axes_pairmass_residual_cv_v1'
MODES = ('signal', 'permuted')
VOWELS = ('ㅣ', 'ㅡ', 'ㅜ', 'other')


def vowel_groups(features):
    return np.asarray([VOWELS.index(z['observed_V']) if z.get('observed_V') in VOWELS[:3] else 3
                       for z in features], np.int64)


def fit_axes(deltas, components, device):
    """Largest four cross-half eigenvectors per primary role, no matrix tuning."""
    q = torch.as_tensor(components, device=device, dtype=torch.float64)
    d = torch.as_tensor(deltas[:, :, :2], device=device, dtype=torch.float64)
    d = d-(d@q)@q.T
    axes, eigenvalues = [], []
    for role in range(2):
        a, b = d[:, 0, role], d[:, 1, role]
        matrix = (a.T@b+b.T@a)/(2*len(a))
        values, vectors = torch.linalg.eigh(matrix)
        if not bool((values[-4:] > 1e-9).all()):
            raise ValueError('fewer than four positive cross-half directions; no K adjustment')
        basis = vectors[:, -4:].flip(1)
        basis = basis-q@(q.T@basis)
        basis, _ = torch.linalg.qr(basis)
        peaks = basis.abs().argmax(0)
        signs = basis[peaks, torch.arange(4, device=device)].sign()
        basis = basis*torch.where(signs == 0, torch.ones_like(signs), signs)
        axes.append(basis.float().cpu())
        eigenvalues.append(values[-4:].flip(0).cpu().tolist())
    result = torch.stack(axes)
    error = max(float((torch.as_tensor(components, dtype=torch.float32).T@z).abs().max()) for z in result)
    if error > 2e-6:
        raise ValueError('new axes are not orthogonal to retained PCA')
    return result, {'top4_cross_half_eigenvalues': eigenvalues, 'PCA_axis_max_abs_dot': error}


def axis_observations(base, hidden, mask, phase, axes):
    weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2)*mask[:, None, :]
    weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
    z = torch.zeros_like(hidden, dtype=torch.float32)
    z[mask] = (hidden[mask].float()-base.mean)/base.scale
    means = torch.bmm(weights, z)
    full = torch.stack([means[:, 0], means[:, 1]-means[:, 0]], 1)
    return torch.einsum('brd,rdk->brk', full, axes).reshape(len(hidden), 8)


def redistribute_pair(logits, delta):
    """Change S/SS contrast; keep their total mass and other3 probabilities."""
    if logits.ndim != 2 or logits.shape[1] != 5 or delta.shape != (len(logits),):
        raise ValueError('actual-five logits and per-clip contrast required')
    g = logits[:, 1]-logits[:, 0]
    corrected = logits.clone()
    corrected[:, 0] += torch.nn.functional.logsigmoid(-g-delta)-torch.nn.functional.logsigmoid(-g)
    corrected[:, 1] += torch.nn.functional.logsigmoid(g+delta)-torch.nn.functional.logsigmoid(g)
    return corrected


def normalization_digest(normalization):
    return hashlib.sha256(json.dumps(normalization, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class DiscardedAxisResidual(torch.nn.Module):
    def __init__(self, base, axes, normalization, mode):
        super().__init__()
        if (mode not in MODES or tuple(axes.shape) != (2, len(base.mean), 4)
                or normalization.get('partition') != 'train' or normalization.get('schema') != SCHEMA):
            raise ValueError('typed two-role discarded axes required')
        self.mode = mode
        self.base = base.requires_grad_(False).eval()
        self.register_buffer('axes', torch.as_tensor(axes).float().clone())
        self.register_buffer('mean', torch.as_tensor(normalization['mean'], dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(normalization['scale'], dtype=torch.float32))
        if self.mean.shape != (8,) or self.scale.shape != (8,) or not bool((self.scale > 0).all()):
            raise ValueError('typed TRAIN axis normalization required')
        self.weight = torch.nn.Parameter(torch.zeros(4, 8))
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def from_observations(self, logits, raw, vowel):
        if raw.shape != (len(logits), 8) or vowel.shape != (len(logits),) or bool(((vowel < 0) | (vowel > 3)).any()):
            raise ValueError('runtime observed vowel group/axis observations differ')
        x = ((raw-self.mean)/self.scale).clamp(-8, 8)
        delta = (x*self.weight[vowel]).sum(-1)+self.bias[vowel]
        return redistribute_pair(logits, delta)

    def forward(self, hidden, mask, relative, vowel):
        logits, phase = self.base(hidden, mask, relative)
        raw = axis_observations(self.base, hidden, mask, phase, self.axes)
        return self.from_observations(logits, raw, vowel), phase


def axis_artifact(model, base_artifact, normalization):
    original = load_probe(base_artifact, base_artifact['binding'], 'cpu')
    if tensor_digest(model.base.state_dict()) != tensor_digest(original.state_dict()):
        raise ValueError('immutable parent changed')
    branch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('base.')}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact), 'branch': branch,
        'normalization': copy.deepcopy(normalization), 'binding': {'mode': model.mode,
            'schema': SCHEMA, 'classes': list(CLASSES), 'runtime_vowel_groups': list(VOWELS),
            'base_head_sha256': tensor_digest(base_artifact['head']), 'branch_sha256': tensor_digest(branch),
            'normalization_sha256': normalization_digest(normalization)}}


def load_axis_residual(artifact, expected_base, mode, device):
    binding = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or binding.get('mode') != mode
            or binding.get('schema') != SCHEMA or binding.get('classes') != list(CLASSES)
            or binding.get('runtime_vowel_groups') != list(VOWELS)
            or binding.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or binding.get('branch_sha256') != tensor_digest(artifact['branch'])
            or binding.get('normalization_sha256') != normalization_digest(artifact['normalization'])):
        raise ValueError('typed discarded-axis model binding differs')
    model = DiscardedAxisResidual(load_probe(artifact['base'], expected_base, device),
        artifact['branch']['axes'], artifact['normalization'], mode).to(device).eval()
    state = model.state_dict()
    expected = {k: v for k, v in state.items() if not k.startswith('base.')}
    if set(expected) != set(artifact['branch']) or any(
            artifact['branch'][k].shape != v.shape or artifact['branch'][k].dtype != v.dtype
            for k, v in expected.items()):
        raise ValueError('discarded-axis branch keys/shapes/dtypes differ')
    state.update(artifact['branch'])
    model.load_state_dict(state, strict=True)
    for key in ('mean', 'scale'):
        if not torch.equal(getattr(model, key).cpu(), torch.as_tensor(artifact['normalization'][key], dtype=torch.float32)):
            raise ValueError('normalization metadata differs from runtime buffers')
    return model

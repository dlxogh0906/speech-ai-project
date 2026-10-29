"""Typed low-band residual retaining an immutable native actual-phone head."""
import copy
import torch
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_pitchfree_spectrum import SCHEMA, BINS, normalization_digest
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'pitchfree_lowband_residual_cv_v1'


class SpectralResidual(torch.nn.Module):
    def __init__(self, base, mode):
        super().__init__()
        if mode not in ('fine', 'coarse_bands'):
            raise ValueError('typed spectral mode required')
        self.mode = mode
        self.base = base.requires_grad_(False).eval()
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv1d(BINS+3, 32, 5, padding=2),
            torch.nn.Conv1d(32, 32, 5, padding=4, dilation=2),
            torch.nn.Conv1d(32, 32, 5, padding=10, dilation=5)])
        self.output = torch.nn.Linear(64, len(CLASSES))
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, hidden, mask, relative, spectrum, spectral_mask):
        if (spectrum.ndim != 3 or spectrum.shape[-1] != BINS+3
                or spectrum.shape[:2] != spectral_mask.shape):
            raise ValueError('spectral grid/mask mismatch')
        logits, phase = self.base(hidden, mask, relative)
        # Failed measurements do not become deletion, nor a learned missing bias.
        z = spectrum.masked_fill(~spectral_mask[..., None], 0).transpose(1, 2)
        for layer in self.layers:
            z = torch.nn.functional.gelu(layer(z)).masked_fill(~spectral_mask[:, None], 0)
        count = spectral_mask.sum(-1, keepdim=True)
        mean = z.sum(-1)/count.clamp_min(1)
        maximum = z.masked_fill(~spectral_mask[:, None], -torch.inf).amax(-1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        residual = self.output(torch.cat([mean, maximum], -1))
        return logits+residual*(count > 0), phase


def spectral_artifact(model, base_artifact, normalization):
    if tensor_digest(model.base.state_dict()) != tensor_digest(
            load_probe(base_artifact, base_artifact['binding'], 'cpu').state_dict()):
        raise ValueError('immutable base differs from serialized base')
    branch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('base.')}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact), 'branch': branch,
            'normalization': copy.deepcopy(normalization), 'binding': {
                'mode': model.mode, 'schema': SCHEMA, 'bins': BINS, 'classes': list(CLASSES),
                'normalization_sha256': normalization_digest(normalization),
                'base_head_sha256': tensor_digest(base_artifact['head']),
                'branch_sha256': tensor_digest(branch)}}


def load_spectral(artifact, expected_base, mode, device):
    bound = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or bound.get('mode') != mode
            or bound.get('schema') != SCHEMA or bound.get('bins') != BINS
            or bound.get('classes') != list(CLASSES)
            or bound.get('normalization_sha256') != normalization_digest(artifact['normalization'])
            or bound.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or bound.get('branch_sha256') != tensor_digest(artifact['branch'])):
        raise ValueError('spectral artifact binding differs')
    base = load_probe(artifact['base'], expected_base, device)
    model = SpectralResidual(base, mode).to(device).eval()
    state = model.state_dict()
    expected = {k: v for k, v in state.items() if not k.startswith('base.')}
    if set(expected) != set(artifact['branch']) or any(
            artifact['branch'][k].shape != v.shape or artifact['branch'][k].dtype != v.dtype
            for k, v in expected.items()):
        raise ValueError('spectral branch keys/shapes/dtypes differ')
    state.update(artifact['branch'])
    model.load_state_dict(state, strict=True)
    return model

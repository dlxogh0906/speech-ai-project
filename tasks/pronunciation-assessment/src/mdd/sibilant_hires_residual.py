"""Typed raw acoustic temporal residual, with an immutable native HuBERT head."""
import copy
import torch
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hires_acoustics import SCHEMA, NAMES, normalization_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'hires_acoustic_residual_cv_v1'


class HiresResidual(torch.nn.Module):
    def __init__(self, base, mode):
        super().__init__()
        if mode not in ('fine', 'coarse20ms'):
            raise ValueError('fine/coarse20ms residual required')
        self.mode = mode
        self.base = base.requires_grad_(False).eval()
        channels = 2*len(NAMES)+1
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv1d(channels, 32, 5, padding=2),
            torch.nn.Conv1d(32, 32, 5, padding=8, dilation=4),
            torch.nn.Conv1d(32, 32, 5, padding=32, dilation=16)])
        self.output = torch.nn.Linear(64, len(CLASSES))
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, hidden, mask, relative, raw, raw_mask, raw_relative):
        if (raw.shape[:2] != raw_mask.shape or raw.shape[-1] != 2*len(NAMES)
                or raw_relative.shape != raw_mask.shape or not bool(raw_mask.any(1).all())):
            raise ValueError('native acoustic frame grid/mask mismatch')
        logits, phase = self.base(hidden, mask, relative)
        z = torch.cat([raw, raw_relative[..., None]], -1)
        z = z.masked_fill(~raw_mask[..., None], 0).transpose(1, 2)
        for layer in self.layers:
            z = torch.nn.functional.gelu(layer(z)).masked_fill(~raw_mask[:, None], 0)
        mean = z.sum(-1)/raw_mask.sum(-1, keepdim=True)
        maximum = z.masked_fill(~raw_mask[:, None], -torch.inf).amax(-1)
        return logits+self.output(torch.cat([mean, maximum], -1)), phase


def residual_artifact(model, base_artifact, normalization):
    branch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('base.')}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact), 'branch': branch,
        'normalization': copy.deepcopy(normalization), 'binding': {
            'mode': model.mode, 'acoustic_schema': SCHEMA, 'acoustic_names': list(NAMES),
            'normalization_sha256': normalization_digest(normalization), 'classes': list(CLASSES),
            'base_head_sha256': tensor_digest(base_artifact['head']),
            'branch_sha256': tensor_digest(branch)}}


def load_residual(artifact, expected_base, mode, device):
    bound = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or bound.get('mode') != mode
            or bound.get('acoustic_schema') != SCHEMA or bound.get('acoustic_names') != list(NAMES)
            or bound.get('classes') != list(CLASSES)
            or bound.get('normalization_sha256') != normalization_digest(artifact['normalization'])
            or bound.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or bound.get('branch_sha256') != tensor_digest(artifact['branch'])):
        raise ValueError('typed acoustic residual/base/schema/normalization binding differs')
    base = load_probe(artifact['base'], expected_base, device)
    if artifact['base']['binding']['classes'] != list(CLASSES):
        raise ValueError('base actual-class identity differs')
    model = HiresResidual(base, mode).to(device).eval()
    state = model.state_dict()
    state.update(artifact['branch'])
    expected_branch = {k for k in state if not k.startswith('base.')}
    if set(artifact['branch']) != expected_branch:
        raise ValueError('raw residual branch keys differ')
    model.load_state_dict(state, strict=True)
    return model

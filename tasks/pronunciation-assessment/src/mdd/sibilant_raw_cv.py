"""Automatic undivided C/V pooling of observed raw acoustic trajectories.

Role supervision is coarse TRAIN TAG support, not physical frication, vowel
resonance or glottal landmarks. No annotation/actual phone is a forward input.
"""
import copy
import torch
from mdd.sibilant_hires_residual import HiresResidual
from mdd.sibilant_hires_acoustics import SCHEMA, NAMES, normalization_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'raw_predicted_CV_residual_v1'
ROLE_SCHEMA = 'coarse_TAG_background_C_undividedV_raw2ms_guard25ms_v1'


def coarse_role_loss(logits, target):
    known = target >= 0
    if not bool(known.any()):
        return logits.sum()*0
    return torch.nn.functional.cross_entropy(logits[known], target[known])


def cv_terms(features, role, mask, relative):
    if (features.shape[:2] != mask.shape or role.shape != (*mask.shape, 3)
            or relative.shape != mask.shape or not bool(mask.any(1).all())):
        raise ValueError('raw feature/CV role time grid differs')
    z = features.masked_fill(~mask[..., None], 0)
    r = relative.masked_fill(~mask, 0)
    weights = role.softmax(-1)[..., 1:].transpose(1, 2)*mask[:, None]
    weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
    means = torch.bmm(weights, z)
    vw = weights[:, 1]
    centered_time = r-(vw*r).sum(-1, keepdim=True)
    std = (vw*centered_time.square()).sum(-1, keepdim=True).clamp(min=1e-8).sqrt()
    drift = (vw[..., None]*centered_time[..., None]*z).sum(1)/std
    return torch.cat([means[:, 0], means[:, 1]-means[:, 0], drift], -1), weights


class RawCVResidual(HiresResidual):
    def __init__(self, base, mode):
        if mode not in ('unguided', 'guided'):
            raise ValueError('matched unguided/guided raw CV model required')
        super().__init__(base, 'fine')
        self.mode = mode
        self.role = torch.nn.Linear(32, 3)
        self.output = torch.nn.Linear(96, len(CLASSES))
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def raw_terms(self, raw, mask, relative):
        if (raw.shape[:2] != mask.shape or raw.shape[-1] != 2*len(NAMES)
                or relative.shape != mask.shape or not bool(mask.any(1).all())):
            raise ValueError('raw acoustic observations/relative clock required')
        z = torch.cat([raw, relative[..., None]], -1).masked_fill(~mask[..., None], 0).transpose(1, 2)
        for layer in self.layers:
            z = torch.nn.functional.gelu(layer(z)).masked_fill(~mask[:, None], 0)
        z = z.transpose(1, 2)
        role = self.role(z)
        summary, weights = cv_terms(z, role, mask, relative)
        return summary, role, weights

    def forward(self, hidden, mask, relative, raw, raw_mask, raw_relative):
        logits, base_phase = self.base(hidden, mask, relative)
        summary, role, _ = self.raw_terms(raw, raw_mask, raw_relative)
        return logits+self.output(summary), base_phase, role


def cv_artifact(model, base_artifact, normalization):
    if tensor_digest(model.base.head.state_dict()) != tensor_digest(base_artifact['head']):
        raise ValueError('immutable base head differs from supplied artifact')
    branch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('base.')}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact), 'branch': branch,
        'normalization': copy.deepcopy(normalization), 'binding': {
            'mode': model.mode, 'role_schema': ROLE_SCHEMA, 'acoustic_schema': SCHEMA,
            'acoustic_names': list(NAMES), 'classes': list(CLASSES),
            'normalization_sha256': normalization_digest(normalization),
            'base_head_sha256': tensor_digest(base_artifact['head']),
            'branch_sha256': tensor_digest(branch)}}


def load_cv(artifact, expected_base, mode, device):
    bound = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or bound.get('mode') != mode
            or bound.get('role_schema') != ROLE_SCHEMA or bound.get('acoustic_schema') != SCHEMA
            or bound.get('acoustic_names') != list(NAMES) or bound.get('classes') != list(CLASSES)
            or bound.get('normalization_sha256') != normalization_digest(artifact['normalization'])
            or bound.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or bound.get('branch_sha256') != tensor_digest(artifact['branch'])):
        raise ValueError('typed raw CV support/observation/base binding differs')
    if artifact['base']['binding']['classes'] != list(CLASSES):
        raise ValueError('base actual-class order differs')
    model = RawCVResidual(load_probe(artifact['base'], expected_base, device), mode).to(device).eval()
    reference = model.state_dict()
    keys = {k for k in reference if not k.startswith('base.')}
    if keys != set(artifact['branch']) or any(
            artifact['branch'][k].shape != reference[k].shape
            or artifact['branch'][k].dtype != reference[k].dtype
            or not bool(torch.isfinite(artifact['branch'][k]).all()) for k in keys):
        raise ValueError('raw CV branch keys/shapes/dtypes differ')
    reference.update(artifact['branch'])
    model.load_state_dict(reference, strict=True)
    return model

"""Separate fixed automatic C/V localization from actual-phone feature learning."""
import copy
import torch
from mdd.sibilant_raw_cv import RawCVResidual, cv_terms, load_cv, ROLE_SCHEMA
from mdd.sibilant_temporal_residual import shuffle_observed
from mdd.sibilant_hires_acoustics import SCHEMA, NAMES, normalization_digest
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'frozen_raw_CV_teacher_residual_v1'


class FrozenRawRoles(RawCVResidual):
    def __init__(self, base, teacher, mode):
        if mode not in ('aligned', 'permuted'):
            raise ValueError('aligned/permuted frozen role control required')
        super().__init__(base, 'unguided')
        self.mode = mode
        self.teacher = teacher.requires_grad_(False).eval()
        # The student never learns localization. Its acoustic CNN and actual
        # readout have their own CE gradients; no teacher gradient reaches them.
        self.role.requires_grad_(False)

    def teacher_roles(self, raw, mask, relative, permutation=None):
        with torch.no_grad():
            _, role, _ = self.teacher.raw_terms(raw, mask, relative)
        if self.mode == 'permuted':
            if permutation is None:
                raise ValueError('fixed runtime-identity permutation required')
            role = shuffle_observed(role, mask, permutation)
        elif permutation is not None:
            raise ValueError('aligned prediction must not shuffle roles')
        return role

    def forward(self, hidden, mask, relative, raw, raw_mask, raw_relative, permutation=None):
        logits, base_phase = self.base(hidden, mask, relative)
        role = self.teacher_roles(raw, raw_mask, raw_relative, permutation)
        z = torch.cat([raw, raw_relative[..., None]], -1).masked_fill(~raw_mask[..., None], 0).transpose(1, 2)
        for layer in self.layers:
            z = torch.nn.functional.gelu(layer(z)).masked_fill(~raw_mask[:, None], 0)
        summary, _ = cv_terms(z.transpose(1, 2), role, raw_mask, raw_relative)
        return logits+self.output(summary), base_phase, role


def role_artifact(model, base_artifact, teacher_artifact, normalization):
    if tensor_digest(model.base.head.state_dict()) != tensor_digest(base_artifact['head']):
        raise ValueError('immutable native head changed')
    branch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if not k.startswith(('base.', 'teacher.'))}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact),
        'teacher': copy.deepcopy(teacher_artifact), 'normalization': copy.deepcopy(normalization),
        'branch': branch, 'binding': {'mode': model.mode, 'role_schema': ROLE_SCHEMA,
            'acoustic_schema': SCHEMA, 'acoustic_names': list(NAMES), 'classes': list(CLASSES),
            'normalization_sha256': normalization_digest(normalization),
            'base_head_sha256': tensor_digest(base_artifact['head']),
            'teacher_tensor_sha256': tensor_digest(model.teacher.state_dict()),
            'branch_sha256': tensor_digest(branch)}}


def load_roles(artifact, expected_base, mode, device):
    bound = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or bound.get('mode') != mode
            or bound.get('role_schema') != ROLE_SCHEMA or bound.get('acoustic_schema') != SCHEMA
            or bound.get('acoustic_names') != list(NAMES) or bound.get('classes') != list(CLASSES)
            or bound.get('normalization_sha256') != normalization_digest(artifact['normalization'])
            or bound.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or bound.get('branch_sha256') != tensor_digest(artifact['branch'])):
        raise ValueError('frozen role artifact identity/schema/base/branch differs')
    teacher = load_cv(artifact['teacher'], expected_base, 'guided', device)
    if (tensor_digest(teacher.state_dict()) != bound.get('teacher_tensor_sha256')
            or normalization_digest(artifact['teacher']['normalization']) != bound['normalization_sha256']):
        raise ValueError('frozen role teacher or its observation normalization differs')
    model = FrozenRawRoles(load_probe(artifact['base'], expected_base, device), teacher, mode).to(device).eval()
    state = model.state_dict()
    keys = {k for k in state if not k.startswith(('base.', 'teacher.'))}
    if keys != set(artifact['branch']) or any(
            artifact['branch'][k].shape != state[k].shape
            or artifact['branch'][k].dtype != state[k].dtype
            or not bool(torch.isfinite(artifact['branch'][k]).all()) for k in keys):
        raise ValueError('frozen role student keys/shapes/dtypes differ')
    state.update(artifact['branch'])
    model.load_state_dict(state, strict=True)
    return model

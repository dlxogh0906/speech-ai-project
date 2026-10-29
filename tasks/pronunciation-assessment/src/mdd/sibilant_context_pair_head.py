"""Single-clip actual-phone readout; pair supervision never enters inference."""
import copy
import torch
from mdd.sibilant_frozen_probe import load_probe, project_observed
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_vowel_presence import CLASSES
from mdd.sibilant_context_pairs import SCHEMA

READOUT = 'context_pair_actual_head_cv_v1'
MODES = ('ce_only', 'matched', 'shuffled')


class ContextPairHead(torch.nn.Module):
    def __init__(self, base, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('typed comparison mode required')
        self.mode = mode
        self.base = base.requires_grad_(False).eval()
        self.head = copy.deepcopy(base.head).requires_grad_(False)
        self.head.actual.requires_grad_(True)
        self.head.amount.requires_grad_(True)

    def forward(self, hidden, mask, relative):
        projection = {k: getattr(self.base, k) for k in ('mean', 'scale', 'components')}
        return self.head(project_observed(hidden, mask, projection), mask, relative)


def pair_artifact(model, base_artifact):
    reference = load_probe(base_artifact, base_artifact['binding'], 'cpu')
    if (tensor_digest(model.base.state_dict()) != tensor_digest(reference.state_dict())
            or tensor_digest(model.head.support.state_dict()) != tensor_digest(reference.head.support.state_dict())):
        raise ValueError('immutable base/projection/role weights changed')
    head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact), 'head': head,
            'binding': {'mode': model.mode, 'supervision_schema': SCHEMA, 'classes': list(CLASSES),
                'base_head_sha256': tensor_digest(base_artifact['head']), 'head_sha256': tensor_digest(head)}}


def load_pair_head(artifact, expected_base, mode, device):
    bound = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or bound.get('mode') != mode
            or bound.get('supervision_schema') != SCHEMA or bound.get('classes') != list(CLASSES)
            or bound.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or bound.get('head_sha256') != tensor_digest(artifact['head'])):
        raise ValueError('typed context-pair artifact binding differs')
    base = load_probe(artifact['base'], expected_base, device)
    model = ContextPairHead(base, mode).to(device).eval()
    expected = model.head.state_dict()
    if set(expected) != set(artifact['head']) or any(
            artifact['head'][k].shape != v.shape or artifact['head'][k].dtype != v.dtype
            for k, v in expected.items()):
        raise ValueError('context-pair head keys/shapes/dtypes differ')
    model.head.load_state_dict(artifact['head'])
    if tensor_digest(model.head.support.state_dict()) != tensor_digest(model.base.head.support.state_dict()):
        raise ValueError('immutable support head changed')
    return model

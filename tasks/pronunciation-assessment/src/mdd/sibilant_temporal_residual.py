"""Fixed-feature temporal interactions, with an equal-size permutation control."""
import hashlib
import numpy as np
import torch
from mdd.sibilant_frozen_probe import FrozenProbe, project_observed, projection_digest
from mdd.sibilant_role_time import temporal_moments
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'temporal_residual_cv_v1'
BRANCH = 'conv32_k1_k3d1_k3d2_meanmax_zero64_v1'


def make_permutations(features, lengths, width, seed):
    """Shuffle observed frames only, from identifiers without actual labels."""
    if len(features) != len(lengths) or any(not 0 < int(n) <= width for n in lengths):
        raise ValueError('observed lengths and feature identities required')
    result = np.tile(np.arange(width, dtype=np.int64), (len(features), 1))
    for i, (feature, length) in enumerate(zip(features, lengths)):
        prefix = str(seed) + '/' + feature['uid'] + '/' + str(feature['q_runtime']) + '/'
        result[i, :length] = sorted(range(int(length)),
                                    key=lambda j: hashlib.sha256((prefix + str(j)).encode()).digest())
    return result


def shuffle_observed(hidden, mask, permutation):
    if permutation.shape != mask.shape or permutation.dtype != torch.long:
        raise ValueError('one integer permutation per observed prefix required')
    expected = torch.arange(mask.shape[1], device=mask.device).expand_as(permutation)
    if not torch.equal(permutation.sort(1).values, expected):
        raise ValueError('duplicate or out-of-range frame permutation')
    if not torch.equal(mask.gather(1, permutation), mask):
        raise ValueError('padding cannot be shuffled into observed frames')
    return hidden.gather(1, permutation[..., None].expand_as(hidden))


class TemporalBranch(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv1d(dimensions, 32, 1),
            torch.nn.Conv1d(32, 32, 3, padding=1),
            torch.nn.Conv1d(32, 32, 3, padding=2, dilation=2)])
        self.output = torch.nn.Linear(64, 64, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    def forward(self, hidden, mask):
        value = hidden.masked_fill(~mask[..., None], 0).transpose(1, 2)
        for layer in self.layers:
            value = torch.nn.functional.gelu(layer(value))
            value = value.masked_fill(~mask[:, None, :], 0)
        mean = value.sum(-1) / mask.sum(1, keepdim=True)
        maximum = value.masked_fill(~mask[:, None, :], -torch.inf).max(-1).values
        return self.output(torch.cat([mean, maximum], -1))


class TemporalResidualProbe(FrozenProbe):
    def __init__(self, projection, mode='ordered'):
        if mode not in ('ordered', 'shuffled'):
            raise ValueError('ordered or shuffled branch mode required')
        super().__init__(projection)
        # Construct the branch after the base head to preserve its initialization.
        self.branch = TemporalBranch(self.components.shape[1])
        self.mode = mode
        self.pca_variance = projection['explained_variance_ratio']

    def forward(self, hidden, mask, relative, permutation=None):
        if relative.shape != mask.shape or not bool(mask.any(1).all()):
            raise ValueError('observed frames and relative coordinates required')
        projection = {key: getattr(self, key) for key in ('mean', 'scale', 'components')}
        observed = project_observed(hidden, mask, projection)
        phase = self.head.support(torch.cat([observed, relative[..., None]], -1))
        weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2) * mask[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, observed)
        summary = torch.cat([means[:, 0], means[:, 1] - means[:, 0],
                             means[:, 2] - means[:, 1]], -1)
        value = self.head.actual[0](summary) + self.head.amount(temporal_moments(phase, mask))
        if self.mode == 'shuffled':
            if permutation is None:
                raise ValueError('shuffled diagnostic requires its fixed permutation')
            branch_input = shuffle_observed(observed, mask, permutation)
        else:
            if permutation is not None:
                raise ValueError('ordered candidate must not receive a permutation')
            branch_input = observed
        value = value + self.branch(branch_input, mask)
        return self.head.actual[2](self.head.actual[1](value)), phase


def temporal_artifact(model, binding):
    projection = {key: getattr(model, key).detach().cpu().clone()
                  for key in ('mean', 'scale', 'components')}
    projection['explained_variance_ratio'] = model.pca_variance
    return {'readout': READOUT, 'branch_schema': BRANCH,
            'binding': dict(binding, classes=list(CLASSES), temporal_mode=model.mode,
                            projection_sha256=projection_digest(projection)),
            'projection': projection,
            'head': {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
            'branch': {k: v.detach().cpu().clone() for k, v in model.branch.state_dict().items()}}


def load_temporal_probe(artifact, expected, device):
    if artifact.get('readout') != READOUT or artifact.get('branch_schema') != BRANCH:
        raise ValueError('typed temporal residual artifact required')
    if artifact['binding'].get('classes') != list(CLASSES):
        raise ValueError('actual class order differs')
    for key, value in expected.items():
        if artifact['binding'].get(key) != value:
            raise ValueError('temporal artifact binding mismatch: ' + key)
    if artifact['binding']['projection_sha256'] != projection_digest(artifact['projection']):
        raise ValueError('projection fingerprint differs')
    model = TemporalResidualProbe(artifact['projection'], artifact['binding']['temporal_mode']).to(device).eval()
    model.head.load_state_dict(artifact['head'])
    model.branch.load_state_dict(artifact['branch'])
    return model

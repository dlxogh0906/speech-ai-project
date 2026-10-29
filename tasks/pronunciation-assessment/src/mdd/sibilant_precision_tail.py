"""Frozen observations and actual-five readout for TRAIN precision-tail risk.

Only the training loss sees real-error/normal sets. Runtime remains single-clip
actual-phone inference, with unchanged role availability. No pitch/TAG input.
"""
import copy
import math
import torch
from mdd.sibilant_frozen_probe import load_probe, project_observed
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_role_time import temporal_moments
from mdd.sibilant_vowel_presence import CLASSES

READOUT = 'precision_tail_actual_head_cv_v1'
SUMMARY_SCHEMA = 'frozen_native_PCA_rolemeans3_moments9_float32_v1'
MODES = ('ce_full', 'fullrank', 'tailrank')


class PrecisionTailHead(torch.nn.Module):
    def __init__(self, base, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('typed precision-tail mode required')
        self.mode = mode
        self.base = base.requires_grad_(False).eval()
        self.head = copy.deepcopy(base.head).requires_grad_(False)
        self.head.actual.requires_grad_(True)
        self.head.amount.requires_grad_(True)

    def summarize(self, hidden, mask, relative):
        """The same role means/moments as PresenceReadout, before trained layers."""
        if hidden.shape[:2] != mask.shape or relative.shape != mask.shape or not bool(mask.any(1).all()):
            raise ValueError('observed native frame grid required')
        projection = {k: getattr(self.base, k) for k in ('mean', 'scale', 'components')}
        h = project_observed(hidden, mask, projection)
        phase = self.head.support(torch.cat([h.float(), relative[..., None]], -1))
        weights = phase.softmax(-1)[:, :, 1:].transpose(1, 2)*mask[:, None, :]
        weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights, h.float())
        summary = torch.cat([means[:, 0], means[:, 1]-means[:, 0], means[:, 2]-means[:, 1]], -1)
        return summary, temporal_moments(phase, mask), phase

    def from_summary(self, summary, moments):
        if summary.ndim != 2 or moments.shape != (len(summary), 9):
            raise ValueError('native frozen summary/moments required')
        z = self.head.actual[0](summary)+self.head.amount(moments)
        return self.head.actual[2](self.head.actual[1](z))

    def forward(self, hidden, mask, relative):
        summary, moments, phase = self.summarize(hidden, mask, relative)
        return self.from_summary(summary, moments), phase


def opposite_score(logits, opposite):
    if logits.ndim != 2 or logits.shape[1] != len(CLASSES) or opposite not in (0, 1):
        raise ValueError('actual-five logits and S/SS opposite identity required')
    other = [k for k in range(len(CLASSES)) if k != opposite]
    return logits[:, opposite]-torch.logsumexp(logits[:, other], -1)


def rank_risk(logits, groups, mode, fraction=.01, margin=1.):
    """Logistic ranking surrogate; not an exact structural-SVM pAUC solver.

    Both conditions score the same full population and use identical positives.
    Tail mode only changes which confirmed-normal scores enter the rank mean.
    The top-k selection and its gradients use TRAIN predictions exclusively.
    """
    if mode not in ('fullrank', 'tailrank') or not 0 < fraction <= 1 or len(groups) != 2:
        raise ValueError('two directional TRAIN groups and a valid rank mode required')
    losses, selections = [], []
    for opposite, group in enumerate(groups):
        pos, neg = [torch.as_tensor(group[k], device=logits.device, dtype=torch.long)
                    for k in ('positive', 'negative')]
        if (pos.ndim != 1 or neg.ndim != 1 or not len(pos) or not len(neg)
                or bool((pos < 0).any()) or bool((neg < 0).any())
                or bool((pos >= len(logits)).any()) or bool((neg >= len(logits)).any())
                or bool(torch.isin(pos, neg).any())):
            raise ValueError('disjoint nonempty observed positive/normal sets required')
        score = opposite_score(logits, opposite)
        chosen = neg
        if mode == 'tailrank':
            count = max(1, int(math.ceil(fraction*len(neg))))
            chosen = neg[torch.topk(score[neg], count, sorted=False).indices]
        difference = score[pos, None]-score[chosen][None, :]
        losses.append(torch.nn.functional.softplus(float(margin)-difference).mean())
        selections.append(chosen.detach())
    return torch.stack(losses).mean(), selections


def tail_artifact(model, base_artifact):
    original = load_probe(base_artifact, base_artifact['binding'], 'cpu')
    if (tensor_digest(model.base.state_dict()) != tensor_digest(original.state_dict())
            or tensor_digest(model.head.support.state_dict()) != tensor_digest(original.head.support.state_dict())):
        raise ValueError('immutable native base/roles changed')
    head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
    return {'readout': READOUT, 'base': copy.deepcopy(base_artifact), 'head': head,
        'binding': {'mode': model.mode, 'summary_schema': SUMMARY_SCHEMA, 'classes': list(CLASSES),
            'base_head_sha256': tensor_digest(base_artifact['head']), 'head_sha256': tensor_digest(head)}}


def load_tail_head(artifact, expected_base, mode, device):
    bound = artifact.get('binding', {})
    if (artifact.get('readout') != READOUT or bound.get('mode') != mode
            or bound.get('summary_schema') != SUMMARY_SCHEMA or bound.get('classes') != list(CLASSES)
            or bound.get('base_head_sha256') != tensor_digest(artifact['base']['head'])
            or bound.get('head_sha256') != tensor_digest(artifact['head'])):
        raise ValueError('typed precision-tail artifact binding differs')
    model = PrecisionTailHead(load_probe(artifact['base'], expected_base, device), mode).to(device).eval()
    state = model.head.state_dict()
    if set(state) != set(artifact['head']) or any(v.shape != artifact['head'][k].shape
            or v.dtype != artifact['head'][k].dtype for k, v in state.items()):
        raise ValueError('precision-tail head keys/shapes/dtypes differ')
    model.head.load_state_dict(artifact['head'])
    if tensor_digest(model.head.support.state_dict()) != tensor_digest(model.base.head.support.state_dict()):
        raise ValueError('immutable role head differs')
    return model

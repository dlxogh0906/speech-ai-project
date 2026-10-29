"""Retained-error exposure controls and runtime-expected-C readout routing.

Source labels select TRAIN observations only. Runtime routing uses the canonical
C already computed by the service, never a TAG actual or annotation boundary.
"""
import collections
import hashlib
import numpy as np
import torch
from mdd.sibilant_precision_tail import (PrecisionTailHead, tail_artifact,
                                       load_tail_head)
from mdd.sibilant_frozen_probe import load_probe

TOKENS = ('ㅅ', 'ㅆ')
READOUT = 'runtime_expected_C_conditional_precision_head_v1'


def retained_plans(original, y, labels, features, eligibility, offset, seed,
                   steps=300, batch=16):
    """Keep the first original stream, augment only later repeated error slots."""
    result, reports = {}, {}
    if original.ndim != 2 or original.shape[1] != batch:
        raise ValueError('original replay batch width differs')
    for token in TOKENS:
        stream = []
        for value in original.ravel():
            i = int(value)
            if i < 0 or i >= offset:
                raise ValueError('original source index outside original population')
            if eligibility[i]['target'] is None:
                continue
            if features[i]['canonical'] != token or labels[i]['expected'] != token:
                continue
            if y[i] < 0:
                raise ValueError('unknown TRAIN target in original eligible stream')
            stream.append(i)
        if not stream or len(stream) >= steps*batch:
            raise ValueError('no room to retain original stream and augment errors')
        stream = np.asarray(stream, dtype=np.int64)
        original_errors = {int(i) for i in stream if labels[int(i)]['label_kind']=='pair_error'}
        pools = [[], []]
        for i, label in sorted(labels.items()):
            if (label['label_kind'] != 'pair_error' or label['expected'] != token
                    or features[i]['canonical'] != token or y[i] != 1-TOKENS.index(token)
                    or label['following'] != features[i]['expected_V']):
                continue
            if i < offset and (eligibility[i]['target'] is None or i in original_errors):
                continue
            pools[int(i >= offset)].append(i)
        def priority(i):
            key = 'conditional-retained-v1:%s:%s:%s' % (seed, labels[i]['uid'], features[i]['q_runtime'])
            return hashlib.sha256(key.encode()).hexdigest(), i
        pools = [sorted(pool, key=priority) for pool in pools]
        count = len(pools[1])
        if not count or len(pools[0]) < count:
            raise ValueError('matched unused-old control pool insufficient')
        old_extra, new_extra = pools[0][:count], pools[1]
        template = np.resize(stream, steps*batch)
        positions = [j for j in range(len(stream), len(template))
                     if labels[int(template[j])]['label_kind']=='pair_error']
        if len(positions) < count:
            raise ValueError('not all new error features fit after original first cycle')
        old, new = template.copy(), template.copy()
        assignment = np.resize(np.random.RandomState(seed+TOKENS.index(token)).permutation(count), len(positions))
        for pos, at in zip(positions, assignment):
            old[pos], new[pos] = old_extra[int(at)], new_extra[int(at)]
        np.testing.assert_array_equal(old[:len(stream)], stream)
        np.testing.assert_array_equal(new[:len(stream)], stream)
        mapping = dict(zip(old_extra, new_extra))
        for a, b, source in zip(old, new, template):
            a, b, source = int(a), int(b), int(source)
            if a != b and mapping.get(a) != b:
                raise ValueError('extra-error reuse differs between data conditions')
            if labels[source]['label_kind'] != 'pair_error' and (a != source or b != source):
                raise ValueError('normal/third/deletion draw changed')
            if y[a] != y[b]:
                raise ValueError('actual-class draw balance differs')
        if not set(new_extra).issubset(set(new.tolist())):
            raise ValueError('new error coverage incomplete')
        if not original_errors.issubset(set(old.tolist()) & set(new.tolist())):
            raise ValueError('original errors were replaced instead of retained')
        result[token] = {'old':old.reshape(steps,batch), 'new':new.reshape(steps,batch)}
        reports[token] = {'original_stream_draws':len(stream),
            'original_unique_errors_retained':len(original_errors),
            'unused_old_control_pool':len(pools[0]), 'new_error_features':count,
            'post_original_pair_slots':len(positions), 'draws_each':len(template),
            'unique_inputs_each':len(set(old.tolist())),
            'first_original_cycle_unchanged':True, 'extra_pair_reuse_bijective':True,
            'nonpair_draw_identity_unchanged':True}
        if len(set(old.tolist())) != len(set(new.tolist())):
            raise ValueError('unique observation counts differ')
    return result, reports


class ConditionalPrecisionHead(torch.nn.Module):
    """Two actual-five heads; expected_index is runtime S=0 / SS=1."""
    def __init__(self, heads):
        super().__init__()
        if len(heads) != 2:
            raise ValueError('both expected-C heads required')
        self.heads = torch.nn.ModuleList(heads)

    def from_summary(self, summary, moments, expected_index):
        if (expected_index.ndim != 1 or len(expected_index) != len(summary)
                or expected_index.dtype != torch.long
                or bool(((expected_index<0) | (expected_index>1)).any())):
            raise ValueError('runtime canonical S/SS indices required')
        # Score whole batches before routing, keeping original GEMM boundaries.
        values = [head.from_summary(summary,moments) for head in self.heads]
        return torch.where(expected_index[:,None]==0, values[0], values[1])

    def forward(self, hidden, mask, relative, expected_index):
        summary, moments, phase = self.heads[0].summarize(hidden,mask,relative)
        return self.from_summary(summary,moments,expected_index), phase


def new_conditional(base, binding, device):
    return ConditionalPrecisionHead([PrecisionTailHead(load_probe(base,binding,device),'ce_full')
                                     for _ in TOKENS]).to(device).eval()


def conditional_artifact(model, base):
    heads = {}
    for token, head in zip(TOKENS,model.heads):
        heads[token] = dict(tail_artifact(head,base),runtime_expected_C=token)
    return {'readout':READOUT, 'runtime_expected_order':list(TOKENS), 'heads':heads}


def load_conditional(artifact, binding, device):
    if (artifact.get('readout') != READOUT
            or artifact.get('runtime_expected_order') != list(TOKENS)
            or set(artifact.get('heads',{})) != set(TOKENS)
            or any(artifact['heads'][token].get('runtime_expected_C')!=token for token in TOKENS)):
        raise ValueError('typed conditional artifact scope differs')
    models = [load_tail_head(artifact['heads'][token],binding,'ce_full',device) for token in TOKENS]
    # The two branches must share exactly the same immutable observation basis.
    from mdd.sibilant_hubert_adaptation import tensor_digest
    if tensor_digest(models[0].base.state_dict()) != tensor_digest(models[1].base.state_dict()):
        raise ValueError('conditional branches have different frozen bases')
    return ConditionalPrecisionHead(models).to(device).eval()


def fixed_direction_proxy(ps, available, y, features, labels, budgets=(33,54)):
    """TRAIN diagnostic only: fixed independent FP budgets, preserving score ties."""
    keys = ('normal_FP','novel_TP','duplicate_TP','wrong_actual')
    branches = []
    for token, budget in zip(TOKENS,budgets):
        ci = 1-TOKENS.index(token)
        groups = collections.defaultdict(collections.Counter)
        for i, cls in enumerate(y):
            if (cls<0 or not available[i] or features[i]['canonical']!=token
                    or ps[i].argmax()!=ci or ps[i,ci]<.5):
                continue
            lab = labels[i]
            if lab['label_kind']=='correct':
                key = 'normal_FP'
            elif lab['label_kind']=='pair_error' and lab['expected']==token:
                phone = lab['existing']['phone'] or {}
                exact = (bool(phone.get('errorType')) and phone.get('alpha_jamo')==token
                         and phone.get('actual_jamo')==lab['actual'])
                key = 'duplicate_TP' if exact else 'novel_TP'
            else:
                key = 'wrong_actual'
            groups[float(ps[i,ci])][key] += 1
        curve = [dict(threshold=None,**{k:0 for k in keys})]
        counts = collections.Counter()
        for threshold in sorted(groups,reverse=True):
            counts.update(groups[threshold])
            if counts['normal_FP']<=budget:
                curve.append(dict(threshold=threshold,**{k:counts[k] for k in keys}))
        best = max(curve,key=lambda z:(z['novel_TP'],-z['normal_FP'],-z['wrong_actual'],z['duplicate_TP']))
        branches.append(dict(best,fixed_parent_normal_budget=budget))
    return dict({k:sum(b[k] for b in branches) for k in keys},
                S_to_SS=branches[0],SS_to_S=branches[1],
                scope='Original TRAIN diagnosis only; fixed33/54 budgets, not service 1percent')

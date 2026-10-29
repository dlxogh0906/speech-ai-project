"""Actual following-vowel supervision, separate from expected-vowel inputs.

Only TRAIN targets use TAG identity/support. Runtime receives acoustic frames
and automatic relative positions, never expected or transcribed vowel labels.
"""
import numpy as np
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_transition import VOWELS

TOKENS = tuple(VOWELS)
IDS = {token:i for i,token in enumerate(TOKENS)}


def intended_vowel_target(label, row, mapping, phase=None):
    q, a = label['q'], label.get('a')
    if label.get('alignment_ambiguous') or a is None:
        return -100, 'C_alignment_unknown'
    if q+1 >= len(mapping) or mapping[q+1] != {a+1}:
        return -100, 'intended_V_slot_not_unique_adjacent'
    if a+1 >= len(row['actual']) or row['actual'][a+1] not in IDS:
        return -100, 'intended_V_identity_unavailable'
    if phase is not None and not bool(np.isin(phase, [2,3]).any()):
        return -100, 'TRAIN_no_known_observed_V_frame'
    return IDS[row['actual'][a+1]], 'known_actual_intended_V'


def vowel_loss(logits, targets):
    known = targets >= 0
    if not bool(known.any()):
        return logits.sum()*0
    return torch.nn.functional.cross_entropy(logits[known], targets[known])


class VowelAuxReadout(SupportReadout):
    def __init__(self, hidden_size):
        super().__init__(hidden_size, True)
        self.vowel = torch.nn.Sequential(torch.nn.Linear(2*hidden_size,64),torch.nn.Tanh(),torch.nn.Linear(64,len(TOKENS)))

    def load_base(self,state):
        missing,unexpected = self.load_state_dict(state,strict=False)
        if set(missing) != {'vowel.0.weight','vowel.0.bias','vowel.2.weight','vowel.2.bias'} or unexpected:
            raise ValueError('ordinary learned-support base required')

    def all_outputs(self, hidden, mask, relative):
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not bool(mask.any(1).all()):
            raise ValueError('observed local frame grid required')
        phase = self.support(torch.cat([hidden.float(),relative[...,None]],-1))
        weights = phase.softmax(-1)[:,:,1:].transpose(1,2)*mask[:,None,:]
        weights = weights/weights.sum(-1,keepdim=True).clamp(min=1e-8)
        means = torch.bmm(weights,hidden.float())
        summary = torch.cat([means[:,0],means[:,1]-means[:,0],means[:,2]-means[:,1]],-1)
        return self.actual(summary),phase,self.vowel(torch.cat([means[:,1],means[:,2]],-1))

    def forward(self, hidden, mask, relative):
        actual,phase,_ = self.all_outputs(hidden,mask,relative)
        return actual,phase

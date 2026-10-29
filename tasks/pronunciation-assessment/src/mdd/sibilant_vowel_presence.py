"""V-referenced actual-phone classification with an explicit TAG-deletion class.

Neither a CTC token gap nor absent periodicity supplies the deletion label.
TRAIN label reliability uses existing TAG support; runtime inputs never do.
"""
import math
import numpy as np
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_observed_anchor import linked_anchor
from mdd.sibilant_transition import PAIR,VOWELS
from mdd.sibilant_role_time import temporal_moments

SCHEMA='runtime_nextV_original20left11right_presence_v1'
CLASSES=('s','ss','affricate','other_sound','TAG_deleted')


def next_reference(tokens,spans,canon,sites,q,optimal):
    if not 0<=q<len(canon)-1 or canon[q] not in PAIR or canon[q+1] not in VOWELS:
        raise ValueError('runtime s/ss before V required')
    site=sites[q+1]
    if site.get('j') is None or site.get('pred') not in VOWELS:
        return None,'next_V_unlinked'
    anchor=linked_anchor(tokens,spans,site)
    if optimal[q+1]!={site['j']}:
        return None,'next_V_ambiguous'
    return anchor,'observed_next_V'


def actual_class(label):
    if label.get('alignment_ambiguous') or label['label_kind']=='ambiguous':
        return -100
    token=label.get('actual')
    if token is None:
        return 4 if label['label_kind']=='deletion' else -100
    if token=='ㅅ':return 0
    if token=='ㅆ':return 1
    if token in ('ㅈ','ㅉ','ㅊ'):return 2
    return 3


def overlap_fraction(left,right,interval,prefix=None):
    if interval is None or not interval[0]<interval[1]:return 0.
    lo=int(np.ceil(interval[0]*16000));end=min(interval[1],interval[0]+prefix) if prefix else interval[1]
    hi=int(np.ceil(end*16000))
    return max(0,min(right,hi)-max(left,lo))/max(1,hi-lo)


def training_target(label,c,v,left,right,centers,adjacent,edge=.01):
    y=actual_class(label);phase=np.full(len(centers),-100,np.int64)
    if y<0:return -100,phase,'label_unknown'
    if v is None or not v[0]<v[1] or v[2] not in VOWELS:
        return -100,phase,'intended_V_bounds_unknown'
    if overlap_fraction(left,right,v,.12)<.5:
        return -100,phase,'intended_V_not_supported'
    if y!=4:
        if c is None or not adjacent or not c[0]<c[1]<=v[0]:
            return -100,phase,'C_V_relation_unknown'
        if overlap_fraction(left,right,c)<.5:
            return -100,phase,'C_not_supported'
    phase[:]=0
    boundaries=[v[0],min(v[1],v[0]+.12)]
    if y!=4:
        phase[(centers>=c[0])&(centers<c[1])]=1
        boundaries.extend(c[:2])
    phase[(centers>=v[0])&(centers<boundaries[1])]=2
    for boundary in boundaries:phase[abs(centers-boundary)<=edge]=-100
    return y,phase,'known_supported'


class PresenceReadout(SupportReadout):
    def __init__(self,hidden_size,split_absence):
        super().__init__(hidden_size,True)
        self.split_absence=bool(split_absence)
        if split_absence:self.actual[2]=torch.nn.Linear(64,5)
        self.amount=torch.nn.Linear(9,64,bias=False)
        torch.nn.init.zeros_(self.amount.weight)

    def load_base(self,state):
        state=dict(state)
        if self.split_absence:
            w,b=state['actual.2.weight'],state['actual.2.bias']
            state['actual.2.weight']=torch.cat([w,w[3:4]],0)
            state['actual.2.bias']=torch.cat([b[:3],b[3:4]-math.log(2),b[3:4]-math.log(2)])
        missing,unexpected=self.load_state_dict(state,strict=False)
        if missing!=['amount.weight'] or unexpected:raise ValueError('ordinary learned base required')

    def forward(self,hidden,mask,relative_vowel):
        if hidden.shape[:2]!=mask.shape or relative_vowel.shape!=mask.shape or not bool(mask.any(1).all()):
            raise ValueError('V-referenced observed prefix required')
        phase=self.support(torch.cat([hidden.float(),relative_vowel[...,None]],-1))
        weights=phase.softmax(-1)[:,:,1:].transpose(1,2)*mask[:,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp(min=1e-8)
        means=torch.bmm(weights,hidden.float())
        summary=torch.cat([means[:,0],means[:,1]-means[:,0],means[:,2]-means[:,1]],-1)
        z=self.actual[0](summary)+self.amount(temporal_moments(phase,mask))
        return self.actual[2](self.actual[1](z)),phase


def observed_vowel_support(phase,mask):
    """No C peak required for a deletion hypothesis; not physical proof of V."""
    p=phase.softmax(-1)[:,:,2:].sum(-1).masked_fill(~mask,0)
    return p.max(-1).values>=.5

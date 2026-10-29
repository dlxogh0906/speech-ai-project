"""Observe a missing C slot from a linked following-V reference, not a boundary."""
import numpy as np
from mdd.sibilant_transition import PAIR,VOWELS
from mdd.sibilant_observed_anchor import linked_anchor

SCHEMA='unlinked_C_observed_nextV_source20left11right_v1'


def annotated_intended_vowel_interval(intervals,options):
    """Diagnostic TAG only; reject ambiguous, nonvowel or invalid intervals."""
    if intervals is None or len(options)!=1 or None in options:
        return None
    i=next(iter(options))
    if not 0<=i<len(intervals) or intervals[i] is None:
        return None
    value=intervals[i]
    return value if value[2] in VOWELS and value[0]<value[1] else None


def reference_for_site(tokens,spans,canon,sites,q,optimal_mapping):
    if not 0<=q<len(canon)-1 or canon[q] not in PAIR or canon[q+1] not in VOWELS:
        raise ValueError('runtime s/ss before a vowel required')
    if sites[q].get('j') is not None or sites[q].get('pred') is not None:
        return None,'C_has_decoder_link'
    nxt=sites[q+1]
    if nxt.get('j') is None or nxt.get('pred') not in VOWELS:
        return None,'next_V_not_linked'
    anchor=linked_anchor(tokens,spans,nxt)
    if optimal_mapping[q+1]!={nxt['j']}:
        return None,'next_V_optimal_mapping_ambiguous'
    return anchor,'observed_next_V'


def original_window(sample_map,source_length,frame,dt):
    indices=np.asarray(sample_map)
    if indices.ndim!=1 or not len(indices) or np.any(np.diff(indices)<=0):
        raise ValueError('monotonic retained source sample map required')
    center=int((frame+.5)*dt*16000)
    if not 0<=center<len(indices):raise ValueError('vowel reference outside observed audio')
    shift=int(indices[center])-center
    left=max(0,int((frame-20)*dt*16000)+shift)
    right=min(source_length,int((frame+11)*dt*16000)+shift)
    if not left<right:raise ValueError('empty original observation')
    return left,right,shift

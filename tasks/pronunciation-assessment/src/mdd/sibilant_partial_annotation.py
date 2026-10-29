"""Conservative local TAG labels with unresolved spans preserved as ANY*.

None in an input sequence denotes unknown annotation of zero or more phones,
not silence, a CTC blank, a deletion, or permission to infer an actual phone.
All minimum-edit correspondences are retained. These are TRAIN annotation
operations; no annotation identity or boundary is a runtime acoustic input.
"""
import math
from mdd.sibilant_labels import PAIR,VOWELS

UNKNOWN = 'unknown_annotation_span'
SCHEMA = 'TAG_LAB_partial_ANYstar_alloptimal_literal_singleton_CV_v1'


def _edges(reference,target,i,j):
    n,m = len(reference),len(target)
    if i < n:
        yield i+1,j,0 if reference[i] is None else 1,UNKNOWN if reference[i] is None else None
    if j < m:
        yield i,j+1,0 if target[j] is None else 1,None
    if i < n and j < m:
        if reference[i] is None:
            yield i,j+1,0,None
        elif target[j] is None:
            yield i+1,j,0,UNKNOWN
        else:
            yield i+1,j+1,int(reference[i] != target[j]),j


def partial_mapping(reference,target):
    """Map known reference positions to known indices, UNKNOWN, or deletion.

Each unknown span may absorb any number of opposite-side tokens at zero cost.
Ordinary known-token substitutions/insertions/deletions cost one. Even equal
partial sequences must traverse the graph: unknown spans can hide ambiguity.
"""
    if any(z is not None and not isinstance(z,str) for z in list(reference)+list(target)):
        raise ValueError('literal string phones or unresolved None spans required')
    n,m = len(reference),len(target)
    f = [[10**9]*(m+1) for _ in range(n+1)]
    b = [[10**9]*(m+1) for _ in range(n+1)]
    f[0][0],b[n][m] = 0,0
    for i in range(n+1):
        for j in range(m+1):
            for a,c,cost,_ in _edges(reference,target,i,j):
                f[a][c] = min(f[a][c],f[i][j]+cost)
    for i in range(n,-1,-1):
        for j in range(m,-1,-1):
            for a,c,cost,_ in _edges(reference,target,i,j):
                b[i][j] = min(b[i][j],cost+b[a][c])
    score,options = f[n][m],[set() for _ in reference]
    for i in range(n):
        if reference[i] is None:
            options[i].add(UNKNOWN)
            continue
        for j in range(m+1):
            for a,c,cost,mapping in _edges(reference,target,i,j):
                if a == i+1 and f[i][j]+cost+b[a][c] == score:
                    options[i].add(mapping)
        if not options[i]:
            raise ValueError('known source token has no optimal correspondence')
    return options,score


def is_jamo(value):
    return len(value) == 1 and 0x3131 <= ord(value) <= 0x3163


def annotation_sequence(phones,allowed):
    """Keep raw index/legacy index; grouped phones have no invented inner time."""
    sequence,geometry,raw_indices,legacy_indices,legacy = [],[],[],[],[]
    for raw_index,(start,end,text) in enumerate(phones):
        text = ''.join(c for c in str(text).strip() if not c.isspace())
        if not text:
            continue
        first_legacy = len(legacy)
        legacy.extend(c for c in text if is_jamo(c))
        if all(is_jamo(c) and c in allowed for c in text):
            for k,token in enumerate(text):
                sequence.append(token)
                raw_indices.append(raw_index);legacy_indices.append(first_legacy+k)
                try:
                    bounds = (float(start),float(end),token)
                except (TypeError,ValueError):
                    bounds = None
                geometry.append(bounds if len(text) == 1 else None)
        else:
            sequence.append(None);geometry.append(None)
            raw_indices.append(raw_index);legacy_indices.append(None)
    return {'tokens':sequence,'geometry':geometry,'raw_indices':raw_indices,
            'legacy_indices':legacy_indices,'legacy_tokens':legacy}


def recover_sites(meta,allowed,frames):
    expected = annotation_sequence(meta.get('lab_phones',[]),allowed)
    actual = annotation_sequence(meta.get('tag_phones',[]),allowed)
    if expected['legacy_tokens'] != meta['canon'] or actual['legacy_tokens'] != meta['ltr']:
        raise ValueError('raw TAG/LAB differs from original cached parser tokens')
    reference,target = expected['tokens'],actual['tokens']
    options,score = partial_mapping(reference,target)
    sites = []
    for q,token in enumerate(reference[:-1]):
        if token not in PAIR or reference[q+1] not in VOWELS:
            continue
        row = {'partial_q':q,'legacy_q':expected['legacy_indices'][q],
               'expected':token,'following':reference[q+1],
               'position':'initial' if q == 0 else 'post_vowel' if reference[q-1] in VOWELS
                   else 'unknown_context' if reference[q-1] is None else 'post_consonant',
               'state':'unknown_correspondence','label_kind':None}
        if len(options[q]) != 1 or len(options[q+1]) != 1:
            sites.append(row);continue
        a,v = next(iter(options[q])),next(iter(options[q+1]))
        if not isinstance(a,int) or not isinstance(v,int):
            sites.append(row);continue
        if v != a+1 or target[a] not in PAIR or target[v] != reference[q+1]:
            row['state'] = 'not_literal_pair_C_and_same_V'
            sites.append(row);continue
        c,vowel = actual['geometry'][a],actual['geometry'][v]
        if c is None or vowel is None:
            row['state'] = 'grouped_or_missing_actual_geometry'
            sites.append(row);continue
        if (not all(math.isfinite(z) for z in c[:2]+vowel[:2])
                or not 0 <= c[0] < c[1] <= vowel[0] < vowel[1] <= frames/16000.):
            row['state'] = 'invalid_or_outside_original_CV_geometry'
            sites.append(row);continue
        row.update(state='literal_CV_same_all_optimal_correspondences',actual=target[a],
            label_kind='correct' if target[a] == token else 'pair_error',
            partial_actual_index=a,actual_legacy_index=actual['legacy_indices'][a],
            actual_raw_TAG_index=actual['raw_indices'][a],vowel_raw_TAG_index=actual['raw_indices'][v],
            C_TAG_interval=list(c),V_TAG_interval=list(vowel))
        sites.append(row)
    return {'schema':SCHEMA,'expected_partial':reference,'actual_partial':target,
            'minimum_known_edit_cost':score,'sites':sites,
            'expected_unknown_spans':reference.count(None),'actual_unknown_spans':target.count(None)}

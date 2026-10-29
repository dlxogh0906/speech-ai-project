"""Optional warning deferral, preserving heard tokens and original phone scores.

This is a separate, inactive response adapter. A deferral is uncertainty, not a
claim of correct pronunciation. Reproduce baseline aggregates before modifying
them; reject unsupported formatter behavior rather than silently corrupt it.
"""
import copy
import numpy as np
from mdd import speechsuper as ss


def review_queries(canon, phones, features, probabilities, available, thresholds):
    selected = {}
    for index, feature in enumerate(features):
        q = feature['q_runtime']
        token = canon[q]
        if token not in ('ㅅ', 'ㅆ') or not phones[q].get('errorType'):
            continue
        threshold = thresholds.get(token)
        ci = 0 if token == 'ㅅ' else 1
        ps = np.asarray(probabilities[index])
        if ps.shape != (4,) or not np.isfinite(ps).all():
            raise ValueError('four finite actual-class probabilities required')
        if threshold is not None and available[index] and ps.argmax() == ci and float(ps[ci]) >= threshold:
            selected[q] = {'probability': float(ps[ci]), 'threshold': threshold}
    return selected


def _word_type(word):
    ph = word['phonemes']
    n_sub = sum(bool(p['actual_jamo']) and p['actual_jamo'] != p['alpha_jamo']
                and not p.get('uncertain') for p in ph)
    n_del = sum(p['actual_jamo'] is None and not p.get('uncertain') for p in ph)
    if ph and n_del == len(ph):
        return ss.READ_MISS
    if word['_raw_mean'] is not None and word['_raw_mean'] < ss.BAD_SCORE and (n_sub or n_del):
        return ss.READ_BAD
    if n_sub or n_del:
        return ss.READ_SUB
    return ss.READ_INS if word.get('inserted') else ss.READ_NORMAL


def _overall(formatter, raw, words):
    got = [0 if w['readType'] == ss.READ_BAD else w['_raw_mean'] for w in words
           if w['_raw_mean'] is not None]
    pron = int(round(sum(got) / len(got))) if got else None
    us = (raw.get('scores') or {}) if any(s.get('pred') is not None for s in raw['sites']) else {}
    flu = ss.human_to_100(us.get('flu'), 'flu', formatter.pct)
    if formatter.flu and raw.get('feats'):
        from mdd.fluency import fluency_rule
        flu = int(round(fluency_rule(raw['feats'], formatter.flu['band'])))
    comp = ss.human_to_100(us.get('comp'), 'comp', formatter.pct) if formatter.utt_comp else None
    parts = []
    for key, value in [('pron', pron), ('flu', flu), ('comp', comp)]:
        if value is not None:
            parts.append((ss.W_OVERALL[key], value))
    overall = int(round(sum(w * v for w, v in parts) / sum(w for w, v in parts))) if parts else None
    return {'overall': ss.nz(overall), 'pronunciation': ss.nz(pron), 'fluency': ss.nz(flu),
            'rhythm': ss.nz(comp), 'comprehensibility': ss.nz(comp)}


def defer_warnings(formatter, body, raw, reviews, vowels):
    if not reviews:
        return body
    candidate = copy.deepcopy(body)
    words = candidate['result']['words']
    phones = [p for w in words for p in w['phonemes']]
    canon = raw['canon']
    if [p['alpha_jamo'] for p in phones] != canon:
        raise ValueError('formatter canonical phone order changed')
    offset = 0
    for word in words:
        scores = []
        for phone in word['phonemes']:
            i = offset
            key = formatter.ctx(canon, i, vowels) if formatter.ctx and vowels else None
            score = (formatter._score(key, raw['margins'][i] if i < len(raw['margins']) else None)
                     if raw['sites'][i].get('pred') is not None else None)
            if ss.nz(score) != phone['pronunciation']:
                raise ValueError('original phone score did not reproduce')
            if score is not None:
                scores.append(score)
            offset += 1
        word['_raw_mean'] = int(round(sum(scores) / len(scores))) if scores else None
        if _word_type(word) != word['readType']:
            raise ValueError('original word type did not reproduce')
        if ss.nz(0 if word['readType'] == ss.READ_BAD else word['_raw_mean']) != word['score']:
            raise ValueError('original word score did not reproduce')
    old = _overall(formatter, raw, words)
    if any(candidate['result'].get(k) != v for k, v in old.items()):
        raise ValueError('original utterance aggregates did not reproduce')
    for q in reviews:
        phone = phones[q]
        if canon[q] not in ('ㅅ', 'ㅆ') or phone.get('position') != '초성' or not phone.get('errorType') or raw['sites'][q].get('j') is None:
            raise ValueError('only existing linked onset s/ss warnings may be deferred')
        phone.pop('errorType')
        phone['uncertain'] = True
    for word in words:
        word['readType'] = _word_type(word)
        word['score'] = ss.nz(0 if word['readType'] == ss.READ_BAD else word['_raw_mean'])
    candidate['result'].update(_overall(formatter, raw, words))
    for word in words:
        word.pop('_raw_mean')
    return candidate

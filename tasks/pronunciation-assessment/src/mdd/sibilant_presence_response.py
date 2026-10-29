"""Inactive API adapter for V-reference decisions, including unlinked C.

Reproduce baseline aggregates before changes. Preserve every original phone
score, including None for an unrecognized C; never fabricate a GOP score.
"""
import copy
from mdd import speechsuper as ss
from mdd.sibilant_presence_policy import PAIR
from mdd.sibilant_warning_review import _word_type, _overall


def apply_decisions(formatter, body, raw, actions, vowels):
    if not actions:
        return body
    candidate = copy.deepcopy(body)
    words = candidate['result']['words']
    phones = [p for w in words for p in w['phonemes']]
    canon = raw['canon']
    if [p['alpha_jamo'] for p in phones] != canon:
        raise ValueError('canonical phone order changed')
    offset = 0
    for word in words:
        values = []
        for phone in word['phonemes']:
            key = formatter.ctx(canon, offset, vowels) if formatter.ctx and vowels else None
            value = (formatter._score(key, raw['margins'][offset] if offset < len(raw['margins']) else None)
                     if raw['sites'][offset].get('pred') is not None else None)
            if ss.nz(value) != phone['pronunciation']:
                raise ValueError('baseline phone score did not reproduce')
            if value is not None:
                values.append(value)
            offset += 1
        word['_raw_mean'] = int(round(sum(values) / len(values))) if values else None
        if (_word_type(word) != word['readType']
                or ss.nz(0 if word['readType'] == ss.READ_BAD else word['_raw_mean']) != word['score']):
            raise ValueError('baseline word aggregate did not reproduce')
    if any(candidate['result'].get(k) != v for k, v in _overall(formatter, raw, words).items()):
        raise ValueError('baseline utterance aggregate did not reproduce')
    for q, action in actions.items():
        phone = phones[q]
        if (canon[q] not in PAIR or phone.get('position') != '초성' or q + 1 >= len(canon)
                or canon[q + 1] not in vowels or raw['sites'][q + 1].get('j') is None
                or raw['sites'][q + 1].get('pred') not in vowels):
            raise ValueError('observed next-vowel reference required')
        if action['kind'] == 'defer':
            if not phone.get('errorType'):
                raise ValueError('only an existing warning can be deferred')
            phone.pop('errorType')
            phone['uncertain'] = True
        elif action['kind'] == 'pair':
            actual = action['actual']
            if actual != PAIR[canon[q]] or formatter.ctype is None:
                raise ValueError('opposite S/SS call required')
            sequence = list(canon)
            sequence[q] = actual
            phone['actual_jamo'] = actual
            phone['actual'] = formatter.j2p(sequence)[q] if formatter.j2p else None
            phone['errorType'] = formatter.ctype(canon[q], actual, 'sub', False)
            phone.pop('uncertain', None)
            phone['detector'] = action['probability']
        else:
            raise ValueError('unsupported presence decision')
    for word in words:
        word['readType'] = _word_type(word)
        word['score'] = ss.nz(0 if word['readType'] == ss.READ_BAD else word['_raw_mean'])
    candidate['result'].update(_overall(formatter, raw, words))
    for word in words:
        word.pop('_raw_mean')
    old_phones = [p for w in body['result']['words'] for p in w['phonemes']]
    for q, (a, b) in enumerate(zip(old_phones, phones)):
        if a['pronunciation'] != b['pronunciation'] or q not in actions and a != b:
            raise AssertionError('phone score or unrelated phone changed')
    return candidate

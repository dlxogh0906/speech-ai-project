"""Inactive policy retaining a local decoder's observed consonant identity."""
from mdd.sibilant_transition import PAIR, VOWELS


def local_phone_decision(feature, query, policy):
    if query['state'] != 'local_C_V':
        return None
    actual, confidence = query['actual'], query['confidence']
    expected = feature['canonical']
    if (expected not in PAIR or actual is None or len(actual) != 1
            or actual in VOWELS or not 0x3131 <= ord(actual) <= 0x3163
            or confidence is None or not 0 <= confidence <= 1):
        raise ValueError('observed local consonant and finite token confidence required')
    if actual == expected:
        threshold = policy['defer'].get(expected)
        if feature['baseline_warning'] and threshold is not None and confidence >= threshold:
            return {'kind': 'defer', 'probability': confidence}
        return None
    if policy['mode'] == 'pair' and actual != PAIR[expected]:
        return None
    threshold = policy['actual'].get(expected)
    already = feature['baseline_warning'] and feature['baseline_actual'] == actual
    if not already and threshold is not None and confidence >= threshold:
        return {'kind': 'actual', 'actual': actual, 'probability': confidence}
    return None

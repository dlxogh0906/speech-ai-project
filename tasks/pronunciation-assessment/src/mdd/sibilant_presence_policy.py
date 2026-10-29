"""Inactive service policy for V-referenced four/five-class observations.

Only runtime canonical tokens, existing phones, model scores and availability
enter decisions. TAG identities are exclusively calibration/evaluation inputs.
"""
import numpy as np

PAIR = {'ㅅ': 'ㅆ', 'ㅆ': 'ㅅ'}


def decision(feature, probabilities, available, policy):
    p = np.asarray(probabilities)
    if p.shape not in ((4,), (5,)) or not np.isfinite(p).all():
        raise ValueError('finite four/five-class probabilities required')
    if not available:
        return None
    token = feature['canonical']
    if token not in PAIR:
        raise ValueError('runtime s/ss expected')
    expected = 0 if token == 'ㅅ' else 1
    winner = int(p.argmax())
    if winner == expected and feature['baseline_warning']:
        threshold = policy['defer'].get(token)
        if threshold is not None and float(p[expected]) >= threshold:
            return {'kind': 'defer', 'probability': float(p[expected])}
    if winner == 1 - expected:
        threshold = policy['pair'].get(token)
        already = feature['baseline_warning'] and feature['baseline_actual'] == PAIR[token]
        if threshold is not None and not already and float(p[winner]) >= threshold:
            return {'kind': 'pair', 'actual': PAIR[token], 'probability': float(p[winner])}
    return None

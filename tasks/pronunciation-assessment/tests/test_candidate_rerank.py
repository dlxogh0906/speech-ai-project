import numpy as np
from mdd.candidate_rerank import features, budget_threshold, outputs


def test_budget_ties():
    s = np.array([1.]*3 + [-np.inf]*197)
    t, b = budget_threshold(s)
    assert b == 2 and (s >= t).sum() == 0
    t, b = budget_threshold(np.array([4., 3., 2.]+[-np.inf]*197))
    assert b == 2 and t > 2 and t < 3


def test_audio_reference_only_features_and_missing():
    vocab = ['<blank>', 'ㅅ', 'ㅆ', 'ㅏ']
    p = np.full((15, 4), .01); p[:, 0] = .97
    p[4] = [.01, .70, .28, .01]; p[10] = [.01, .01, .01, .97]
    x = np.sin(np.arange(6000)*.08).astype(np.float32)
    rows, hyp = features(x, np.log(p), 280+320*np.arange(15), ['ㅅ', 'ㅏ'], vocab)
    assert hyp == ['ㅅ', 'ㅏ'] and rows[0]['available']
    assert rows[0]['center'] == 1560 and rows[0]['vowel_center'] == 3480
    assert np.isfinite(rows[0]['segment']).all()
    # No TAG argument can influence peak selection. Missing emitted outputs stay missing.
    p[:] = [.97, .01, .01, .01]
    rows, _ = features(x, np.log(p), 280+320*np.arange(15), ['ㅅ', 'ㅏ'], vocab)
    assert not rows[0]['available']


def test_other_veto_and_one_phone_choice():
    class Model:
        classes_ = np.array(['OTHER', 'ㅅ', 'ㅆ'])
        def predict_log_proba(self, x):
            return np.log([[.7, .2, .1], [.1, .2, .7]])
    r = dict(base=[0.], original='ㅅ', expected='ㅅ')
    values = outputs(Model(), [r, r], 'base')
    assert values[0] == ('ㅅ', -np.inf, True)
    assert values[1][0] == 'ㅆ' and values[1][1] > 0 and not values[1][2]

import json
import numpy as np
from mdd.phone_probability_atlas import greedy_runs, ranked_frame


VOCAB = ['<blank>', 'ㅅ', 'ㅆ', 'ㅈ', '<pad>']


def test_blank_separates_two_identical_emissions():
    probabilities = np.array([
        [.1, .7, .1, .05, .05], [.05, .8, .05, .05, .05],
        [.8, .05, .05, .05, .05], [.1, .6, .2, .05, .05]])
    runs = greedy_runs(np.log(probabilities), VOCAB, 0)
    assert [(r['token'], r['start'], r['end'], r['peak']) for r in runs] == [
        ('ㅅ', 0, 2, 1), ('ㅅ', 3, 4, 3)]


def test_rank_uses_output_peak_and_preserves_full_softmax():
    probabilities = np.array([[.05, .6, .3, .04, .01], [.1, .8, .05, .04, .01]])
    lp = np.log(probabilities)
    run = greedy_runs(lp, VOCAB, 0)[0]
    assert run['peak'] == 1  # The true ㅆ maximum was at frame 0.
    ranked = ranked_frame(lp[run['peak']], VOCAB, 'ㅆ', 0)
    assert ranked['actual_rank_jamo'] == 2
    assert ranked['actual_rank_all_tokens'] == 3
    assert np.isclose(ranked['actual_probability'], .05)
    assert np.isclose(ranked['blank_probability'], .1)
    assert np.isclose(sum(r['probability'] for r in ranked['top_phones']), .89)
    json.dumps(ranked)  # Receipt must contain JSON-compatible scalar values.


def test_tied_probabilities_have_competition_rank():
    ranked = ranked_frame(np.log([.05, .4, .25, .25, .05]), VOCAB, 'ㅈ', 0)
    assert ranked['actual_rank_jamo'] == 2
    assert ranked['actual_tie_count_jamo'] == 2
    assert [r['phone'] for r in ranked['top_phones']] == ['ㅅ', 'ㅆ', 'ㅈ']

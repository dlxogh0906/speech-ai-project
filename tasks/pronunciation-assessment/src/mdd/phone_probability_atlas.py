"""Rank all CTC alternatives at a fixed, output-selected frame. No fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np


def jamo_ids(vocab):
    return [i for i, token in enumerate(vocab) if len(token) == 1 and '\u3131' <= token <= '\u3163']


def greedy_runs(lp, vocab, blank):
    """Collapse consecutive equal tokens BEFORE blank removal; keep frame spans."""
    if lp.ndim != 2 or lp.shape[1] != len(vocab) or not len(lp):
        raise ValueError('Nonempty frame x vocabulary log probabilities required')
    chosen = lp.argmax(axis=1); starts = np.r_[0, np.flatnonzero(np.diff(chosen)) + 1]
    ends = np.r_[starts[1:], len(lp)]; allowed = set(jamo_ids(vocab)); rows = []
    for start, end in zip(starts, ends):
        token = int(chosen[start])
        if token == blank or token not in allowed: continue
        peak = int(start + np.argmax(lp[start:end, token]))
        rows.append(dict(token=vocab[token], token_id=token, start=int(start), end=int(end),
                         peak=peak, probability=float(np.exp(lp[peak, token]))))
    return rows


def ranked_frame(log_probs, vocab, actual, blank):
    """Original softmax probabilities; never renormalize after excluding blank.

    Rank is competition rank: 1 + number of strictly higher probabilities.
    Ties are explicitly counted. Stable token-id order only orders display rows.
    """
    lp = np.asarray(log_probs)
    if lp.shape != (len(vocab),) or not np.isfinite(lp).all():
        raise ValueError('Finite full-vocabulary vector required')
    probabilities = np.exp(lp.astype(float))
    if not np.isclose(probabilities.sum(), 1., atol=2e-6):
        raise ValueError('Expected normalized full-vocabulary log probabilities')
    actual_id = vocab.index(actual); phones = jamo_ids(vocab)
    order = sorted(phones, key=lambda i: (-float(lp[i]), i))
    rank = lambda ids: 1 + sum(lp[i] > lp[actual_id] for i in ids)
    return dict(top_phones=[dict(phone=vocab[i], probability=float(probabilities[i])) for i in order[:5]],
        actual_probability=float(probabilities[actual_id]), actual_rank_jamo=int(rank(phones)),
        actual_rank_all_tokens=int(rank(range(len(vocab)))),
        actual_tie_count_jamo=int(sum(lp[i] == lp[actual_id] for i in phones)),
        blank_probability=float(probabilities[blank]), probability_sum=float(probabilities.sum()))


class ProbabilityAtlas:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.report = json.loads((self.folder/'report.json').read_text())
        self.rows = json.loads((self.folder/'cases.json').read_text())

    def summary_table(self):
        import pandas as pd
        return pd.DataFrame([{
            '실제 TAG': r['phone'], '오발음 전사율': '%d/%d (%.1f%%)' % (r['error_correct'],r['error_n'],r['error_accuracy_pct']),
            '전사 실패': r['misses'], '순위 확인': r['ranked'], '삭제 출력': r['deleted'],
            '재현 불일치': r['unreproduced'],
            '실제음 2위': '%d/%d' % (r['rank2'],r['ranked']),
            '실제음 3위 이내': '%d/%d' % (r['rank_le3'],r['ranked']),
            '실제음 순위 중앙값': r['median_rank'],
            '실제음 확률 중앙값(%)': round(r['median_actual_probability']*100,3) if r['median_actual_probability'] is not None else None
        } for r in self.report['summary']])

    def case_table(self, phone=None, examples=False, known_ss=False):
        import pandas as pd
        rows = [r for r in self.rows if r['status']=='ranked' and (phone is None or r['actual']==phone)]
        if known_ss:
            keys = set(self.report['previous_ss_case_keys'])
            rows = [r for r in rows if r['key'] in keys]
        elif examples:
            rows = [r for r in rows if r['key'] in self.report['example_keys']]
        out = []
        for r in rows:
            top = r['top_phones']
            fmt = lambda x: '%s %.3f%%' % (x['phone'],100*x['probability'])
            out.append({'문장':r['prompt'], '기대 LAB':r['expected'], '실제 TAG':r['actual'], '모델 출력':r['predicted'],
                '1위':fmt(top[0]), '2위':fmt(top[1]), '3위':fmt(top[2]),
                '실제음 순위':r['actual_rank_jamo'], '실제음 확률(%)':round(100*r['actual_probability'],3),
                'blank(%)':round(100*r['blank_probability'],3), '경고':'있음' if r['warning'] else '없음',
                '사례 ID':r['key']})
        return pd.DataFrame(out)

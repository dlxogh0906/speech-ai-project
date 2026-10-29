"""Sum CTC paths for C/null alternatives in the observed local phone context.

The local decoder's other phones and observed V stay fixed. All consonant
identities and no-C compete; canonical consonant and TAG never enter scoring.
Scores are conditional on this candidate family, not calibrated phone truth.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import collections
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_observed_anchor import observed_tokens
from mdd.sibilant_transition import VOWELS


def hypotheses(query, vocab, frames):
    if query['state'] not in ('local_C_V', 'local_V_without_C_token'):
        return []
    v = query['local_V_index']
    tokens = list(query['tokens'])
    if tokens[v] not in VOWELS:
        raise ValueError('observed local vowel required')
    has_c = query['actual'] is not None
    position = v - 1 if has_c else v
    if has_c and (position < 0 or tokens[position] != query['actual'] or tokens[position] in VOWELS):
        raise ValueError('local predecessor consonant identity differs')
    if has_c:
        del tokens[position]
    consonants = sorted((t for t in vocab if t not in VOWELS), key=lambda t: vocab[t])
    result = []
    for actual in [None] + consonants:
        seq = list(tokens)
        if actual is not None:
            seq.insert(position, actual)
        ids = [vocab[t] for t in seq]
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if minimum <= frames:
            result.append((actual, ids))
    if not result:
        raise ValueError('no feasible original/null CTC sequence')
    return result


def score_batch(log_probs, lengths, queries, vocab, blank):
    owners, alternatives, sequences = [], [], []
    for i, q in enumerate(queries):
        for actual, ids in hypotheses(q, vocab, int(lengths[i])):
            owners.append(i)
            alternatives.append(actual)
            sequences.append(ids)
    ps = np.full((len(queries), 5), .2, np.float32)
    available = np.zeros(len(queries), bool)
    detail = [{'state': 'local_V_unavailable'} for _ in queries]
    if not owners:
        return ps, available, detail
    ix = torch.tensor(owners, dtype=torch.long)
    lp = torch.as_tensor(log_probs, dtype=torch.float32)[ix]
    target_lengths = torch.tensor([len(s) for s in sequences])
    flat = torch.tensor([v for s in sequences for v in s])
    input_lengths = torch.as_tensor(lengths, dtype=torch.long)[ix]
    with torch.backends.cudnn.flags(enabled=False):
        nll = torch.nn.functional.ctc_loss(lp.transpose(0, 1), flat, input_lengths, target_lengths,
                                         blank=blank, reduction='none', zero_infinity=False)
    if not bool(torch.isfinite(nll).all()):
        raise ValueError('nonfinite feasible CTC hypothesis likelihood')
    for i, q in enumerate(queries):
        positions = [k for k, owner in enumerate(owners) if owner == i]
        if not positions:
            continue
        posterior = (-nll[positions]).softmax(-1).numpy()
        choices = [alternatives[k] for k in positions]
        coarse = np.zeros(5, np.float32)
        for actual, value in zip(choices, posterior):
            klass = 4 if actual is None else 0 if actual == 'ㅅ' else 1 if actual == 'ㅆ' else 2 if actual in ('ㅈ', 'ㅉ', 'ㅊ') else 3
            coarse[klass] += value
        ps[i] = coarse
        a, b = q['spans'][q['local_V_index']]
        vowel = q['tokens'][q['local_V_index']]
        v_conf = float(np.exp(log_probs[i, a:b, vocab[vowel]]).max())
        winners = np.flatnonzero(posterior == posterior.max())
        available[i] = v_conf >= .5 and len(winners) == 1
        detail[i] = {'state': 'context_hypothesis_supported' if available[i] else 'weak_or_ambiguous_context',
                     'chosen': choices[int(winners[0])] if len(winners) == 1 else None,
                     'V_confidence': v_conf, 'hypotheses': choices, 'conditional_posterior': posterior.tolist(),
                     'negative_log_likelihood': nll[positions].tolist(),
                     'no_C_is_phonemic_no_output_not_silence': True}
    return ps, available, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen input changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    queries = service.read_jsonl(p['queries'])
    lp = np.load(p['log_probs'])['log_probs']
    lengths = np.load(Path(p['cache']) / 'data.npz')['lengths']
    vocabulary = json.loads(Path(p['vocab']).read_text())
    vocab, blank = vocabulary['vocab'], vocabulary['blank']
    dictionary = {i: t for t, i in vocab.items()}
    class Dictionary:
        def __getitem__(self, i):
            return dictionary.get(i, '<special>')
    for i, q in enumerate(queries):
        tokens, spans = observed_tokens(lp[i, :lengths[i]], Dictionary(), blank)
        if tokens != q['tokens'] or [list(s) for s in spans] != q['spans']:
            raise ValueError('stored local decoder sequence differs from probabilities')
    torch.set_num_threads(2)
    probabilities, available, details = [], [], []
    tick = time.monotonic()
    for start in range(0, len(queries), p['batch']):
        z, av, ds = score_batch(lp[start:start + p['batch']], lengths[start:start + p['batch']],
                              queries[start:start + p['batch']], vocab, blank)
        probabilities.extend(z)
        available.extend(av)
        details.extend(ds)
    np.savez(out / 'scores.npz', probabilities=np.asarray(probabilities), available=np.asarray(available))
    with (out / 'hypotheses.jsonl').open('x') as f:
        for q, z in zip(queries, details):
            service.write_line(f, dict(uid=q['uid'], q_runtime=q['q_runtime'], **z))
    report = {'scope': p['scope'], 'seconds': time.monotonic() - tick, 'inputs': len(queries),
              'available': int(sum(available)), 'states': dict(collections.Counter(z['state'] for z in details)),
              'TAG_or_expected_C_input': False, 'deployment': False, 'scores_sha256': service.sha(out / 'scores.npz')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

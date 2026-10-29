"""Decode original V-reference clips with the frozen full phone CTC model.

No canonical consonant hypothesis, TAG time/identity or fitted local head enters
recognition. Match the observed global V token to the nearest local V token.
Token positions are decoder events, never physical boundaries or glottal onsets.
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
from transformers import Wav2Vec2BertForCTC
import eval_sibilant_directional as service
from mdd.sibilant_observed_anchor import observed_tokens
from mdd.sibilant_transition import VOWELS
from mdd.sibilant_local_ctc_artifact import load_artifact


def classify(lp, feature, relative, dictionary, blank):
    tokens, spans = observed_tokens(lp, dictionary, blank)
    anchor = int(np.abs(relative).argmin())
    options = [(abs((spans[j][0] + spans[j][1] - 1) / 2. - anchor), j)
               for j, token in enumerate(tokens) if token == feature['observed_V']]
    detail = {'tokens': tokens, 'spans': spans, 'anchor': anchor, 'state': 'local_V_missing',
              'actual': None, 'confidence': None}
    if not options:
        return None, detail
    options.sort()
    if len(options) > 1 and options[0][0] == options[1][0]:
        detail['state'] = 'local_V_ambiguous'
        return None, detail
    distance, j = options[0]
    a, b = spans[j]
    vp = float(np.exp(lp[a:b, dictionary.index(tokens[j])]).max())
    c = tokens[j - 1] if j and tokens[j - 1] not in VOWELS else None
    confidence = vp
    if c is not None:
        left, right = spans[j - 1]
        confidence = min(vp, float(np.exp(lp[left:right, dictionary.index(c)]).max()))
    klass = 4 if c is None else 0 if c == 'ㅅ' else 1 if c == 'ㅆ' else 2 if c in ('ㅈ', 'ㅉ', 'ㅊ') else 3
    # This vector encodes a greedy decision and its weakest token confidence;
    # it is not a calibrated class posterior or a physical deletion probability.
    ps = np.full(5, (1. - confidence) / 4., np.float32)
    ps[klass] = confidence
    detail.update(state='local_C_V' if c is not None else 'local_V_without_C_token',
                  actual=c, confidence=confidence, local_V_index=j, V_distance_frames=distance,
                  coarse_class=klass)
    return ps, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    cache = Path(p['cache'])
    data = dict(np.load(cache / 'data.npz'))
    if set(data) != {'x', 'relative_vowel', 'lengths'}:
        raise ValueError('V-reference acoustic cache required')
    features = service.read_jsonl(cache / 'data.features.jsonl')
    vocab = json.loads((Path(p['checkpoint']) / 'jamo_vocab.json').read_text())
    model = Wav2Vec2BertForCTC.from_pretrained(p['checkpoint']).to('cuda').eval()
    if p.get('adaptation'):
        artifact = torch.load(p['adaptation'], map_location='cpu')
        load_artifact(model.wav2vec2_bert.encoder.layers, model.lm_head, artifact,
                      service.sha(Path(p['checkpoint']) / 'jamo_vocab.json'))
    model.half()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    dictionary = ['<special>' + str(i) for i in range(model.config.vocab_size)]
    for token, i in vocab['vocab'].items():
        dictionary[i] = token
    torch.set_num_threads(2)
    probabilities, available, details, saved = [], [], [], []
    tick = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(features), p['batch']):
            x = torch.as_tensor(data['x'][start:start + p['batch']], device=device,
                                dtype=next(model.parameters()).dtype)
            length = torch.as_tensor(data['lengths'][start:start + p['batch']], device=device)
            mask = torch.arange(x.shape[1], device=device)[None, :] < length[:, None]
            logits = model(input_features=x, attention_mask=mask.long()).logits
            if logits.shape[:2] != x.shape[:2]:
                raise ValueError('local CTC grid differs')
            lp = logits.float().log_softmax(-1).cpu().numpy()
            for k in range(len(x)):
                i, n = start + k, int(length[k])
                ps, detail = classify(lp[k, :n], features[i], data['relative_vowel'][i, :n], dictionary, vocab['blank'])
                probabilities.append(np.full(5, .2) if ps is None else ps)
                available.append(ps is not None)
                details.append(dict(uid=features[i]['uid'], q_runtime=features[i]['q_runtime'], **detail))
                saved.append(lp[k])
            if p['batch'] == 1 and (start + 1) % 500 == 0:
                print('single requests', start + 1, 'seconds', round(time.monotonic() - tick, 1), flush=True)
    np.savez(out / 'scores.npz', probabilities=np.asarray(probabilities), available=np.asarray(available))
    np.savez(out / 'local_log_probs.npz', log_probs=np.asarray(saved))
    with (out / 'queries.jsonl').open('x') as stream:
        for z in details:
            service.write_line(stream, z)
    report = {'scope': p['scope'], 'frozen_full_phone_CTC': not bool(p.get('adaptation')),
              'adaptation': p.get('adaptation'), 'TAG_inputs': False, 'deployment': False,
              'inputs': len(features), 'batch': p['batch'], 'precision': 'original_service_fp16',
              'input_padding': 'fixed31feature_cache_with_observed_prefix_attention_mask',
              'seconds': time.monotonic() - tick,
              'states': dict(collections.Counter(z['state'] for z in details)),
              'greedy_actual': dict(collections.Counter(str(z['actual']) for z in details)),
              'source_sha256': p['sha256'], 'scores_sha256': service.sha(out / 'scores.npz')}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

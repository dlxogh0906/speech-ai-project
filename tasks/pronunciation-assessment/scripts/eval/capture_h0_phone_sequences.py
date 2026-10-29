"""Recover complete greedy sequences for the existing H0 selection audit.

No optimization, threshold fitting, test-set evaluation or production changes.
Every recovered canonical projection must equal the original sites ledger.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]

import argparse
import gc
import itertools
import json
import time
import numpy as np
import torch
from train_h0_backbone_control import Backbone, file_sha
from train_w2vbert_ctc import read_split, read_wav
from eval_actual_ctc import project_on_canon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--experiments', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit-utterances', type=int, default=0,
                    help='Execution check only; 0 recovers every selection utterance')
    a = ap.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    bindings = json.loads(Path(protocol['input_audit']).read_text())['sha256']
    manifest = Path(protocol['stage2']['manifest'])
    exp, out = Path(a.experiments), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    input_data = {}
    for split in ('valid', 'valid_unseen'):
        for ext in ('tsv', 'ltr', 'canon', 'meta.jsonl'):
            f = manifest / (split + '.' + ext)
            if file_sha(f) != bindings[str(f)]:
                raise ValueError('manifest changed: ' + str(f))
        audio, actual = read_split(str(manifest), split)
        canon = [line.split() for line in (manifest / (split + '.canon')).read_text().splitlines()]
        input_data[split] = audio, actual, canon
    start = time.time()
    receipt = dict(protocol_sha256=file_sha(a.protocol), training_updates=0,
                   selection_only=True, smoke=bool(a.limit_utterances), conditions={})
    for kind in ('w2vbert', 'xlsr300m'):
        for step in (2000, 4000):
            name = '%s_eval_%d' % (kind, step)
            report = json.loads((exp / name / 'report.json').read_text())
            checkpoint = exp / (kind + '_stage2') / ('step_%d.pt' % step)
            if file_sha(checkpoint) != report['checkpoint_sha256']:
                raise ValueError('checkpoint changed')
            ledger = {}
            for line in (exp / name / 'sites.jsonl').open():
                s = json.loads(line)
                if s['role'] == 'selection':
                    ledger.setdefault((s['split'], s['row']), []).append(s)
            keys = sorted(ledger)
            if a.limit_utterances:
                # Keep both seen and unseen represented in the execution check.
                keys = [k for split in ('valid', 'valid_unseen')
                        for k in [k for k in keys if k[0] == split][:a.limit_utterances]]
            ck = torch.load(str(checkpoint), map_location='cpu')
            if ck['protocol_sha256'] != receipt['protocol_sha256'] or ck['stage'] != 'stage2':
                raise ValueError('checkpoint protocol/stage mismatch')
            vocab = ck['vocab']; inv = {v: k for k, v in vocab.items()}
            model = Backbone(kind, protocol['backbones'][kind]['path'], max(vocab.values()) + 1,
                             torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(ck['model'], strict=True); model.eval(); del ck
            count = 0
            with (out / (name + '.jsonl')).open('x') as fw, torch.inference_mode():
                for split, row in keys:
                    old = sorted(ledger[(split, row)], key=lambda s: s['q'])
                    audio, actual, canon = input_data[split]
                    path, frames = audio[row]; x = read_wav(path, 20.)
                    if len(x) != frames:
                        raise ValueError('truncated audio')
                    with torch.cuda.amp.autocast(enabled=model.device.type == 'cuda'):
                        logits, lengths = model([x])
                    lp = logits[0, :int(lengths[0])].float().log_softmax(-1).cpu().numpy()
                    if not np.isfinite(lp).all():
                        raise ValueError('nonfinite logits')
                    pred = [inv[t] for t, _ in itertools.groupby(lp.argmax(-1)) if t in inv]
                    if project_on_canon(canon[row], pred) != [s['predicted'] for s in old]:
                        raise ValueError('recovered greedy projection differs: %s/%d' % (split, row))
                    if canon[row] != [s['expected'] for s in old]:
                        raise ValueError('canonical identity differs')
                    record = dict(split=split, row=row, uid=old[0]['uid'], speaker=old[0]['speaker'],
                                  item=old[0]['item'], canonical=canon[row], actual=actual[row], predicted=pred)
                    fw.write(json.dumps(record, ensure_ascii=False) + '\n'); count += 1
                    if count % 100 == 0:
                        fw.flush()
                        status = dict(condition=name, utterances=count, target=len(keys),
                                      elapsed_seconds=time.time() - start)
                        print(json.dumps(status), flush=True)
                        (out / 'live_status.json').write_text(json.dumps(status, indent=2))
            receipt['conditions'][name] = dict(utterances=count, projection_exact_match=True,
                checkpoint_sha256=report['checkpoint_sha256'],
                sites_sha256=file_sha(exp / name / 'sites.jsonl'),
                sequences_sha256=file_sha(out / (name + '.jsonl')))
            del model, logits, lp, ledger
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    receipt['elapsed_seconds'] = time.time() - start
    (out / 'finished.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    (out / 'live_status.json').write_text(json.dumps(dict(status='finished', **receipt), indent=2))


if __name__ == '__main__':
    main()

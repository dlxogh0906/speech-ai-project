"""Token lengths and complete normal-sequence predictions by prompt.

Descriptive follow-up to the all-phone audit; no model or threshold selection.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequences', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(a.sequences); out = Path(a.out)
    receipt = json.loads((root/'finished.json').read_text())
    if receipt['smoke']: raise ValueError('not a full capture')
    out.mkdir(parents=True, exist_ok=False)
    result = {}; rows = []
    for name, spec in receipt['conditions'].items():
        path = root/(name+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sequences_sha256']:
            raise ValueError('sequence hash mismatch: '+name)
        lengths = collections.defaultdict(collections.Counter)
        normals = collections.Counter(); predictions = collections.Counter()
        for line in path.open():
            r = json.loads(line); v = lengths[r['split']]
            for key in ('canonical','actual','predicted'): v[key+'_tokens'] += len(r[key])
            v['utterances'] += 1
            if r['actual'] == r['canonical']:
                key = (r['split'],r['item'],''.join(r['canonical']))
                normals[key] += 1
                predictions[key+(''.join(r['predicted']),)] += 1
        result[name] = dict(lengths)
        for (split,item,canon,pred), n in predictions.items():
            rows.append(dict(condition=name,split=split,item=item,canonical=canon,predicted=pred,
                             n=n,TAG_normal_utterances=normals[(split,item,canon)]))
    with (out/'normal_sequence_confusions.tsv').open('w') as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    report = dict(lengths=result, scope='TAG-complete-normal utterances; exploratory; no listening validation',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sequence_receipt_sha256=hashlib.sha256((root/'finished.json').read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

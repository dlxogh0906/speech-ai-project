"""Select label-independent new TRAIN normals in original draw contexts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import gzip
import heapq
import itertools
import json
import time
import numpy as np
from prepare_sibilant_rank_pilot import priority
from inventory_sibilant_extra_normals import pcm_hashes
import eval_sibilant_directional as service


def key(row, site):
    return (row['item'], site['q'], site.get('expected', site.get('canon')),
            site['following'], row['utterance_all_correct'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for file, digest in p['sha256'].items():
        if service.sha(file) != digest:
            raise ValueError('frozen pilot source changed: '+file)
    inventory = json.loads(Path(p['inventory_report']).read_text())
    if not inventory['proceed_new_normal_pilot']:
        raise ValueError('uncapped source inventory gate failed')
    rows = {z['uid']:z for z in service.read_jsonl(p['old_train'])}
    features = service.read_jsonl(p['old_features'])
    labels = {z['feature_index']:z for z in service.read_jsonl(p['old_labels']) if z['feature_index'] is not None}
    targets = service.read_jsonl(p['old_ctc_targets'])
    plan = np.load(p['old_plan'])
    needed = collections.Counter()
    for i in plan.ravel():
        if targets[i]['target'] is not None and labels[i]['label_kind'] == 'correct':
            needed[key(rows[features[i]['uid']], labels[i])] += 1
    pools, considered = collections.defaultdict(list), collections.Counter()
    tick = time.monotonic()
    with gzip.open(p['inventory_rows'], 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            for k in {key(row, site) for site in row['sites']} & set(needed):
                considered[k] += 1
                entry = (-priority(row['uid']), row['uid'], row)
                if len(pools[k]) < p['per_context']:
                    heapq.heappush(pools[k], entry)
                elif entry[0] > pools[k][0][0]:
                    heapq.heapreplace(pools[k], entry)
    selected = {entry[1]:entry[2] for pool in pools.values() for entry in pool}
    if not selected:
        raise ValueError('no source rows in original normal draw contexts')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(a.protocol).read_text())
    with (out/'rows.jsonl').open('x') as stream:
        for row in sorted(selected.values(), key=lambda z:priority(z['uid'])):
            full, prefix = pcm_hashes(row)
            if (full, prefix) != (row['full_PCM16_sha256'], row['runtime_prefix_PCM16_sha256']):
                raise ValueError('selected original PCM changed')
            service.write_line(stream, row)
    annotations = set()
    with gzip.open(p['annotation_cache'], 'rt', encoding='utf-8') as stream, (out/'annotations.jsonl').open('x') as dest:
        for line in stream:
            meta = json.loads(line)
            uid = meta['stem']
            if uid in selected and uid not in annotations:
                row = selected[uid]
                if meta['ltr'] != row['actual'] or meta['canon'] != row['canonical']:
                    raise ValueError('original TAG cache changed')
                service.write_line(dest, {k:meta[k] for k in ('stem', 'ltr', 'canon', 'tag_phones')})
                annotations.add(uid)
    if annotations != set(selected):
        raise ValueError('selected TAG metadata missing')
    verified = set()
    manifest = Path(p['manifest'])
    with (manifest/'train.tsv').open() as ft, (manifest/'train.ltr').open() as fl:
        next(ft)
        for t, ltr in itertools.zip_longest(ft, fl):
            if t is None or ltr is None:
                raise ValueError('official TRAIN row/label length differs')
            uid = Path(t.split('\t')[0]).stem
            if uid in selected:
                if selected[uid]['actual'] != ltr.split():
                    raise ValueError('official TRAIN actual label differs: '+uid)
                verified.add(uid)
    if verified != set(selected):
        raise ValueError('selected UID absent from official TRAIN')
    report = {'scope':__doc__, 'utterances':len(selected), 'speakers':len({z['speaker'] for z in selected.values()}),
        'items':len({z['item'] for z in selected.values()}), 'whole_utterance_correct':sum(z['utterance_all_correct'] for z in selected.values()),
        'other_phone_errors_retained':sum(not z['utterance_all_correct'] for z in selected.values()),
        'needed_contexts':len(needed), 'contexts_with_new_rows':len(pools),
        'normal_draw_slots':sum(needed.values()),
        'contexts':[{'key':list(k), 'original_draws':n, 'new_eligible':considered[k], 'selected':len(pools[k])} for k, n in sorted(needed.items())],
        'selected_original_PCM_and_train_ltr_verified':len(verified), 'new_model_or_admission_selection':False,
        'all_sites':sum(len(z['sites']) for z in selected.values()), 'seconds':time.monotonic()-tick,
        'sha256':{name:service.sha(out/name) for name in ('rows.jsonl', 'annotations.jsonl')},
        'goal_complete':False, 'deployment':False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps({k:v for k,v in report.items() if k != 'contexts'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

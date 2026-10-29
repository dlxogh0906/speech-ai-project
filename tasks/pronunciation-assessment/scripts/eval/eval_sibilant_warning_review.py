"""Render a frozen warning-deferral policy through original API responses.

All original evaluation outcomes, including rejects, remain in denominators.
Only the inactive adapter changes responses; production sources stay unchanged.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import collections
import json
import time
import numpy as np
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import restored_sites, detail
from eval_speech_rescue import observe, metrics
from mdd.sibilant_warning_review import review_queries, defer_warnings


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
    app = app_from(p)
    app.S['dec'].model.to('cpu')
    report = {'scope': p['scope'], 'deployment': False, 'thresholds': p['thresholds'], 'cohorts': {}}
    for name, dataset in p['evaluation_datasets'].items():
        features = service.read_jsonl(dataset['features_prefix'] + '.features.jsonl')
        warnings = service.read_jsonl(dataset['features_prefix'] + '.warnings.jsonl')
        scores = np.load(dataset['scores'])
        by_uid = collections.defaultdict(list)
        for i, feature in enumerate(features):
            by_uid[feature['uid']].append(i)
        rows, changes, timings, counts = [], [], [], collections.Counter()
        counts.update({'warning_sites': len(warnings), 'measured_warning_sites': len(features),
                       'unmeasured_warnings_preserved': sum(w['feature_index'] is None for w in warnings),
                       'strict_support_available': int(scores['available'].sum())})
        with (out / (name + '.responses.jsonl')).open('x') as stream:
            for record in service.read_jsonl(dataset['prefix'] + '.responses.jsonl'):
                row, base = record['row'], record['prior']
                candidate, reviews = dict(base), {}
                indices = by_uid[row['uid']]
                counts['utterances'] += 1
                counts['input_rejected_utterances'] += int(bool(base['rejected']))
                if indices:
                    raw = base['raw']
                    phones = service.phonemes(base['body'])
                    sites = restored_sites(raw, base['body'], p['pair_thresholds'])
                    if render(app, row, raw, sites) != base['body']:
                        raise AssertionError('prior complete API body did not reproduce')
                    counts['complete_API_bodies_reproduced'] += 1
                    fs = [features[i] for i in indices]
                    reviews = review_queries(raw['canon'], phones, fs, scores['probabilities'][indices],
                                             scores['available'][indices], p['thresholds'])
                    tick = time.monotonic()
                    candidate['body'] = defer_warnings(app.S['ss'], base['body'], dict(raw, sites=sites), reviews, app.S['vowels'])
                    timings.append(time.monotonic() - tick)
                    after = service.phonemes(candidate['body'])
                    for q, (before, proposed) in enumerate(zip(phones, after)):
                        if before.get('actual_jamo') != proposed.get('actual_jamo') or before.get('pronunciation') != proposed.get('pronunciation'):
                            raise AssertionError('heard token or original phone score changed')
                        if q not in reviews and before != proposed:
                            raise AssertionError('unselected phone changed')
                    counts['deferred_warnings'] += len(reviews)
                views = {'original': observe(record['original'], row), 'reference': observe(record['existing'], row),
                         'prior': observe(base, row), 'candidate': observe(candidate, row)}
                for site in row['sites']:
                    value = dict(site, expected=site['canon'], uid=row['uid'], item=row['item'], speaker=row['speaker'],
                                 language=row.get('language'))
                    value.update({key: view[site['q']] for key, view in views.items()})
                    rows.append(value)
                    if value['prior'] != value['candidate']:
                        changes.append(value)
                service.write_line(stream, {'uid': row['uid'], 'body': candidate['body'], 'rejected': candidate['rejected'],
                                             'deferred_positions': reviews})
        with (out / (name + '.sites.jsonl')).open('x') as stream:
            for value in rows:
                service.write_line(stream, value)
        pair_rows = [dict(r, baseline=r['prior']) for r in rows]
        result = detail(rows, counts)
        result['pair_directions'] = metrics(pair_rows)
        result['changes'] = changes
        result['adapter_latency_ms_p50_p95'] = (1000 * np.quantile(timings, [.5, .95])).tolist() if timings else None
        report['cohorts'][name] = result
        (out / 'report.json').write_text(json.dumps(service.jsonable(report), ensure_ascii=False, indent=2) + '\n')
        print(name, json.dumps(result['versus_prior'], ensure_ascii=False), flush=True)
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()

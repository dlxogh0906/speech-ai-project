"""Render frozen V-reference decisions with original scores and API aggregates."""
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
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import restored_sites, detail
from eval_speech_rescue import observe, metrics
from mdd.sibilant_presence_policy import decision
from mdd.sibilant_presence_response import apply_decisions


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
    report = {'scope': p['scope'], 'deployment': False, 'policy': p['policy'], 'cohorts': {}}
    for name, dataset in p['evaluation_datasets'].items():
        cache = Path(dataset['cache'])
        features = service.read_jsonl(cache / 'data.features.jsonl')
        scores = np.load(dataset['scores'])
        by_uid = collections.defaultdict(list)
        for i, f in enumerate(features):
            by_uid[f['uid']].append(i)
        rows, changes, timings, counts = [], [], [], collections.Counter()
        counts.update({'observed_inputs': len(features), 'V_role_available': int(scores['available'].sum())})
        with (out / (name + '.responses.jsonl')).open('x') as stream:
            for record in service.read_jsonl(cache / 'data.responses.jsonl'):
                row, base = record['row'], record['prior']
                candidate, actions = dict(base), {}
                counts['utterances'] += 1
                counts['input_rejected_utterances'] += bool(base['rejected'])
                indices = by_uid[row['uid']]
                if indices:
                    raw = base['raw']
                    sites = restored_sites(raw, base['body'], p['pair_thresholds'])
                    if render(app, row, raw, sites) != base['body']:
                        raise AssertionError('prior complete API body did not reproduce')
                    counts['complete_API_bodies_reproduced'] += 1
                    for i in indices:
                        action = decision(features[i], scores['probabilities'][i], scores['available'][i], p['policy'])
                        if action is not None:
                            actions[features[i]['q_runtime']] = action
                            counts[action['kind']] += 1
                            counts[action['kind'] + ('_C_linked' if features[i]['C_linked'] else '_C_unlinked')] += 1
                    tick = time.monotonic()
                    candidate['body'] = apply_decisions(app.S['ss'], base['body'], raw, actions, app.S['vowels'])
                    timings.append(time.monotonic() - tick)
                views = {'original': observe(record['original'], row), 'reference': observe(record['existing'], row),
                         'prior': observe(base, row), 'candidate': observe(candidate, row)}
                for site in row['sites']:
                    value = dict(site, expected=site['canon'], uid=row['uid'], item=row['item'], speaker=row['speaker'],
                                 language=row.get('language'))
                    value.update({key: view[site['q']] for key, view in views.items()})
                    rows.append(value)
                    if value['prior'] != value['candidate']:
                        changes.append(value)
                service.write_line(stream, {'uid': row['uid'], 'body': candidate['body'],
                                            'rejected': candidate['rejected'], 'actions': actions})
        with (out / (name + '.sites.jsonl')).open('x') as stream:
            for value in rows:
                service.write_line(stream, value)
        result = detail(rows, counts)
        result['pair_directions'] = metrics([dict(r, baseline=r['prior']) for r in rows])
        result['changes'] = changes
        result['adapter_latency_ms_p50_p95'] = (1000 * np.quantile(timings, [.5, .95])).tolist() if timings else None
        report['cohorts'][name] = result
        (out / 'report.json').write_text(json.dumps(service.jsonable(report), ensure_ascii=False, indent=2) + '\n')
        print(name, json.dumps(result['versus_prior'], ensure_ascii=False), flush=True)
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()

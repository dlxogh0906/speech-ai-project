"""Read-only membership, cached error-flow, cohorts and uncertainty audit."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'), str(Path(__file__).parent)]
import argparse
import collections
import copy
import gzip
import hashlib
import json
import time
import numpy as np
from mdd.sibilant_diagnostic_audit import (exact, flagged, error_stage,
    false_warning_stage, grouped_bootstrap, summary)
from eval_speech_rescue import observe


def read(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', p)
    tick = time.monotonic()
    report = {'scope': 'Offline existing-record audit. No new inference, training or protected-model scoring.',
              'partitions': {}, 'production_changed': False, 'reserved_scored': False}
    populations, records, labels, row_by_uid = {}, {}, {}, {}
    for name, spec in p['partitions'].items():
        rr = {}; ll = []
        if 'responses' in spec:
            for z in read(spec['responses']):
                row = z['row']; uid = row['uid']
                if uid in rr:
                    raise ValueError('duplicate source UID')
                rr[uid] = row; records.setdefault(name, {})[uid] = z
        else:
            rr = {z['uid']: z for z in read(spec['rows'])}
        if 'labels' in spec:
            ll = list(read(spec['labels']))
        populations[name] = set(rr); labels[name] = ll
        row_by_uid.update(rr)
        info = {'recordings': len(rr), 'source_split': dict(collections.Counter(z.get('source_split') for z in rr.values())),
                'language_recordings': dict(collections.Counter(z.get('language') for z in rr.values())),
                'speakers': len({z['speaker'] for z in rr.values()}),
                'items': len({z['item'] for z in rr.values()})}
        if not ll:
            report['partitions'][name] = info
            continue
        ledger = []
        features = {z['feature_index']: z for z in read(spec['features'])} if 'features' in spec else {}
        rawmap = {}
        if name in records:
            # Copy only per-site observations: do not re-infer audio or change policies.
            for uid, rec in records[name].items():
                raw = rec[spec.get('baseline', 'existing')]['raw']
                if raw:
                    rawmap[uid] = raw
        queries = {(z['uid'], z['q_runtime']): z for z in read(spec['queries'])} if 'queries' in spec else {}
        for z in ll:
            # Extra unmapped runtime queries are not new source GT sites.
            if name in ('D6', 'D7') and 'partial_q' not in z:
                continue
            v = copy.deepcopy(z['existing'])
            raw = rawmap.get(z['uid']); q = z.get('q_runtime')
            rs = raw['sites'][q] if raw and q is not None else None
            f = features.get(z.get('feature_index'))
            query = queries.get((z['uid'], q))
            entry = dict(z, view=v, raw_site=rs,
                query_state=(query or {}).get('state'),
                C_linked=(query or f or {}).get('C_linked'),
                has_auxiliary_input=z.get('feature_index') is not None)
            if z['label_kind'] not in ('correct', 'ambiguous'):
                entry['error_stage'] = error_stage(z, v, rs)
            if z['label_kind'] == 'correct':
                entry['false_warning_stage'] = false_warning_stage(z, v, rs)
            ledger.append(entry)
        info['summary'] = summary(ledger)
        info['view_states_by_kind'] = {k: dict(collections.Counter(z['view']['state'] for z in ledger if z['label_kind'] == k))
                                      for k in sorted({z['label_kind'] for z in ledger})}
        info['error_stages'] = dict(collections.Counter(z['error_stage'] for z in ledger if 'error_stage' in z))
        info['false_warning_stages'] = dict(collections.Counter(z['false_warning_stage'] for z in ledger if z.get('false_warning_stage')))
        info['no_auxiliary_input_by_kind'] = dict(collections.Counter(z['label_kind'] for z in ledger if not z['has_auxiliary_input']))
        info['directions'] = {t + '→' + alt: summary([z for z in ledger if z['expected'] == t and z.get('actual') == alt])
                              for t, alt in [('ㅅ', 'ㅆ'), ('ㅆ', 'ㅅ')]}
        info['language_sites'] = {lang: summary([z for z in ledger if z.get('language') == lang])
                                  for lang in sorted({z.get('language') or 'UNKNOWN' for z in ledger})}
        if name in ('D5', 'CAL', 'primary'):
            br = []
            for z in ledger:
                err = z['label_kind'] not in ('correct', 'ambiguous')
                normal = z['label_kind'] == 'correct'
                br.append({'speaker': z['speaker'], 'errors': int(err), 'normal': int(normal),
                    'normal_known': int(normal and z['view']['phone'] is not None),
                    'exact': int(exact(z, z['view'])), 'FP': int(normal and flagged(z['view'])),
                    'rejected_error': int(err and z['view']['state'] == 'input_rejected')})
            info['speaker_bootstrap'] = grouped_bootstrap(br, {'exact_recall': ('exact', 'errors'),
                'normal_alert_rate_all': ('FP', 'normal'), 'normal_alert_rate_known': ('FP', 'normal_known'),
                'error_rejection_rate': ('rejected_error', 'errors')}, p['bootstrap_repetitions'], p['seed'])
        with (out / (name + '.sites.jsonl')).open('x') as stream:
            for z in ledger:
                stream.write(json.dumps(z, ensure_ascii=False) + '\n')
        report['partitions'][name] = info
        print('accounted', name, len(rr), len(ledger), flush=True)
    selected = set().union(*populations.values())
    membership = collections.defaultdict(dict)
    for key, spec in p['membership_manifests'].items():
        with Path(spec['tsv']).open() as ft, Path(spec['ltr']).open() as fl:
            next(ft)
            for line, target in zip(ft, fl):
                uid = Path(line.split('\t')[0]).stem
                if uid in selected:
                    d = membership[uid].setdefault(key, {'rows': 0, 'target_variants': {}})
                    d['rows'] += 1
                    label = ' '.join(target.split()); hh = hashlib.sha256(label.encode()).hexdigest()
                    d['target_variants'][hh] = label
        print('membership scanned', key, flush=True)
    for name, uids in populations.items():
        info = report['partitions'][name]
        info['manifest_membership'] = {key: {'recordings_in_manifest': sum(key in membership[u] for u in uids),
            'matched_rows_with_repetition': sum(membership[u].get(key, {}).get('rows', 0) for u in uids)}
            for key in p['membership_manifests']}
    with (out / 'membership_by_uid.jsonl').open('x') as stream:
        for uid in sorted(selected):
            stream.write(json.dumps({'uid': uid, 'memberships': membership[uid]}, ensure_ascii=False) + '\n')
    metadata = {}
    with gzip.open(p['parsed_source_cache'], 'rt') as stream:
        for line in stream:
            # Filter JSON strings before decoding the expensive TAG arrays.
            marker = line.split('"stem":', 1)
            if len(marker) != 2:
                continue
            uid = marker[1].lstrip().split('"', 2)[1]
            if uid in selected:
                z = json.loads(line)
                metadata[uid] = {k: z.get(k) for k in ('proficiency', 'language', 'ds', 'part', 'dropped')}
    with Path(p['audio_index']).open() as stream:
        next(stream)
        for line in stream:
            cols = line.rstrip().split('\t')
            if cols[0] in selected:
                metadata.setdefault(cols[0], {}).update(src_sr=cols[1], src_ch=cols[2])
    for name, uids in populations.items():
        info = report['partitions'][name]
        info['proficiency_recordings'] = dict(collections.Counter(metadata.get(u, {}).get('proficiency') for u in uids))
        info['source_sample_rate_recordings'] = dict(collections.Counter(metadata.get(u, {}).get('src_sr') for u in uids))
        info['source_channels_recordings'] = dict(collections.Counter(metadata.get(u, {}).get('src_ch') for u in uids))
        info['source_dropped_marker_recordings'] = sum(bool(metadata.get(u, {}).get('dropped')) for u in uids)
    write(out / 'selected_source_metadata.json', metadata)
    report['seconds'] = time.monotonic() - tick
    report['limits'] = ['Manifest membership is not a per-update exposure ledger.',
        'No audio was independently listened to; disagreement is not adjudicated TAG error.',
        'Source split/name and missing input do not establish physical pronunciation causes.',
        'Bootstrap conditions on frozen outputs; repeated selection/CAL use is not corrected.']
    write(out / 'report.json', report)
    write(out / 'finished.json', {'report_sha256': sha(out / 'report.json')})
    print('finished', report['seconds'], flush=True)


if __name__ == '__main__':
    main()

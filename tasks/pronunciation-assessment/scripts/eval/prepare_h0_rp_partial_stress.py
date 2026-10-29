"""Prepare unscored RP inputs and conservative partial-annotation support.

No model import, audio inference, label repair or complete-case selection occurs.
This does not implement calibrated GOP or insertion-gap scoring for unknown spans.
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
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import os
import time
from mdd.h0_partial_reference import (
    UNKNOWN, partial_tokens, partial_mapping, classify_truth, input_windows)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.%s.tmp' % os.getpid())
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(str(temp), str(path))


def emit(f, value):
    f.write(json.dumps(value, ensure_ascii=False) + '\n')


def analyze(job):
    record, vocabulary = job
    ref, owners, ignored_ref = partial_tokens(record['lab_phones'], vocabulary, reference=True)
    tag, tag_owners, ignored_tag = partial_tokens(record['tag_phones'], vocabulary)
    distance, options = partial_mapping(ref, tag)
    sites = []
    for i, (phone, values, owner) in enumerate(zip(ref, options, owners)):
        kind, actual = classify_truth(phone, values)
        sites.append(dict(index=i, expected=phone, possible_actual=None if values is None else sorted(values),
                          status=kind, identified_actual=actual, owner=owner))
    return dict(uid=record['stem'], item=record['item'], speaker=record['spk'],
                reference=ref, actual=tag, actual_owners=tag_owners, sites=sites,
                ignored_reference=ignored_ref, ignored_actual=ignored_tag,
                wildcard_edit_cost=distance, model_scored=False,
                independent_listening=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readiness', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    started = time.time()
    if args.out.exists():
        raise FileExistsError('preserve prior results: choose a new output directory')
    r = args.readiness
    candidates_path = r / 'reserved_unscored_candidates.jsonl'
    records_path = r / 'source_records.jsonl.gz'
    audit_path = r / 'recordings.jsonl'
    report = json.loads((r / 'report.json').read_text())
    vocabulary = report['vocabulary']
    candidates = [json.loads(line) for line in candidates_path.open()]
    uids = {c['uid'] for c in candidates}
    if len(uids) != len(candidates):
        raise AssertionError('duplicate candidate UID')
    args.out.mkdir(parents=True)
    sources = [Path(__file__).resolve(), ROOT / 'tasks/pronunciation-assessment/src/mdd/h0_partial_reference.py']
    protocol = dict(version='RP_partial_support_v1', created=datetime.datetime.now().astimezone().isoformat(),
        user_constraint='no additional independent audio/TAG/LAB data available',
        scope='unscored long-form stress input and partial-annotation preparation',
        candidates=len(candidates), no_model_scores=True, optimizer_updates=0,
        input='12 s fixed cores, 400 ms context on each side; all audio and tails retained',
        annotation='whole-recording original order; unknown labels are zero-or-more wildcard spans',
        mask='annotation-only all minimum edit paths; unknown or nonunique actual is unidentifiable',
        metrics='site-identifiable conditional counts only; unknown reference span lengths remain unknown',
        inference_plan='join core-owned CTC frames before collapsing; no TAG boundary cropping',
        limits=['not an independent final evaluation', 'three source scripts',
                'no physical boundary or listening ground truth', 'no calibrated GOP or insertion policy yet',
                'wildcard edit cost is not a phone error rate'],
        input_bindings={str(x): sha(x) for x in (candidates_path, records_path, audit_path,
                                                r / 'report.json', r / 'closure.json')},
        code_bindings={str(x.relative_to(ROOT)): sha(x) for x in sources})
    write(args.out / 'protocol.json', protocol)

    # Freeze input geometry before loading or aligning the source annotations.
    input_counts = collections.Counter()
    with (args.out / 'inputs.jsonl').open('w', encoding='utf8') as f:
        for candidate in candidates:
            windows = input_windows(candidate['frames'])
            if sum(w['core_end'] - w['core_start'] for w in windows) != candidate['frames']:
                raise AssertionError('input samples lost')
            for i, window in enumerate(windows):
                emit(f, dict(uid=candidate['uid'], path=candidate['path'], chunk=i, **window))
            input_counts['recordings'] += 1
            input_counts['samples'] += candidate['frames']
            input_counts['windows'] += len(windows)
            input_counts['read_samples_including_context'] += sum(w['read_end'] - w['read_start'] for w in windows)
    write(args.out / 'inputs_frozen.json', dict(counts=dict(input_counts),
        sha256=sha(args.out / 'inputs.jsonl'), labels_used=False, model_scores_used=False))
    audits = {}
    for line in audit_path.open():
        a = json.loads(line)
        if a['uid'] in uids:
            audits[a['uid']] = a
    if set(audits) != uids or any(not a['raw_matches_cache'] or a.get('failure') for a in audits.values()):
        raise AssertionError('source parse receipt missing or failed')
    for c in candidates:
        a = audits[c['uid']]
        for field in ('frames', 'pcm_sha256', 'source_hashes', 'path'):
            if c[field] != a[field]:
                raise AssertionError('candidate binding mismatch: ' + field)
    records = {}
    with gzip.open(str(records_path), 'rt', encoding='utf8') as f:
        for line in f:
            x = json.loads(line)
            if x['stem'] in uids:
                records[x['stem']] = x
    if set(records) != uids:
        raise AssertionError('source record missing')
    counts = collections.defaultdict(collections.Counter)
    phones = collections.defaultdict(collections.Counter)
    observed = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = ((records[c['uid']], vocabulary) for c in candidates)
        with gzip.open(str(args.out / 'annotation_records.jsonl.gz'), 'wt', encoding='utf8') as f:
            for index, result in enumerate(pool.map(analyze, jobs, chunksize=4), 1):
                uid = result['uid']; observed.add(uid)
                a = audits[uid]
                result['source_timing_flags'] = dict(tag=a['tag_check']['issues'], lab=a['lab_check']['issues'])
                result['source_hashes'] = a['source_hashes']
                result['audio_pcm_sha256'] = a['pcm_sha256']
                emit(f, result)
                for key in ('all', result['item']):
                    cc = counts[key]
                    cc['recordings'] += 1
                    cc['timing_flagged_recordings'] += int(bool(a['tag_check']['issues'] or a['lab_check']['issues']))
                    cc['unknown_actual_spans'] += result['actual'].count(UNKNOWN)
                    cc['explicit_deletion_intervals'] += sum(x['reason'] == 'explicit_deletion' for x in result['ignored_actual'])
                    for site in result['sites']:
                        cc[site['status']] += 1
                        cc['reference_entries_including_unknown_spans'] += 1
                        cc['known_reference_tokens'] += int(site['expected'] != UNKNOWN)
                for site in result['sites']:
                    phones[(result['item'], site['expected'])][site['status']] += 1
                if index % 100 == 0:
                    write(args.out / 'live_status.json', dict(status='partial_annotation_preparation',
                          completed=index, total=len(candidates), optimizer_updates=0, model_scored=False))
                    print(json.dumps(dict(completed=index, total=len(candidates))), flush=True)
    if observed != uids:
        raise AssertionError('lost candidate')
    with (args.out / 'phone_support.tsv').open('w', encoding='utf8') as f:
        statuses = ['normal', 'substitution', 'deletion', 'alignment_ambiguous', 'unknown_actual_possible', 'unknown_reference']
        f.write('item\tphone\t' + '\t'.join(statuses) + '\n')
        for item in sorted({c['item'] for c in candidates}):
            for phone in vocabulary + [UNKNOWN]:
                cc = phones[(item, phone)]
                f.write('\t'.join([item, phone] + [str(cc[s]) for s in statuses]) + '\n')
    result = dict(status='unscored_partial_support_prepared', counts=dict(counts), inputs=dict(input_counts),
                  candidate_count=len(observed), original_audio_samples_retained=True,
                  no_complete_case_selection=True, new_independent_data_available=False,
                  model_inferences=0, optimizer_updates=0,
                  all_phone_service_evaluation_ready=False,
                  pending=['long input runtime/frame stitching parity',
                           'partial reference GOP and deletion/insertion policy',
                           'score-independent protocol freeze before C0/C1/C2 evaluation'],
                  elapsed_seconds=time.time()-started,
                  limitations=protocol['limits'])
    write(args.out / 'report.json', result)
    write(args.out / 'finished.json', dict(status=result['status'], model_inferences=0, optimizer_updates=0,
          bindings={x: sha(args.out / x) for x in ('protocol.json', 'inputs.jsonl',
                    'annotation_records.jsonl.gz', 'phone_support.tsv', 'report.json')}))
    write(args.out / 'live_status.json', dict(status=result['status'], completed=len(observed),
          model_scored=False, optimizer_updates=0))


if __name__ == '__main__':
    main()

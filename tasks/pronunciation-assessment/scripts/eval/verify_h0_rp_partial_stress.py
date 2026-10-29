"""Recount partial support independently; verify original owners and all inputs."""
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
import csv
import gzip
import hashlib
import json
from mdd.h0_partial_reference import UNKNOWN, UNOBSERVED, DELETED


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readiness', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    out = args.out
    if (out / 'verification.json').exists():
        raise FileExistsError('verification receipt already exists')
    protocol = json.loads((out / 'protocol.json').read_text())
    for path, digest in protocol['input_bindings'].items():
        assert sha(path) == digest, path
    for path, digest in protocol['code_bindings'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in json.loads((out / 'finished.json').read_text())['bindings'].items():
        assert sha(out / path) == digest, path
    frozen = json.loads((out / 'inputs_frozen.json').read_text())
    assert sha(out / 'inputs.jsonl') == frozen['sha256']
    candidates = {x['uid']: x for x in map(json.loads, (args.readiness / 'reserved_unscored_candidates.jsonl').open())}
    inputs = collections.defaultdict(list)
    for x in map(json.loads, (out / 'inputs.jsonl').open()):
        inputs[x['uid']].append(x)
    assert set(inputs) == set(candidates)
    total_samples, read_samples, n_windows = 0, 0, 0
    for uid, windows in inputs.items():
        c = candidates[uid]; previous = 0
        for i, w in enumerate(windows):
            assert w['path'] == c['path'] and w['chunk'] == i
            assert w['core_start'] == previous == i * 192000
            assert w['core_end'] == min(c['frames'], (i + 1) * 192000)
            assert w['read_start'] == max(0, w['core_start'] - 6400)
            assert w['read_end'] == min(c['frames'], w['core_end'] + 6400)
            total_samples += w['core_end'] - w['core_start']
            read_samples += w['read_end'] - w['read_start']
            previous = w['core_end']; n_windows += 1
        assert previous == c['frames']
    with gzip.open(str(args.readiness / 'source_records.jsonl.gz'), 'rt', encoding='utf8') as f:
        sources = {x['stem']: x for x in map(json.loads, f) if x['stem'] in candidates}
    counts = collections.defaultdict(collections.Counter)
    phones = collections.defaultdict(collections.Counter)
    seen = set()
    with gzip.open(str(out / 'annotation_records.jsonl.gz'), 'rt', encoding='utf8') as f:
        for x in map(json.loads, f):
            uid = x['uid']; assert uid not in seen; seen.add(uid)
            original = sources[uid]; c = candidates[uid]
            assert x['source_hashes'] == c['source_hashes']
            assert x['audio_pcm_sha256'] == c['pcm_sha256']
            assert x['model_scored'] is False and x['independent_listening'] is False
            assert len(x['reference']) == len(x['sites'])
            for owner in [s['owner'] for s in x['sites']]:
                a, b, label = original['lab_phones'][owner['interval']]
                assert [owner['start'], owner['end'], owner['raw_label']] == [a, b, str(label).strip()]
            for owner in x['actual_owners']:
                a, b, label = original['tag_phones'][owner['interval']]
                assert [owner['start'], owner['end'], owner['raw_label']] == [a, b, str(label).strip()]
            for key in ('all', x['item']):
                counts[key]['recordings'] += 1
                counts[key]['timing_flagged_recordings'] += int(any(x['source_timing_flags'].values()))
                counts[key]['unknown_actual_spans'] += sum(t == UNKNOWN for t in x['actual'])
                counts[key]['explicit_deletion_intervals'] += sum(t['reason'] == 'explicit_deletion' for t in x['ignored_actual'])
            for i, site in enumerate(x['sites']):
                exp, values = site['expected'], site['possible_actual']
                assert site['index'] == i and exp == x['reference'][i]
                actual = None
                if exp == UNKNOWN:
                    status = 'unknown_reference'; assert values is None
                elif UNOBSERVED in values:
                    status = 'unknown_actual_possible'
                elif len(values) != 1:
                    status = 'alignment_ambiguous'; assert len(values) > 1
                else:
                    actual = values[0]
                    status = 'normal' if actual == exp else ('deletion' if actual == DELETED else 'substitution')
                assert (status, actual) == (site['status'], site['identified_actual'])
                for key in ('all', x['item']):
                    counts[key][status] += 1
                    counts[key]['reference_entries_including_unknown_spans'] += 1
                    counts[key]['known_reference_tokens'] += int(exp != UNKNOWN)
                phones[(x['item'], exp)][status] += 1
    assert seen == set(candidates)
    report = json.loads((out / 'report.json').read_text())
    assert dict(counts) == report['counts']
    assert dict(recordings=len(seen), samples=total_samples, windows=n_windows,
                read_samples_including_context=read_samples) == report['inputs'] == frozen['counts']
    with (out / 'phone_support.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            for name, value in row.items():
                if name not in ('item', 'phone'):
                    assert int(value) == phones[(row['item'], row['phone'])][name]
    vocabulary = json.loads((args.readiness / 'report.json').read_text())['vocabulary']
    missing = [p for p in vocabulary if sum(c['normal'] for (item, phone), c in phones.items() if phone == p) == 0]
    result = dict(status='verified_unscored_inputs_and_partial_annotation_support',
        recordings=len(seen), windows=n_windows, full_audio_samples=total_samples,
        known_reference_tokens=counts['all']['known_reference_tokens'],
        zero_identified_normal_symbols=missing, original_annotation_owners_match=True,
        counts_recomputed_without_mapping_function=True, model_inferences=0, optimizer_updates=0,
        verifier_sha256=sha(Path(__file__).resolve()))
    (out / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

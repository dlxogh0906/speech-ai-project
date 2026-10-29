"""Check frozen native-reference contexts before any acoustic model scoring."""
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
import hashlib
import json
import math
import eval_sibilant_directional as service
from mdd.sibilant_projection_residual import speaker_half


def select_contexts(rows, seed):
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    states = []
    for row in rows:
        c, v = row['label_consonant_interval'], row['label_vowel_interval']
        state = 'eligible_context'
        if (row['phone'] not in ('s0', 'ss') or not all(math.isfinite(x) for x in c+v)
                or not 0 <= c[0] < c[1] <= v[0] < v[1]):
            state = 'invalid_target_annotation'
        elif not row.get('previous_phone') or row['previous_phone'].startswith('<'):
            state = 'previous_phone_unknown_or_marked'
        states.append({'id':row['id'], 'state':state})
        if state != 'eligible_context':
            continue
        key = (row['previous_phone'], row['following_vowel'],
               row['at_labeled_word_start'], row['at_labeled_utterance_start'])
        grouped[key][row['speaker']].append(row)
    ledger, selected, repeated = [], [], 0
    for key, speakers in sorted(grouped.items(), key=lambda z:str(z[0])):
        chosen = []
        for speaker, values in sorted(speakers.items()):
            choice = min(values, key=lambda r:hashlib.sha256(
                ('native-control-dedup-v1:'+r['id']).encode()).digest())
            chosen.append(choice)
            repeated += len(values)-1
        halves = [[r for r in chosen if speaker_half(r['speaker'],seed)==half] for half in (0,1)]
        counts = [[sum(r['phone']==phone for r in half) for phone in ('s0','ss')] for half in halves]
        valid = all(all(n>0 for n in half) for half in counts)
        entry = {'context':list(key), 'ids':[r['id'] for r in chosen],
            'halves':[[r['id'] for r in half] for half in halves],
            'class_counts_s0_ss':counts, 'both_classes_both_halves':valid}
        ledger.append(entry)
        if valid:
            selected.append(entry)
    return selected, ledger, states, repeated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen native inventory input changed: '+name)
    selection = json.loads(Path(p['selection']).read_text())
    manifest = json.loads(Path(p['audio_manifest']).read_text())
    if selection['speakers'] != 40 or selection['total_cv_candidates'] != 8015:
        raise ValueError('original native reference selection changed')
    recordings = {r['recording']:r for r in selection['recordings']}
    if len(recordings) != 40 or len({r['speaker'] for r in recordings.values()}) != 40:
        raise ValueError('one selected recording per original speaker required')
    rows = [r for r in service.read_jsonl(p['index']) if r['recording'] in recordings]
    if len(rows) != 8015 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('original 8015 metadata rows required')
    for r in rows:
        if r['speaker'] != recordings[r['recording']]['speaker']:
            raise ValueError('native source speaker differs')
    selected, ledger, states, repeated = select_contexts(rows, p['seed'])
    by_vowel = collections.Counter(c['context'][1] for c in selected)
    balance = collections.defaultdict(collections.Counter)
    for row in rows:
        balance[row['speaker']][row['phone']+'/'+row['following_vowel']] += 1
    passed = len(selected) >= 50 and by_vowel['ii'] >= 10
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.protocol).read_bytes())
    for name,value in [('qualified_contexts.json',selected),('all_contexts.json',ledger),('row_states.json',states)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    report = {'scope':p['scope'],'source_reference_rows':len(rows), 'source_reference_speakers':40,
        'source_reference_recordings':40,'all_context_cells':len(ledger),'qualified_contexts':len(selected),
        'qualified_contexts_by_vowel':dict(by_vowel),'selected_diagnostic_rows':sum(len(c['ids']) for c in selected),
        'same_speaker_context_repeats_removed':repeated,'row_states':dict(collections.Counter(z['state'] for z in states)),
        'balanced_source_counts':{k:dict(v) for k,v in balance.items()},
        'audio_header_rates_from_existing_manifest':dict(collections.Counter(r['sample_rate'] for r in manifest['recordings'])),
        'predeclared_minimum_contexts':50,'predeclared_minimum_i_contexts':10,'proceed_native_control_scoring':bool(passed),
        'no_audio_or_model_scoring':True,'no_model_training':True,'deployment':False,'goal_complete':False}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps({k:v for k,v in report.items() if k!='balanced_source_counts'},ensure_ascii=False))


if __name__ == '__main__':
    main()

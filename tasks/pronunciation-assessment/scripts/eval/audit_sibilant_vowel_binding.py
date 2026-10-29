"""TRAIN slot-binding diagnosis and full-outcome runtime observation floors.

TAG enters post-inference diagnosis only. A token's time outside a TAG interval
is reported separately and is not equated with a wrong acoustic boundary.
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
import math
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from mdd.sibilant_transition import PAIR, VOWELS
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_frame_clock import stacked_window_centers


def stream(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def crop_actual_indices(tag, actual, selected):
    flat, indices = [], []
    selected = set(selected)
    for i, (_, _, token) in enumerate(tag):
        token = ''.join(c for c in token if not c.isspace())
        if token in ('', '.', '-', 'spn', '<unk>'):
            continue
        for c in token:
            if i in selected:
                indices.append(len(flat))
            flat.append(c)
    if flat != list(actual):
        raise ValueError('TAG actual sequence differs')
    return indices


def sequence_binding(actual, original_indices, vowel_index, query, roi):
    if query['state'] not in ('local_C_V', 'local_V_without_C_token'):
        return 'local_V_unavailable', None, None
    if vowel_index not in original_indices:
        return 'intended_V_outside_supported_sequence', None, None
    left, right = roi
    predicted_indices = [i for i, (a, b) in enumerate(query['spans']) if b > left and a < right]
    selected = query['local_V_index']
    if selected not in predicted_indices:
        return 'selected_V_outside_supported_sequence', None, None
    target = [actual[i] for i in original_indices]
    predicted = [query['tokens'][i] for i in predicted_indices]
    mapping = service.all_optimal_mapping(target, predicted)
    wanted = mapping[original_indices.index(vowel_index)]
    chosen = predicted_indices.index(selected)
    if len(wanted) != 1:
        return 'sequence_mapping_ambiguous', mapping, predicted_indices
    if wanted == {None}:
        return 'intended_V_not_recognized_in_sequence', mapping, predicted_indices
    if wanted != {chosen}:
        return 'selected_other_sequence_slot', mapping, predicted_indices
    return 'sequence_slot_consistent', mapping, predicted_indices


def binding(p, out):
    b = p['binding']
    cache = Path(b['cache'])
    features = service.read_jsonl(cache / 'data.features.jsonl')
    mapped_labels = [z for z in stream(cache / 'data.labels.jsonl') if z['feature_index'] is not None]
    labels = {z['feature_index']: z for z in mapped_labels}
    if len(mapped_labels) != len(features) or len(labels) != len(features):
        raise ValueError('this TRAIN diagnostic requires one preserved GT site per observed query')
    targets = service.read_jsonl(b['targets'])
    y = np.load(cache / 'targets.npz')['y']
    records = {z['row']['uid']: z['row'] for z in stream(cache / 'data.responses.jsonl')}
    annotations = {z['stem']: z for z in stream(b['annotations']) if z['stem'] in records}
    report = {}
    for model, path in b['queries'].items():
        queries = service.read_jsonl(path)
        if len(queries) != len(features):
            raise ValueError('inference feature count differs')
        rows, groups = [], collections.defaultdict(collections.Counter)
        for i, (f, q) in enumerate(zip(features, queries)):
            if (f['uid'], f['q_runtime']) != (q['uid'], q['q_runtime']):
                raise ValueError('query identity mismatch')
            label, row = labels[i], records[f['uid']]
            target = targets[i]['target']
            entry = {'feature_index': i, 'uid': f['uid'], 'q': label['q'], 'following': label['following'],
                     'class_supported': bool(y[i] >= 0), 'label_kind': label['label_kind'],
                     'state': 'supervision_region_unavailable', 'C_state': 'unresolved'}
            if target is not None:
                tag = annotations[f['uid']]['tag_phones']
                actual_indices = crop_actual_indices(tag, row['actual'], target['TAG_indices'])
                if [row['actual'][j] for j in actual_indices] != target['tokens']:
                    raise ValueError('trusted crop token indices differ')
                mapping = service.all_optimal_mapping(row['canonical'], row['actual'])
                options = mapping[label['q'] + 1]
                if len(options) != 1 or None in options or row['actual'][next(iter(options))] not in VOWELS:
                    entry['state'] = 'intended_V_identity_unknown'
                else:
                    vi = next(iter(options))
                    state, match, pred_indices = sequence_binding(row['actual'], actual_indices, vi, q, target['roi'])
                    entry.update(state=state, actual_V=row['actual'][vi])
                    if state == 'sequence_slot_consistent':
                        entry['V_identity_matches_TAG'] = q['tokens'][q['local_V_index']] == row['actual'][vi]
                        chosen_c = q['local_V_index'] - 1 if q['actual'] is not None else None
                        if label['a'] is None:
                            entry['C_state'] = 'no_C_token_matches_TAG_omission' if chosen_c is None else 'C_token_before_TAG_omitted_C'
                        elif label['a'] not in actual_indices:
                            entry['C_state'] = 'target_C_outside_supported_sequence'
                        else:
                            possibilities = match[actual_indices.index(label['a'])]
                            pc = pred_indices.index(chosen_c) if chosen_c in pred_indices else None
                            if len(possibilities) != 1:
                                entry['C_state'] = 'C_sequence_mapping_ambiguous'
                            elif possibilities == {None}:
                                entry['C_state'] = 'target_C_not_recognized'
                            elif pc is None or possibilities != {pc}:
                                entry['C_state'] = 'chosen_preV_C_is_different_slot_or_insertion'
                            elif q['actual'] == label['actual']:
                                entry['C_state'] = 'C_identity_correct'
                            else:
                                entry['C_state'] = 'S_SS_swap' if label['actual'] in PAIR and q['actual'] == PAIR[label['actual']] else 'C_other_identity_confusion'
                    if q['state'] in ('local_C_V', 'local_V_without_C_token'):
                        ivs = single_phone_intervals(tag, row['actual'])
                        if ivs is not None and ivs[vi] is not None:
                            a, z, _ = ivs[vi]
                            first, last = q['spans'][q['local_V_index']]
                            centers = stacked_window_centers(f['left'], f['right'], f['input_frames']) / 16000.
                            midpoint = float(centers[first:last].mean())
                            entry.update(token_grid_center_in_TAG_V=bool(a <= midpoint < z),
                                         token_grid_offset_from_TAG_V_start=midpoint - a)
            for group in ['all', 'class_supported' if y[i] >= 0 else 'class_unknown',
                          'kind/' + label['label_kind'], 'vowel/' + label['following']]:
                groups[group]['inputs'] += 1
                groups[group]['V/' + entry['state']] += 1
                groups[group]['C/' + entry['C_state']] += 1
            rows.append(entry)
        with (out / (model + '.binding.jsonl')).open('x') as f:
            for row in rows:
                service.write_line(f, row)
        report[model] = {k: dict(v) for k, v in groups.items()}
    return report


def coverage(p, out):
    report = {}
    for name, spec in p['cohorts'].items():
        counts, by_vowel, cases = collections.Counter(), collections.defaultdict(collections.Counter), []
        exact_queries = {(z['uid'], z['q_runtime']): z for z in stream(spec['queries'])} if spec.get('queries') else None
        for record in stream(spec['records']):
            row, base = record['row'], record['prior']
            counts['utterances'] += 1
            counts['input_rejected_utterances'] += bool(base['rejected'])
            views = observe(base, row)
            raw = base['raw']
            mapping = service.all_optimal_mapping(row['canonical'], raw['canon']) if raw else None
            for label in row['sites']:
                if label['label_kind'] != 'correct':
                    counts['non_normal_sites_retained'] += 1
                    continue
                counts['normal_all'] += 1
                phone = views[label['q']]['phone']
                counts['normal_known'] += phone is not None
                if not (phone or {}).get('errorType'):
                    continue
                counts['baseline_normal_warnings'] += 1
                options = mapping[label['q']]
                rq = next(iter(options)) if len(options) == 1 and None not in options else None
                eligible = rq is not None and rq + 1 < len(raw['canon']) and raw['canon'][rq] in PAIR and raw['canon'][rq + 1] in VOWELS
                c_link = bool(eligible and raw['sites'][rq].get('j') is not None)
                v_link = bool(eligible and raw['sites'][rq + 1].get('j') is not None and raw['sites'][rq + 1].get('pred') in VOWELS)
                state = ('ineligible_runtime_slot' if not eligible else
                         ('C_linked' if c_link else 'C_unlinked') + '/' + ('V_linked' if v_link else 'V_unlinked_or_nonvowel'))
                counts[state] += 1
                by_vowel[label['following']][state] += 1
                counts['optimistically_observable_V'] += v_link
                if exact_queries is not None:
                    query = exact_queries.get((row['uid'], rq))
                    counts['exact_observable_V'] += query is not None and query['feature_index'] is not None
                if not v_link:
                    left = next((j for j in range(rq - 1, -1, -1) if raw['sites'][j].get('j') is not None), None) if eligible else None
                    right = next((j for j in range(rq + 2, len(raw['sites'])) if raw['sites'][j].get('j') is not None), None) if eligible else None
                    bracket = 'two_neighbor_links' if left is not None and right is not None else 'right_neighbor_only' if right is not None else 'left_neighbor_only' if left is not None else 'no_neighbor_links'
                    counts['unobserved/' + bracket] += 1
                    cases.append({'uid': row['uid'], 'q': label['q'], 'q_runtime': rq, 'item': row['item'],
                                  'following': label['following'], 'C_linked': c_link, 'state': state,
                                  'left_runtime_reference': left, 'right_runtime_reference': right,
                                  'unknown_slots_between_references': right - left - 1 if left is not None and right is not None else None})
        n, fp = counts['normal_known'], counts['baseline_normal_warnings']
        floor = fp - counts['optimistically_observable_V']
        item = {'counts': dict(counts), 'target_false_warning_budget_known': math.floor(.01 * n),
                'optimistic_fixed_reference_V_observer_floor': {'remaining': floor, 'rate_known': floor / n if n else None,
                    'rate_all': floor / counts['normal_all'] if counts['normal_all'] else None,
                    'over_1_percent_known': floor > math.floor(.01 * n)},
                'normal_warning_vowels': {k: dict(v) for k, v in by_vowel.items()}}
        if exact_queries is not None:
            floor = fp - counts['exact_observable_V']
            item['exact_input_scope_floor'] = {'remaining': floor, 'rate_known': floor / n if n else None,
                                              'rate_all': floor / counts['normal_all'] if counts['normal_all'] else None}
        with (out / (name + '.unobserved_normal_warnings.jsonl')).open('x') as f:
            for row in cases:
                service.write_line(f, row)
        report[name] = item
        print(name, json.dumps(item, ensure_ascii=False), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen input changed: ' + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(Path(a.protocol).read_text())
    report = {'scope': p['scope'], 'deployment': False, 'new_service_policy': False,
              'TRAIN_binding': binding(p, out), 'observation_coverage': coverage(p, out)}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'finished.json').write_text(json.dumps({'sha256': service.sha(out / 'report.json')}) + '\n')


if __name__ == '__main__':
    main()

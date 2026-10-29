"""Privileged TAG-location versus identical-length runtime-location diagnosis."""
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
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_vowel_reference import annotated_intended_vowel_interval
from mdd.sibilant_frozen_probe import load_probe, project_observed
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_trajectory_abx import unit_rows, angular_matrix, dtw
from mdd.sibilant_location_abx import crop_indices, retained_abx, location_result


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen location diagnostic input changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only diagnosis required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.protocol).read_bytes())
    tick = time.monotonic()
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    by = {z['feature_index']: z for z in labels if z['feature_index'] is not None}
    y = np.load(p['targets'])['y']
    cells = json.loads(Path(p['cells']).read_text())
    selected = sorted({i for cell in cells for i in cell['indices']})
    wanted = {features[i]['uid'] for i in selected}
    annotations = {z['stem']: z for z in service.read_jsonl(p['annotations']) if z['stem'] in wanted}
    geometry = {}
    for z in service.read_jsonl(cache/'data.responses.jsonl'):
        row = z['row']
        if row['uid'] in wanted:
            tag = annotations[row['uid']]
            if tag['spk'] != row['speaker'] or tag['item'] != row['item']:
                raise ValueError('annotation identity differs')
            geometry[row['uid']] = (single_phone_intervals(tag['tag_phones'], row['actual']),
                service.all_optimal_mapping(row['canonical'], row['actual']))
    if set(geometry) != wanted:
        raise ValueError('missing original response/annotation source')
    original = torch.load(p['control_artifact'], map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    base = load_probe(original, p['binding'], device).requires_grad_(False).eval()
    fingerprint = tensor_digest(base.state_dict())
    raw = dict(np.load(p['data']))
    sequences, ledger, validity = {}, [], np.zeros(len(y), bool)
    grouped = {key: collections.defaultdict(collections.Counter) for key in
               ['following', 'position', 'item', 'language', 'actual', 'expected']}
    with torch.no_grad():
        for start in range(0, len(selected), 64):
            ix = selected[start:start+64]
            x = torch.as_tensor(raw['x'][ix], device=device)
            lengths = raw['lengths'][ix]
            mask = torch.arange(x.shape[1], device=device)[None] < torch.as_tensor(lengths, device=device)[:, None]
            projected = project_observed(x, mask, {k:getattr(base, k) for k in ('mean', 'scale', 'components')}).cpu().numpy()
            for j, i in enumerate(ix):
                f, lab = features[i], by[i]
                times = (f['frame_first_source_sample']+320*np.arange(int(lengths[j])))/16000.
                intervals, mapping = geometry[f['uid']]
                opts = mapping[lab['q']+1]
                adjacent = lab['a'] is not None and opts == {lab['a']+1}
                c = intervals[lab['a']] if intervals is not None and lab['a'] is not None else None
                v = annotated_intended_vowel_interval(intervals, opts)
                tag, control, state = crop_indices(times, c if adjacent else None, v, f['observed_reference_seconds'])
                if len(tag):
                    if c[2] != lab['actual'] or len(tag) != len(control):
                        raise ValueError('actual target or matched length changed')
                    try:
                        sequences[i] = [unit_rows(projected[j, idx]) for idx in [tag, control]]
                        validity[i] = True
                    except ValueError as exc:
                        state = 'measurement_failure:'+str(exc)
                entry = {k:lab.get(k) for k in ['uid','q','a','expected','actual','following','position','item','language']}
                entry.update(feature_index=i, state=state, valid=bool(validity[i]),
                    C_interval=c, V_interval=v, TAG_indices=tag.tolist(), control_indices=control.tolist())
                ledger.append(entry)
                for key in grouped:
                    grouped[key][str(lab.get(key))].update({'all':1, 'available':int(validity[i])})
    old = np.load(p['original_distances'])
    old_records = json.loads(Path(p['original_cell_scores']).read_text())
    distances, pairs, results, all_scores = [], [], [], []
    offset = 0
    for k, cell in enumerate(cells):
        left, right = cell['halves']
        n = len(left)*len(right)
        expected_pairs = np.asarray([(k,i,j) for i in left for j in right])
        np.testing.assert_array_equal(old['pairs'][offset:offset+n], expected_pairs)
        values = np.full((len(left), len(right), 2), .5)
        for a, i in enumerate(left):
            for b, j in enumerate(right):
                if validity[i] and validity[j]:
                    values[a,b] = [dtw(angular_matrix(sequences[i][mode], sequences[j][mode])) for mode in (0,1)]
        scores = [retained_abx(values[:,:,mode], y[left], y[right], validity[left], validity[right]) for mode in (0,1)]
        baseline = retained_abx(old['distances'][offset:offset+n,0].reshape(len(left),len(right)),
                               y[left], y[right], np.ones(len(left),bool), np.ones(len(right),bool))
        np.testing.assert_allclose(baseline['retained_accuracy'], old_records[k]['accuracy']['ordered'], atol=1e-12)
        all_scores.append([scores[0]['retained_accuracy'], scores[1]['retained_accuracy'], baseline['retained_accuracy']])
        results.append({'cell':cell['cell'], 'TAG':scores[0], 'control':scores[1], 'baseline':baseline})
        distances.extend(values.reshape(-1,2));pairs.extend(expected_pairs);offset += n
        if k%40 == 0:
            print(json.dumps({'cells':k+1,'pairs':offset,'seconds':time.monotonic()-tick}),flush=True)
    vowels = np.asarray([features[i]['observed_V'] for i in selected])
    availability = {'all':float(validity[selected].mean()), 'ㅣ':float(validity[np.asarray(selected)[vowels=='ㅣ']].mean())}
    result = location_result(all_scores, [c['cell'][3] for c in cells], availability, p['seed'])
    if tensor_digest(base.state_dict()) != fingerprint:
        raise ValueError('immutable encoder readout changed')
    write(out/'rows.json', ledger)
    write(out/'cells.json', results)
    np.savez(out/'distances.npz', pairs=np.asarray(pairs), distances=np.asarray(distances), available=validity)
    report = {'scope':p['scope'], 'source_flow':json.loads(Path(p['prior_audit_report']).read_text())['flow'],
        'diagnostic_rows':len(selected), 'states':dict(collections.Counter(z['state'] for z in ledger)),
        'availability':availability, 'coverage_groups':{k:{v:dict(c) for v,c in values.items()} for k,values in grouped.items()},
        'result':result, 'pair_distances':offset, 'model_unchanged':True, 'no_model_or_threshold_training':True,
        'TAG_location_and_length_are_privileged_not_service':True, 'seconds':time.monotonic()-tick,
        'deployment':False, 'goal_complete':False}
    write(out/'report.json', report)
    write(out/'finished.json', {'report_sha256':service.sha(out/'report.json')})
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

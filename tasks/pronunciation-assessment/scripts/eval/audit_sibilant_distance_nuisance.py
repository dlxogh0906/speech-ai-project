"""Attribute-matched diagnostics on existing TRAIN distances, no classifier."""
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
import json
import numpy as np
import eval_sibilant_directional as service


def paired_scores(distance, left, right, code_left, code_right):
    distance = np.asarray(distance)
    left, right = np.asarray(left), np.asarray(right)
    code_left, code_right = np.asarray(code_left), np.asarray(code_right)
    if (distance.shape != (len(left), len(right)) or code_left.shape != left.shape
            or code_right.shape != right.shape or not np.isfinite(distance).all()
            or any(set(z.tolist()) != {0, 1} for z in (left, right))):
        raise ValueError('same original two-class cross-speaker populations required')
    halves, counts, queries = [], 0, 0
    for d, q, r, qc, rc in [(distance,left,right,code_left,code_right),
                           (distance.T,right,left,code_right,code_left)]:
        classes = []
        for cls in (0, 1):
            values = []
            for i in np.flatnonzero(q == cls):
                queries += 1
                a = (r == cls) & (rc == qc[i]) & np.isin(rc, ['M', 'F'])
                b = (r != cls) & (rc == qc[i]) & np.isin(rc, ['M', 'F'])
                c = (r == cls) & (rc != qc[i]) & np.isin(rc, ['M', 'F'])
                eligible = qc[i] in ('M', 'F') and a.any() and b.any() and c.any()
                if eligible:
                    scores = []
                    for ref in [b, c]:
                        delta = d[i,a][:,None]-d[i,ref][None,:]
                        scores.append(float(((delta < 0)+.5*(delta == 0)).mean()))
                    counts += 1
                    values.append(scores)
                else:
                    values.append([.5, .5])
            classes.append(np.mean(values, axis=0))
        halves.append(np.mean(classes, axis=0))
    return np.mean(halves, axis=0), counts, queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen nuisance diagnostic changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('TRAIN-only diagnostics required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.protocol).read_bytes())
    cells = json.loads(Path(p['cells']).read_text())
    features = service.read_jsonl(p['features'])
    codes = np.asarray([f['uid'].split('-')[1] for f in features])
    y = np.load(p['targets'])['y']
    data = np.load(p['distances'])
    scores, rows, offset = [], [], 0
    for k, cell in enumerate(cells):
        left, right = cell['halves']
        n = len(left)*len(right)
        np.testing.assert_array_equal(data['pairs'][offset:offset+n],
            [(k,i,j) for i in left for j in right])
        matrix = data['distances'][offset:offset+n, 0].reshape(len(left),len(right))
        values, eligible, all_queries = paired_scores(matrix, y[left], y[right], codes[left], codes[right])
        scores.append(values)
        rows.append({'cell':cell['cell'], 'phonetic_accuracy':float(values[0]),
            'UID_code_accuracy':float(values[1]), 'joint_eligible_queries':eligible,
            'all_queries':all_queries})
        offset += n
    scores = np.asarray(scores)
    groups = {}
    vowels = np.asarray([c['cell'][3] for c in cells])
    for group in ('all', 'ㅣ', 'ㅡ', 'ㅜ'):
        mask = np.ones(len(rows),bool) if group == 'all' else vowels == group
        chosen = [z for z,keep in zip(rows,mask) if keep]
        eligible, total = sum(z['joint_eligible_queries'] for z in chosen),sum(z['all_queries'] for z in chosen)
        groups[group] = {'cells':len(chosen), 'cells_with_joint_queries':sum(z['joint_eligible_queries']>0 for z in chosen),
            'joint_eligible_queries':eligible, 'all_queries':total,
            'joint_availability':eligible/total, 'accuracy_phone_UIDcode':scores[mask].mean(0).tolist()}
    diff = scores[:,1]-scores[:,0]
    sample = np.random.RandomState(p['seed']).randint(0,len(rows),(2000,len(rows)))
    reference = np.percentile(diff[sample].mean(1),[2.5,50,97.5]).tolist()
    all_group, i_group = groups['all'],groups['ㅣ']
    passed = (diff.mean() >= .10 and reference[0] > 0 and diff[vowels=='ㅣ'].mean() >= 0
        and all_group['joint_availability'] >= .5 and i_group['joint_availability'] >= .5
        and all_group['cells_with_joint_queries'] >= 100 and i_group['cells_with_joint_queries'] >= 10)
    report = {'scope':p['scope'],'groups':groups,'unknown_UIDcodes':int((~np.isin(codes,['M','F'])).sum()),
        'code_minus_phone':float(diff.mean()),'paired_context_bootstrap_p025_p50_p975':reference,
        'supports_speaker_attribute_followup':bool(passed),'source_flow':json.loads(Path(p['source_report']).read_text())['source_flow'],
        'model_or_threshold_training':False,'deployment':False,'goal_complete':False}
    for name,value in [('report.json',report),('cells.json',rows)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()

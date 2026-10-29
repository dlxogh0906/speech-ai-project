"""CAL feasibility for rechecking existing warnings with frozen actual-class heads.

This produces proposals, never changes API responses. Every ambiguous, unknown,
non-normal or mismatched-reference warning is a negative safety control.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_context_rank import model, score_single
from eval_speech_rescue import observe, correct


def write(path, value):
    path.write_text(json.dumps(service.jsonable(value), ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, root, out = json.loads(Path(a.protocol).read_text()), Path(a.features), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    marker = json.loads((root / 'finished.json').read_text())
    if marker['sha256'] != service.sha(root / 'report.json'):
        raise ValueError('cache incomplete/changed')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(p['threads'])
    data = dict(np.load(root / 'cal.npz'))
    features = service.read_jsonl(root / 'cal.features.jsonl')
    warnings = service.read_jsonl(root / 'cal.warnings.jsonl')
    by_key = collections.defaultdict(list)
    full_counts = collections.Counter()
    for record in service.read_jsonl(p['complete_cal_prefix'] + '.responses.jsonl'):
        row, prior = record['row'], record['prior']
        viewed = observe(prior, row)
        raw = prior['raw']
        mapping = service.all_optimal_mapping(row['canonical'], raw['canon']) if raw else None
        full_counts['utterances'] += 1
        full_counts['input_rejected_utterances'] += int(bool(prior['rejected']))
        for site in row['sites']:
            obs = viewed[site['q']]
            normal = site.get('actual') == site['canon'] and not site.get('alignment_ambiguous')
            # Use actual identity as well, so label schema spelling cannot
            # silently omit errors.
            known_error = site.get('actual') != site['canon'] and site.get('label_kind') != 'ambiguous' and not site.get('alignment_ambiguous')
            alert = bool((obs['phone'] or {}).get('errorType'))
            full_counts['sites_all'] += 1
            full_counts['normal_all'] += int(normal)
            full_counts['normal_known_accepted'] += int(normal and obs['phone'] is not None)
            full_counts['normal_warnings'] += int(normal and alert)
            full_counts['known_error_all'] += int(known_error)
            exact = correct(dict(site, expected=site['canon'], baseline=obs), 'baseline')
            full_counts['exact_error_warnings'] += int(known_error and exact)
            if mapping is not None:
                options = mapping[site['q']]
                rq = next(iter(options)) if len(options) == 1 and None not in options else None
                if rq is not None:
                    by_key[(row['uid'], rq)].append(dict(site, expected=site['canon'],
                        state=obs['state'], normal=normal, baseline_exact=bool(exact),
                        item=row['item'], speaker=row['speaker'], language=row.get('language')))
    safety = []
    for warning in warnings:
        ls = by_key[(warning['uid'], warning['q_runtime'])]
        safe_normal = len(ls) == 1 and ls[0]['normal'] and ls[0]['state'] == 'available' and ls[0]['expected'] == warning['canonical']
        safety.append(dict(warning, normal_control=bool(safe_normal), label_views=ls))
    write(out / 'protocol.json', p)
    report = {'scope': 'CAL-only warning recheck proposals; not actual API suppression',
              'full_denominators': dict(full_counts), 'warnings': len(warnings), 'features': len(features),
              'inference_batch': 1, 'deployment': False, 'arms': {}}
    for arm, path in p['artifacts'].items():
        artifact = torch.load(path, map_location='cpu')
        if artifact.get('readout', 'existing') not in ('existing',):
            raise ValueError('this evaluator accepts the ordinary learned-support readout only')
        encoder, head = model(p, artifact)
        ps, av, seconds = score_single(encoder, head, data)
        np.savez(out / (arm + '.scores.npz'), probabilities=ps, available=av)
        arm_report, proposals = {'seconds': seconds, 'directions': {}}, []
        for token, class_index in [('ㅅ', 0), ('ㅆ', 1)]:
            measured = [w for w in safety if w['canonical'] == token and w['feature_index'] is not None]
            controls = [w for w in measured if not w['normal_control']]
            # Include even currently unavailable negative controls in the ceiling.
            # Availability can only make the decision more conservative.
            threshold = max(.5, float(np.nextafter(max(float(ps[w['feature_index'], class_index])
                                                       for w in controls), np.inf))) if controls else None
            selected = [w for w in measured if threshold is not None and av[w['feature_index']]
                        and int(ps[w['feature_index']].argmax()) == class_index
                        and float(ps[w['feature_index'], class_index]) >= threshold]
            if any(not w['normal_control'] for w in selected):
                raise AssertionError('CAL safety control would be deferred')
            for warning in selected:
                proposals.append(dict(warning, probability=float(ps[warning['feature_index'], class_index]), threshold=threshold))
            arm_report['directions'][token] = {'threshold': threshold, 'warning_sites_all': sum(w['canonical'] == token for w in safety),
                'measured': len(measured), 'strict_support_available': sum(bool(av[w['feature_index']]) for w in measured),
                'normal_controls': sum(w['normal_control'] for w in measured), 'negative_controls': len(controls),
                'known_exact_error_controls': sum(any(v['baseline_exact'] for v in w['label_views']) for w in controls),
                'normal_warning_deferral_proposals': len(selected), 'lost_exact_proposals': 0}
        arm_report['proposals'] = proposals
        report['arms'][arm] = arm_report
        print(arm, json.dumps(arm_report, ensure_ascii=False), flush=True)
        del encoder, head, artifact
        torch.cuda.empty_cache()
        write(out / 'report.json', report)
    write(out / 'safety_labels.json', safety)
    write(out / 'finished.json', {'sha256': service.sha(out / 'report.json')})


if __name__ == '__main__':
    main()

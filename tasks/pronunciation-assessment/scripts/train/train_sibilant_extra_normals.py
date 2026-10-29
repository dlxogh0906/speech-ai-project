"""Matched new-normal versus old-normal exposure on the immutable native summary.

Only normal recording identity changes. Error draws, update count, actual class,
context and whole-utterance correctness are identical between the two arms.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/eval'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/data')]
import argparse
import collections
import json
import math
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_presence_screen import exact
from train_sibilant_hubert_adaptation import retained_subgroups
from train_sibilant_precision_tail import summary_scores
from prepare_sibilant_extra_normal_pilot import key
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_precision_tail import PrecisionTailHead, tail_artifact, load_tail_head


def proxy(ps, av, y, features, indexed, budget):
    keys, curves = ('normal_FP', 'novel_TP', 'duplicate_TP', 'wrong_actual'), []
    for token, ci in [('ㅅ', 1), ('ㅆ', 0)]:
        groups = collections.defaultdict(collections.Counter)
        for i, cls in enumerate(y):
            if (cls < 0 or i not in indexed or features[i]['canonical'] != token
                    or not av[i] or ps[i].argmax() != ci or ps[i, ci] < .5):
                continue
            lab, values = indexed[i], groups[float(ps[i, ci])]
            if lab['label_kind'] == 'correct':
                values['normal_FP'] += 1
            elif lab['label_kind'] == 'pair_error' and lab['expected'] == token:
                values['duplicate_TP' if exact(lab, lab['existing']) else 'novel_TP'] += 1
            else:
                values['wrong_actual'] += 1
        counts = collections.Counter()
        curve = [dict(threshold=None, **{k:0 for k in keys})]
        for threshold in sorted(groups, reverse=True):
            counts.update(groups[threshold])
            if counts['normal_FP'] <= budget:
                curve.append(dict(threshold=threshold, **{k:counts[k] for k in keys}))
        curves.append(curve)
    def rank(pair):
        a, b = pair
        return (a['novel_TP']+b['novel_TP'], -a['normal_FP']-b['normal_FP'],
            -a['wrong_actual']-b['wrong_actual'], a['duplicate_TP']+b['duplicate_TP'])
    a, b = max(((a,b) for a in curves[0] for b in curves[1]
        if a['normal_FP']+b['normal_FP'] <= budget), key=rank)
    return dict({k:a[k]+b[k] for k in keys}, S_to_SS=a, SS_to_S=b,
        normal_budget=budget, scope='Expanded common TRAIN diagnostic only; not service 1percent')


def proposed(label, ps, av, features, policy):
    i = label['feature_index']
    if i is None:
        return False
    ci = 1 if features[i]['canonical'] == 'ㅅ' else 0
    threshold = policy['S_to_SS' if ci == 1 else 'SS_to_S']['threshold']
    return bool(threshold is not None and av[i] and ps[i].argmax() == ci and ps[i,ci] >= threshold)


def common_scores(model, data, offset, device):
    # Preserve the original batch64 boundary at the old/new source join. Joining
    # the populations must not turn the old final short batch into a larger GEMM.
    parts = [summary_scores(model, {k:v[left:right] for k,v in data.items()}, device)
        for left,right in ((0,offset),(offset,len(data['summary'])))]
    return tuple(np.concatenate([part[k] for part in parts]) for k in (0,1))


def matched_plans(original, y, indexed, features, rows, eligibility, offset, seed):
    pools = [collections.defaultdict(list), collections.defaultdict(list)]
    for i, label in sorted(indexed.items()):
        if (y[i] not in (0, 1) or label['label_kind'] != 'correct'
                or label['expected'] != features[i]['canonical']):
            continue
        # New rows were omitted by the old cap only when every target site was
        # normal. Give the old resampling control the same row-level property;
        # retain non-sibilant errors via the separately matched whole-correct flag.
        if any(z['label_kind'] != 'correct' for z in rows[label['uid']]['sites']):
            continue
        if i < offset and eligibility[i]['target'] is None:
            continue
        pools[int(i >= offset)][key(rows[label['uid']], label)].append(i)
    plans = [original.copy(), original.copy()]
    changed = 0
    rng = np.random.RandomState(seed)
    for position in np.ndindex(original.shape):
        i = int(original[position])
        if eligibility[i]['target'] is None:
            for plan in plans:
                plan[position] = -1
            continue
        label = indexed[i]
        if y[i] < 0:
            raise ValueError('original supported draw has unknown target')
        if label['label_kind'] != 'correct':
            continue
        k = key(rows[label['uid']], label)
        if not pools[1][k]:
            continue
        if not pools[0][k]:
            continue
        u = rng.random_sample()
        for arm in (0, 1):
            plans[arm][position] = pools[arm][k][int(u*len(pools[arm][k]))]
        changed += 1
    for position in np.ndindex(original.shape):
        old, new = [int(plan[position]) for plan in plans]
        if old < 0 or new < 0:
            if old != new:
                raise ValueError('support skip composition changed')
            continue
        if y[old] != y[new]:
            raise ValueError('actual class composition changed')
        if key(rows[indexed[old]['uid']], indexed[old]) != key(rows[indexed[new]['uid']], indexed[new]):
            raise ValueError('context/correctness composition changed')
        if indexed[old]['label_kind'] != 'correct' and old != new:
            raise ValueError('real error identity changed')
    return plans, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for name, digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen newnormal trial changed: '+name)
    old, new = np.load(p['old_summary']), np.load(Path(p['new_summary'])/'summary.npz')
    y = np.concatenate([np.load(p['old_targets'])['y'], new['y']])
    offset = len(old['summary'])
    data = {k:np.concatenate([old[k], new[k]]) for k in ('summary', 'moments', 'available')}
    features = service.read_jsonl(p['old_features'])
    for feature in service.read_jsonl(Path(p['new_summary'])/'data.features.jsonl'):
        features.append(dict(feature, feature_index=feature['feature_index']+offset))
    labels = service.read_jsonl(p['old_labels'])
    for label in service.read_jsonl(Path(p['new_summary'])/'data.labels.jsonl'):
        labels.append(dict(label, feature_index=None if label['feature_index'] is None else label['feature_index']+offset))
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    if len(indexed) != sum(z['feature_index'] is not None for z in labels):
        raise ValueError('multiple GT labels for one input')
    if len(features) != len(y) or any(z['feature_index'] != i for i,z in enumerate(features)):
        raise ValueError('expanded population differs')
    rows = {z['uid']:z for path in (p['old_train'], p['new_rows']) for z in service.read_jsonl(path)}
    original = np.load(p['old_plan'])
    eligibility = service.read_jsonl(p['old_ctc_targets'])
    if original.shape != (p['steps'], 16) or p['steps'] != 300:
        raise ValueError('fixed original300x16 plan required')
    plans, changed = matched_plans(original, y, indexed, features, rows, eligibility, offset, p['seed'])
    normal_count = sum(y[i] >= 0 and z['label_kind'] == 'correct' and z['expected'] == features[i]['canonical']
        for i,z in indexed.items())
    budget = math.floor(.01*normal_count)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(a.protocol).read_text())
    for name, plan in zip(('old_resampled', 'new_normals'), plans):
        np.save(out/(name+'.plan.npy'), plan)
    torch.set_num_threads(2)
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    artifact = torch.load(p['control_artifact'], map_location='cpu')
    native = load_probe(artifact, p['binding'], device)
    model = PrecisionTailHead(native, 'ce_full').to(device).eval()
    initial_ps, av = common_scores(model, data, offset, device)
    np.testing.assert_array_equal(initial_ps[:offset], np.load(p['old_scores'])['probabilities'])
    np.testing.assert_array_equal(initial_ps[offset:], new['probabilities'])
    np.testing.assert_array_equal(av, data['available'])
    output = {'scope':__doc__, 'inputs':len(y), 'all_sites':len(labels), 'old_inputs':offset,
        'new_inputs':len(y)-offset, 'normal_count_common':int(normal_count), 'normal_budget_common':budget,
        'normal_slots_with_new_data':changed, 'original_supported_non_normal_draws_identical':True,
        'class_context_wholecorrect_and_skip_composition_equal':True, 'initial_all_predictions_exact_parent':True,
        'parent':proxy(initial_ps, av, y, features, indexed, budget), 'arms':{},
        'deployment':False, 'goal_complete':False}
    all_ps = {'parent':initial_ps}
    for name, plan in zip(('old_resampled', 'new_normals'), plans):
        torch.manual_seed(p['seed'])
        model = PrecisionTailHead(load_probe(artifact, p['binding'], device), 'ce_full').to(device).eval()
        parameters = [z for z in model.parameters() if z.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=p['head_lr'], weight_decay=.01)
        before = tensor_digest(model.head.state_dict())
        losses, tick = [], time.monotonic()
        for indices in plan:
            ix = indices[indices >= 0]
            optimizer.zero_grad(set_to_none=True)
            logits = model.from_summary(torch.as_tensor(data['summary'][ix], device=device),
                torch.as_tensor(data['moments'][ix], device=device))
            loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[ix], device=device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite actual-phone training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        fitted = tail_artifact(model, artifact)
        if tensor_digest(model.head.state_dict()) == before:
            raise ValueError('training did not change weights')
        torch.save(fitted, out/(name+'.pt'))
        ps, current_av = common_scores(model, data, offset, device)
        np.testing.assert_array_equal(current_av, av)
        reloaded = load_tail_head(torch.load(out/(name+'.pt'), map_location='cpu'), p['binding'], 'ce_full', device)
        check, _ = summary_scores(reloaded, {k:v[:64] for k,v in data.items()}, device)
        np.testing.assert_array_equal(check, ps[:64])
        np.savez(out/(name+'.scores.npz'), probabilities=ps, available=av)
        result = proxy(ps, av, y, features, indexed, budget)
        output['arms'][name] = {'TRAIN_proxy':result, 'trainable_parameters':sum(z.numel() for z in parameters),
            'loss_first_last50':[float(np.mean(losses[:50])),float(np.mean(losses[-50:]))],
            'known_draws':int((plan >= 0).sum()), 'unique_inputs':len(set(plan[plan >= 0].ravel().tolist())),
            'new_normal_draws':int((plan >= offset).sum()), 'actual_optimizer_updates':p['steps'],
            'reload_first64_exact':True, 'base_and_roles_unchanged':True,
            'seconds_training_and_common_scoring':time.monotonic()-tick}
        subgroup = retained_subgroups(ps, av, y, features, labels, result)
        (out/(name+'.subgroups.json')).write_text(json.dumps(subgroup, ensure_ascii=False, indent=2)+'\n')
        all_ps[name] = ps
        print(name, json.dumps(output['arms'][name], ensure_ascii=False), flush=True)
    candidate = output['arms']['new_normals']['TRAIN_proxy']
    control = output['arms']['old_resampled']['TRAIN_proxy']
    output['proceed_CAL'] = (changed > 0 and candidate['novel_TP'] > max(output['parent']['novel_TP'], control['novel_TP'])
        and candidate['wrong_actual'] <= min(output['parent']['wrong_actual'], control['wrong_actual'])
        and all(candidate[d]['novel_TP'] >= output['parent'][d]['novel_TP'] for d in ('S_to_SS','SS_to_S')))
    policies = {'parent':output['parent'], **{k:v['TRAIN_proxy'] for k,v in output['arms'].items()}}
    changes = {name:collections.Counter() for name in output['arms']}
    with (out/'paired_proposal_ledger.jsonl').open('x') as stream:
        for label in labels:
            calls = {name:proposed(label, ps, av, features, policies[name]) for name,ps in all_ps.items()}
            i = label['feature_index']
            exact_calls = {name:bool(call and i is not None
                and label['expected'] == features[i]['canonical']
                and label['actual'] == ('ㅆ' if features[i]['canonical'] == 'ㅅ' else 'ㅅ')
                and label['label_kind'] == 'pair_error') for name,call in calls.items()}
            for name, counts in changes.items():
                counts['new_calls'] += calls[name] and not calls['parent']
                counts['lost_calls'] += calls['parent'] and not calls[name]
                prefix = 'normal' if label['label_kind'] == 'correct' else 'error'
                counts[prefix+'_new_calls'] += calls[name] and not calls['parent']
                counts[prefix+'_lost_calls'] += calls['parent'] and not calls[name]
                counts['new_exact_pair_calls'] += exact_calls[name] and not exact_calls['parent']
                counts['lost_exact_pair_calls'] += exact_calls['parent'] and not exact_calls[name]
            service.write_line(stream, dict(label, TRAIN_y=None if label['feature_index'] is None else int(y[label['feature_index']]),
                proposals=calls, exact_pair_proposals=exact_calls))
    output['paired_call_changes_all_sites'] = {k:dict(v) for k,v in changes.items()}
    output['sha256'] = {f.name:service.sha(f) for f in out.glob('*') if f.suffix in ('.pt','.npz','.npy','.jsonl')}
    (out/'report.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps({k:v for k,v in output.items() if k not in ('sha256','arms')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

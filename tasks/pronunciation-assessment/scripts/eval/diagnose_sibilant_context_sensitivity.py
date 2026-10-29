"""TRAIN-only protected-C/V context intervention with a normalization control."""
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
import hashlib
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_context_diagnostic import (attenuate_outside, capture_native_context,
    forward_context_variant)
from mdd.sibilant_hubert_adaptation import load_native, tensor_digest
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_transition import VOWELS
from mdd.sibilant_vowel_presence import observed_vowel_support


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def selection(p, features, labels, rows, annotations, y):
    candidates, states, geometries = collections.defaultdict(list), collections.Counter(), {}
    for lab in labels:
        i = lab['feature_index']
        if i is None:
            states['no_input'] += 1
            continue
        if y[i] not in (0, 1) or lab['label_kind'] not in ('correct', 'pair_error'):
            states['not_supported_S_SS'] += 1
            continue
        f, uid = features[i], lab['uid']
        if f['canonical'] != lab['expected'] or lab['a'] is None:
            states['canonical_or_actual_mapping_unknown'] += 1
            continue
        if uid not in geometries:
            geometries[uid] = single_phone_intervals(annotations[uid]['tag_phones'], rows[uid]['actual'])
        intervals, a = geometries[uid], lab['a']
        if intervals is None or a+1 >= len(intervals) or intervals[a] is None or intervals[a+1] is None:
            states['grouped_or_unknown_TAG_bounds'] += 1
            continue
        c, v = intervals[a:a+2]
        if c[2] != lab['actual'] or v[2] not in VOWELS or not c[0] < c[1] <= v[0] < v[1]:
            states['invalid_CV_bounds'] += 1
            continue
        # A whole coarse C and first <=120ms V must fit, including a 25ms
        # sample guard. This is stronger than the old half-overlap label rule.
        lo, hi = int(np.floor(c[0]*16000)), int(np.ceil(min(v[1], v[0]+.12)*16000))
        guard = 400
        start = f['left']//320*320
        if lo-guard < f['left'] or hi+guard > f['right']:
            states['full_CV_and_guard_not_in_ROI'] += 1
            continue
        fraction = 1.-(hi-lo+2*guard)/(f['right']-start)
        first, last = (f['left']+319)//320, (f['right']-400)//320
        global_frames = np.arange(first, last+1)
        core = np.flatnonzero((global_frames*320 >= lo) & (global_frames*320+400 <= hi))
        if fraction < p['minimum_outside_fraction'] or len(core) < p['minimum_core_frames']:
            states['insufficient_outside_or_core_frames'] += 1
            continue
        vowel = f['observed_V'] if f['observed_V'] in ('ㅣ', 'ㅡ', 'ㅜ') else 'other'
        stratum = (f['canonical'], lab['actual'], vowel)
        identity = uid+'/'+str(f['q_runtime'])
        digest = hashlib.sha256(identity.encode()).hexdigest()
        candidates[stratum].append((digest, {'feature_index': i, 'uid': uid,
            'q_runtime': f['q_runtime'], 'q': lab['q'], 'a': a,
            'label_kind': lab['label_kind'], 'expected': lab['expected'], 'actual': lab['actual'],
            'observed_V': f['observed_V'], 'C_interval': c, 'V_interval': v,
            'core_source_samples': [lo, hi], 'protected_source_samples': [lo-guard, hi+guard],
            'core_feature_indices': core.tolist(), 'outside_fraction': fraction,
            'selection_hash': digest}))
        states['eligible'] += 1
    selected = []
    for stratum, choices in sorted(candidates.items()):
        selected.extend(z for _, z in sorted(choices)[:p['per_stratum']])
    for z in selected:
        row = rows[z['uid']]
        opts = service.all_optimal_mapping(row['canonical'], row['actual'])[z['q']+1]
        if opts != {z['a']+1}:
            raise ValueError('selected intended V does not map uniquely adjacent to C')
    return selected, {'all_retained_sites': len(labels), 'states': dict(states),
        'eligible_strata': {str(k): len(v) for k, v in sorted(candidates.items())},
        'selected': len(selected), 'selection_scope': 'TRAIN only, max8 per expected/actual/observed-vowel stratum; no score-based selection.'}


def local_hidden(hidden, local):
    return torch.cat([hidden[6], hidden[12]], -1)[0, local].to(torch.float16).float()


def head_scores(head, hidden, relative, subset=None):
    if subset is not None:
        hidden, relative = hidden[subset], relative[subset]
    n = len(hidden)
    x = torch.nn.functional.pad(hidden, (0, 0, 0, 32-n))[None]
    rel = torch.nn.functional.pad(relative, (0, 32-n))[None]
    mask = torch.arange(32, device=x.device)[None] < n
    logits, role = head(x, mask, rel)
    return {'probabilities': logits.softmax(-1)[0].cpu().tolist(),
            'pair_margin': float((logits[0, 0]-logits[0, 1]).cpu()),
            'available': bool(observed_vowel_support(role, mask)[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen context diagnostic input changed: '+path)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('context diagnostic must use TRAIN only')
    features, labels = service.read_jsonl(p['features']), service.read_jsonl(cache/'data.labels.jsonl')
    rows = {z['row']['uid']: z['row'] for z in service.read_jsonl(cache/'data.responses.jsonl')}
    annotations = {z['stem']: z for z in service.read_jsonl(p['annotations'])}
    y = np.load(p['targets'])['y']
    selected, inventory = selection(p, features, labels, rows, annotations, y)
    if not selected:
        raise ValueError('no diagnostic inputs meet frozen eligibility')
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    write(out/'selection.json', {'inventory': inventory, 'cases': selected})
    device = torch.device(p.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(2)
    encoder, processor, _ = load_native(p['model_manifest'], device)
    before = tensor_digest(encoder.state_dict())
    heads = {name: load_probe(torch.load(path, map_location='cpu'), p['binding'], device)
             for name, path in p['heads'].items()}
    stored = np.load(p['data'])
    reference_x, relative = stored['x'], stored['relative_vowel']
    cases, max_identity, max_conv, exact_original = [], 0., 0., 0
    tick = time.monotonic()
    with torch.no_grad():
        for j, chosen in enumerate(selected):
            i, uid = chosen['feature_index'], chosen['uid']
            f = features[i]
            source = service.load_audio(rows[uid])[:320000]
            if hashlib.sha256(source[f['left']:f['right']].tobytes()).hexdigest() != f['clip_sha256']:
                raise ValueError('original source PCM differs')
            start, right = f['left']//320*320, f['right']
            wave = source[start:right]
            lo, hi = [s-start for s in chosen['protected_source_samples']]
            local = torch.arange((f['left']+319)//320, (right-400)//320+1, device=device)-start//320
            core = torch.as_tensor(chosen['core_feature_indices'], device=device)
            rel = torch.as_tensor(relative[i, :len(local)], device=device)
            value = processor(wave, sampling_rate=16000, return_tensors='pt', padding=False).input_values.to(device)
            original, normalization, conv = capture_native_context(encoder, value)
            hidden = local_hidden(original, local)
            if not torch.equal(hidden, torch.as_tensor(reference_x[i, :len(local)], device=device).float()):
                raise ValueError('original diagnostic features differ from frozen native cache')
            exact_original += 1
            identity, identity_conv = forward_context_variant(encoder, value, normalization)
            identity_error = float((local_hidden(identity, local)-hidden).abs().max())
            if identity_error != 0 or not torch.equal(identity_conv, conv):
                raise ValueError('anchored original normalization identity failed')
            max_identity = max(max_identity, identity_error)
            record = dict(chosen, original_clip_sha256=f['clip_sha256'], variants={})
            for name, head in heads.items():
                record['variants']['original/'+name] = {
                    'full': head_scores(head, hidden, rel),
                    'core': head_scores(head, hidden, rel, core)}
            for gain in p['gains']:
                changed = attenuate_outside(wave, lo, hi, gain, p['fade_samples'])
                if not np.array_equal(changed[lo:hi], wave[lo:hi]):
                    raise ValueError('protected original samples changed')
                v = processor(changed, sampling_rate=16000, return_tensors='pt', padding=False).input_values.to(device)
                for held in (False, True):
                    variant = 'gain'+str(gain)+('/held_stats' if held else '/native_stats')
                    hs, modified_conv = forward_context_variant(encoder, v, normalization if held else None)
                    h = local_hidden(hs, local)
                    conv_error = float((modified_conv[0, :, local[core]]-conv[0, :, local[core]]).abs().max())
                    if held:
                        max_conv = max(max_conv, conv_error)
                        if conv_error > p['protected_conv_tolerance']:
                            raise ValueError('protected target convolution changed with held statistics')
                    metadata = {'modified_encoder_wave_sha256': hashlib.sha256(changed.tobytes()).hexdigest(),
                        'protected_PCM_bitwise_equal': True, 'protected_conv_max_abs_error': conv_error,
                        'protected_h6_mean_abs_change': float((hs[6][0, local[core]]-original[6][0, local[core]]).abs().mean()),
                        'protected_h12_mean_abs_change': float((hs[12][0, local[core]]-original[12][0, local[core]]).abs().mean())}
                    for name, head in heads.items():
                        record['variants'][variant+'/'+name] = dict(metadata,
                            full=head_scores(head, h, rel), core=head_scores(head, h, rel, core))
            cases.append(record)
            if (j+1) % 16 == 0:
                print(json.dumps({'done': j+1, 'total': len(selected), 'seconds': time.monotonic()-tick}), flush=True)
    if tensor_digest(encoder.state_dict()) != before:
        raise ValueError('diagnostic changed encoder weights')
    groups = {}
    for name in heads:
        for gain in p['gains']:
            for held in (False, True):
                variant = 'gain'+str(gain)+('/held_stats' if held else '/native_stats')+'/'+name
                for pool in ('full', 'core'):
                    changes, flips, all_winner_flips = [], 0, 0
                    for case in cases:
                        a, b = case['variants']['original/'+name][pool], case['variants'][variant][pool]
                        changes.append(abs(b['pair_margin']-a['pair_margin']))
                        flips += (a['pair_margin'] >= 0) != (b['pair_margin'] >= 0)
                        all_winner_flips += np.argmax(a['probabilities']) != np.argmax(b['probabilities'])
                    groups[variant+'/'+pool] = {'cases': len(cases), 'pair_sign_flips': int(flips),
                        'pair_sign_flip_rate': flips/len(cases), 'class5_winner_flips': int(all_winner_flips),
                        'absolute_pair_margin_change_median_p90': np.percentile(changes, [50, 90]).tolist()}
    primary = groups['gain0.1/held_stats/original/core']
    report = {'scope': p['scope'], 'inventory': inventory, 'groups': groups,
        'all_original_features_bitwise_equal': exact_original, 'held_stats_identity_max_abs_error': max_identity,
        'held_stats_protected_conv_max_abs_error': max_conv, 'encoder_weights_unchanged': True,
        'context_augmentation_warranted_for_TRAIN_trial': primary['pair_sign_flip_rate'] >= p['minimum_context_flip_rate'],
        'decision_scope': 'Sensitivity gate only, not proof that context use is harmful or lexical, and not modified-audio accuracy or service gain. TRAIN-only invariance trial must still beat original on real unmodified data.',
        'seconds': time.monotonic()-tick, 'deployment': False, 'goal_complete': False}
    write(out/'cases.json', cases)
    write(out/'report.json', report)
    write(out/'finished.json', {'report_sha256': service.sha(out/'report.json')})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

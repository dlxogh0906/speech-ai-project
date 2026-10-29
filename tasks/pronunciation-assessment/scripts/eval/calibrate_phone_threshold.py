"""Freeze one stricter phone-display threshold on fresh calibration recordings.

No training, no service writes. TSV/LTR/CANON only; waveforms stay in memory.
Calibration and test use disjoint recordings; test is opened after selection.
"""
import sys as _sys
import pathlib as _pl
_root = _pl.Path(__file__).resolve()
while not (_root / 'pyproject.toml').exists():
    if _root.parent == _root:
        raise RuntimeError('Repository root not found')
    _root = _root.parent
_sys.path[:0] = [str(_root), *(str(p) for p in sorted(_root.glob('tasks/*/src'))),
                str(_pl.Path(__file__).resolve().parents[1] / 'train')]

import argparse
import collections
import gc
import json
import random
import time

import numpy as np

import compare_noise_mixing as mixing
import compare_noise_preprocess as previous
import eval_noise_training as evaluation
from summarize_noise_preprocess import paired_deltas
from train_noise_pilot import digest, selected_data, noise_paths, write_json
from shared.paths import MDD_OUTPUTS

CONDITIONS = ['clean', 'babble20', 'babble10', 'babble5', 'rumble0', 'reverb_babble5']
PRIMARY = ['babble10', 'babble5']
COUNT_PAIRS = [('api_recall', 'api_tp', 'n_err'), ('api_frr', 'api_fp', 'n_ok'),
               ('preservation', 'preserved', 'n_err'),
               ('display_preservation', 'api_preserved', 'n_err'),
               ('raw_recall', 'raw_tp', 'n_err'), ('jer', 'jer_num', 'jer_den')]


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('Refusing to overwrite a frozen experiment decision')
    else:
        write_json(path, value)


def prepare(args):
    from mdd.speechsuper import FLAG_MAX
    out = _pl.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise RuntimeError('Protocol already exists; use later actions')
    prior_out = _pl.Path(args.previous_out)
    old = json.loads((prior_out / 'plan.json').read_text())
    chosen = json.loads((prior_out / 'selection.json').read_text())['chosen']['step']
    target = prior_out / 'checkpoints/targeted' / ('step_%d' % chosen)
    data = {s: mixing.load_split(old['manifest'], s) for s in ('valid', 'test')}
    train_items, _ = selected_data(_pl.Path(old['train_manifest']), 'train', old['train_indices'])
    excluded = {str(p) for p, _ in train_items}
    exclusion_sources = []
    # Exclude old preprocessing/mixing/robustness and simple-training selections.
    for name in old['prior_evaluation_exclusions']:
        p = _pl.Path(name)
        saved = json.loads(p.read_text())
        split = saved.get('split') or saved.get('args', {}).get('split', 'test')
        source = data[split]
        excluded.update(str(source[0] / source[1][i][0]) for i in saved['source_row_indices'])
        exclusion_sources.append({'file': str(p), 'sha256': digest(p),
                                  'split': split, 'rows': len(saved['source_row_indices'])})
    # Also exclude every consistency preparation/smoke/pilot selection, even
    # reserved rows never evaluated. No previous prediction values are read.
    old_plans = sorted(prior_out.parent.glob('*/plan.json'))
    additional = MDD_OUTPUTS / 'analysis/noise_training_2026-10-09/pilot/plan.json'
    if additional.exists():
        old_plans.append(additional)
    for p in old_plans:
        saved = json.loads(p.read_text())
        for split in ('valid', 'test'):
            source = data[split]
            excluded.update(str(source[0] / source[1][i][0])
                            for i, _ in saved.get(split + '_selection', []))
        exclusion_sources.append({'file': str(p), 'sha256': digest(p), 'split': 'valid/test',
                                  'rows': sum(len(saved.get(s + '_selection', []))
                                              for s in ('valid', 'test'))})
    vsel, vstats, vpaths = mixing.choose_samples(data['valid'], args.calibration_per_cohort,
        args.calibration_per_cohort, args.seed + 1, excluded)
    tsel, tstats, tpaths = mixing.choose_samples(data['test'], args.test_per_cohort,
        args.test_per_cohort, args.seed + 2, excluded | vpaths)
    target_paths = vpaths | tpaths
    # Audit against the original model's complete training split without meta.
    training_overlap = 0
    with (_pl.Path(old['manifest']) / 'train.tsv').open() as stream:
        root = _pl.Path(next(stream).rstrip('\n'))
        for line in stream:
            if str(root / line.rstrip('\n').split('\t')[0]) in target_paths:
                training_overlap += 1
    if training_overlap or vpaths & tpaths or target_paths & excluded:
        raise RuntimeError('Non-disjoint evaluation recordings')
    pools = {s: noise_paths(old, s) for s in ('train', 'valid', 'test')}
    sets = {s: set(map(str, pool)) for s, pool in pools.items()}
    if (sets['valid'] & sets['test'] or sets['train'] & (sets['valid'] | sets['test'])
            or target_paths & set.union(*sets.values())):
        raise RuntimeError('Target/noise partition overlap')
    input_files = [_pl.Path(old['manifest']) / (s + ext) for s in ('valid', 'test')
                   for ext in ('.tsv', '.ltr', '.canon')]
    input_files += [_pl.Path(old['manifest']) / 'train.tsv',
                    _pl.Path(old['noise_manifest']) / 'valid.tsv']
    assets = [_pl.Path(old[k]) for k in ('norms', 'calib')]
    for folder in (_pl.Path(old['ckpt']), target):
        assets += [folder / name for name in
                   ('config.json', 'preprocessor_config.json', 'jamo_vocab.json', 'model.safetensors')]
    production = [_root / 'deploy/pronunciation-demo/app.py',
                  _root / 'tasks/pronunciation-assessment/src/mdd/speechsuper.py']
    sources = [_pl.Path(__file__).resolve(), _pl.Path(mixing.__file__), _pl.Path(previous.__file__),
               _pl.Path(evaluation.__file__),
               _root / 'tasks/pronunciation-assessment/scripts/eval/run_phone_threshold.sh',
               _root / 'tasks/pronunciation-assessment/scripts/eval/summarize_noise_preprocess.py',
               _root / 'tasks/pronunciation-assessment/scripts/train/train_noise_pilot.py',
               _root / 'tasks/pronunciation-assessment/src/mdd/noise_augment.py',
               _root / 'tasks/pronunciation-assessment/src/mdd/gop_feats_v2.py',
               _root / 'tasks/pronunciation-assessment/src/mdd/session.py',
               _root / 'tasks/pronunciation-assessment/src/mdd/vad.py']
    plan = {'experiment': 'frozen_targeted_model_scalar_error_display_threshold',
        'seed': args.seed, 'manifest': old['manifest'], 'noise_manifest': old['noise_manifest'],
        'baseline_ckpt': old['ckpt'], 'targeted_ckpt': str(target), 'fixed_step': chosen,
        'norms': old['norms'], 'calib': old['calib'], 'default_threshold': FLAG_MAX,
        'threshold_grid': list(range(FLAG_MAX + 1)),
        'valid_selection': vsel, 'test_selection': tsel,
        'valid_selection_stats': vstats, 'test_selection_stats': tstats,
        'noise_indices': {s: old['noise_indices'][s] for s in ('valid', 'test')},
        'eval_conditions': CONDITIONS, 'primary_conditions': PRIMARY,
        'calibration_policy': 'One integer threshold <= default; no per-condition tuning. '
             'Highest threshold whose false flags do not exceed baseline in EACH of all six '
             'conditions. Missing-score/deletion flags remain unchanged. If none qualifies, '
             'freeze threshold=0 for diagnosis and reject calibration feasibility.',
        'test_policy': {'retune': False, 'point_frr_increase_max_pp': 0.0,
            'per_condition_ci95_frr_upper_max_pp': 0.3,
            'primary_recall_ci95_lower_min_pp': 0.0,
            'primary_display_preservation_ci95_lower_min_pp': 0.0,
            'clean_recall_ci95_lower_min_pp': -2.0,
            'adoption_scope': 'synthetic display candidate only; no deployment'},
        'bootstraps': args.bootstraps,
        'noise_snr_convention': 'whole-utterance RMS; in-memory synthetic noise',
        'eligibility': 'clean-audio low_speech pass, 1.5..10s, canonical error sites; '
            'normal/error balanced. ALL selected noisy inputs scored including gate rejects.',
        'exclusion_sources': exclusion_sources,
        'input_hashes': {str(p): digest(p) for p in input_files},
        'asset_hashes': {str(p): digest(p) for p in assets},
        'production_hashes': {str(p): digest(p) for p in production},
        'source_hashes': {str(p.relative_to(_root)): digest(p) for p in sources}}
    write_json(out / 'plan.json', plan)
    snapshot = out / 'source_snapshots'
    snapshot.mkdir()
    for p in sources:
        (snapshot / p.name).write_bytes(p.read_bytes())
    write_json(out / 'audit.json', {'calibration_recordings': len(vsel), 'test_recordings': len(tsel),
        'excluded_target_paths': len(excluded), 'prior_evaluation_overlap': 0,
        'original_training_target_overlap': training_overlap, 'calibration_test_target_overlap': 0,
        'training_noise_overlap': 0, 'calibration_test_noise_overlap': 0,
        'target_noise_overlap': 0, 'speaker_overlap': 'not audited; no speaker metadata read',
        'new_recordings_of_trained_sentence_items': True, 'audio_written': False,
        'production_changed': False, 'training': False})
    print('[prepared] calibration=%d test=%d step=%d' % (len(vsel), len(tsel), chosen), flush=True)


def scored_sites(decoder, probabilities, canon, tag, sec, formatter):
    from mdd.gop_feats_v2 import utt_features
    from mdd.jamo_to_kog2p import VOWEL
    from mdd.session import phone_context
    targets = [decoder.d.index(t) for t in canon]
    if decoder.d.unk() in targets:
        raise RuntimeError('Canonical phone outside vocabulary')
    _, extra = utt_features(decoder, probabilities, targets, canon, sec)
    pred = [t for t in extra['pred'] if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
    row = previous.phone_diagnostics(canon, tag, pred, extra['gops_m'], formatter,
                                     set(VOWEL), phone_context)
    reference, _ = previous.axis(canon, tag)
    hypothesis, _ = previous.axis(canon, pred)
    hist = {k: [0] * 101 for k in ('tp', 'fp', 'preserved')}
    always = {k: 0 for k in hist}
    deletion = {'tp': 0, 'fp': 0}
    for i, (c, h, t) in enumerate(zip(canon, hypothesis, reference)):
        if h == c:
            continue
        truth = t != c
        key = 'tp' if truth else 'fp'
        margin = extra['gops_m'][i] if i < len(extra['gops_m']) else None
        score = formatter._score(phone_context(canon, i, set(VOWEL)), margin) if h is not None else None
        is_preserved = truth and h == t
        if score is None:
            always[key] += 1
            always['preserved'] += int(is_preserved)
            deletion[key] += int(h is None)
        else:
            if not isinstance(score, int) or not 0 <= score <= 100:
                raise RuntimeError('Unexpected phone-score scale')
            hist[key][score] += 1
            hist['preserved'][score] += int(is_preserved)
    row.update(flag_score_hist=hist, always_flag=always, deletion_flag=deletion)
    from mdd.speechsuper import FLAG_MAX
    default = apply_threshold(row, FLAG_MAX)
    if (default['api_tp'], default['api_fp']) != (row['api_tp'], row['api_fp']):
        raise RuntimeError('Threshold histogram does not reproduce production flags')
    row['prediction_sha256'] = __import__('hashlib').sha256(''.join(pred).encode()).hexdigest()
    return row


def apply_threshold(row, threshold, method=None):
    result = {k: v for k, v in row.items() if k not in ('flag_score_hist',)}
    for name, count in [('tp', 'api_tp'), ('fp', 'api_fp'), ('preserved', 'api_preserved')]:
        result[count] = row['always_flag'][name] + sum(row['flag_score_hist'][name][:threshold])
    result['api_flagged_utterance'] = bool(result['api_tp'] + result['api_fp'])
    result['threshold'] = threshold
    if method:
        result['method'] = method
    return result


def check_hashes(plan):
    for kind in ('input_hashes', 'asset_hashes', 'production_hashes', 'source_hashes'):
        for name, expected in plan[kind].items():
            p = _pl.Path(name)
            if not p.is_absolute():
                p = _root / p
            if digest(p) != expected:
                raise RuntimeError('Frozen input/asset/source changed: ' + kind)


def evaluate(out, plan, split, method, limit=0):
    import torch
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    from mdd.speechsuper import SpeechSuperFormatter
    from mdd.vad import trim
    ckpt = plan[method + '_ckpt']
    folder = out / ('smoke' if limit else split)
    folder.mkdir(exist_ok=True)
    destination = folder / (method + '.jsonl')
    complete = folder / (method + '.complete.json')
    if complete.exists():
        info = json.loads(complete.read_text())
        if info['plan_sha256'] != digest(out / 'plan.json') or info['ckpt'] != ckpt:
            raise RuntimeError('Stale evaluation output')
        return
    if destination.exists():
        raise RuntimeError('Partial result exists; retain it for diagnosis')
    check_hashes(plan)
    root, rows, tags, canons = mixing.load_split(plan['manifest'], split)
    selected = plan[split + '_selection'][:limit] if limit else plan[split + '_selection']
    pool = noise_paths(plan, split)
    seed = plan['seed'] + (1 if split == 'valid' else 2)
    decoder = HFDecoder(ckpt)
    formatter = SpeechSuperFormatter(plan['norms'], calib_path=plan['calib'])
    if formatter.cal is None:
        raise RuntimeError('Fixed phone calibration missing')
    forward_lprobs(decoder, [np.zeros(2 * previous.SR, np.float32)])
    count = 0
    started = time.monotonic()
    with destination.open('w', encoding='utf-8') as stream:
        for k, (i, cohort) in enumerate(selected):
            wav = previous.read_wav(root / rows[i][0])
            original_sha = __import__('hashlib').sha256(wav.tobytes()).hexdigest()
            for condition, signal in evaluation.conditions(wav, i, seed, pool):
                if condition not in CONDITIONS:
                    continue
                processed, vinfo = trim(signal, previous.SR)
                begin = time.perf_counter()
                lp, lengths = forward_lprobs(decoder, [processed])
                record = scored_sites(decoder, lp[0, :int(lengths[0])].numpy(),
                                      canons[i].split(), tags[i].split(), len(processed) / previous.SR,
                                      formatter)
                record.update(k=k, cohort=cohort, condition=condition, method=method,
                    source_audio_sha256=original_sha,
                    processed_audio_sha256=__import__('hashlib').sha256(processed.tobytes()).hexdigest(),
                    model_and_diagnosis_ms=(time.perf_counter() - begin) * 1000,
                    original_sec=len(wav) / previous.SR, processed_sec=len(processed) / previous.SR,
                    original_check_pass=previous.audio_check(wav)['pass'],
                    noisy_check_pass=previous.audio_check(signal)['pass'], cut_sec=vinfo.get('cut_sec', 0))
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
            stream.flush()
            if k == 0 or (k + 1) % 25 == 0 or k + 1 == len(selected):
                elapsed = time.monotonic() - started
                print('[eval %s %s] %d/%d %.1fs eta=%.1fs' %
                      (split, method, k + 1, len(selected), elapsed,
                       elapsed / (k + 1) * (len(selected) - k - 1)), flush=True)
    if count != len(selected) * len(CONDITIONS):
        raise RuntimeError('Incomplete evaluation matrix')
    write_json(complete, {'plan_sha256': digest(out / 'plan.json'), 'ckpt': ckpt,
        'records': count, 'recordings': len(selected), 'elapsed_sec': time.monotonic() - started,
        'results_sha256': digest(destination), 'batch_size': 1,
        'precision': 'HFDecoder GPU fp16 single-input; original calibration fixed'})
    del decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_results(out, split, plan):
    pairs = {}
    result = []
    for method in ('baseline', 'targeted'):
        info = json.loads((out / split / (method + '.complete.json')).read_text())
        path = out / split / (method + '.jsonl')
        if info['plan_sha256'] != digest(out / 'plan.json') or info['results_sha256'] != digest(path):
            raise RuntimeError('Changed evaluation results')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        expected = len(plan[split + '_selection']) * len(CONDITIONS)
        if len(rows) != expected or len({(r['k'], r['condition']) for r in rows}) != expected:
            raise RuntimeError('Incomplete or duplicate model matrix')
        for r in rows:
            identity = (r['k'], r['condition'])
            invariant = tuple(r[k] for k in ('n_err', 'n_ok', 'jer_den', 'cohort',
                                              'source_audio_sha256', 'processed_audio_sha256'))
            if pairs.setdefault(identity, invariant) != invariant:
                raise RuntimeError('Paired label/audio mismatch')
        result.extend(rows)
    return result


def summarize(rows):
    result = {'n': len(rows), 'error_sites': sum(r['n_err'] for r in rows),
              'correct_sites': sum(r['n_ok'] for r in rows)}
    for name, num, den in COUNT_PAIRS:
        total_den = sum(r[den] for r in rows)
        result[name] = sum(r[num] for r in rows) / max(1, total_den)
    normals = [r for r in rows if r['cohort'] == 'normal']
    result['normal_utterance_false_flag'] = float(np.mean([r['api_flagged_utterance'] for r in normals]))
    result['deletion_false_flag_floor'] = sum(r['deletion_flag']['fp'] for r in rows) / max(1, result['correct_sites'])
    result['missing_score_false_flag_floor'] = sum(r['always_flag']['fp'] for r in rows) / max(1, result['correct_sites'])
    return result


def calibrate(out, plan):
    rows = read_results(out, 'valid', plan)
    base = {c: summarize([apply_threshold(r, plan['default_threshold']) for r in rows
                          if r['method'] == 'baseline' and r['condition'] == c]) for c in CONDITIONS}
    curves = []
    for threshold in plan['threshold_grid']:
        vals = {c: summarize([apply_threshold(r, threshold) for r in rows
                             if r['method'] == 'targeted' and r['condition'] == c]) for c in CONDITIONS}
        feasible = all(vals[c]['api_frr'] <= base[c]['api_frr'] for c in CONDITIONS)
        curves.append({'threshold': threshold, 'feasible': feasible, 'metrics': vals})
    eligible = [v for v in curves if v['feasible']]
    chosen = eligible[-1] if eligible else curves[0]
    freeze_json(out / 'selection.json', {'threshold': chosen['threshold'],
        'calibration_feasible': bool(eligible), 'split': 'valid only', 'test_retuning': False,
        'plan_sha256': digest(out / 'plan.json'), 'baseline': base, 'chosen': chosen,
        'policy': plan['calibration_policy']})
    write_json(out / 'valid/threshold_curves.json', curves)
    print('[frozen threshold] %d feasible=%s' % (chosen['threshold'], bool(eligible)), flush=True)
    for c in CONDITIONS:
        v = chosen['metrics'][c]
        print('[calibration %s] recall %.2f -> %.2f false flags %.2f -> %.2f floor %.2f' %
              (c, 100*base[c]['api_recall'], 100*v['api_recall'],
               100*base[c]['api_frr'], 100*v['api_frr'], 100*v['missing_score_false_flag_floor']), flush=True)


def pool(rows, method, names):
    by_key = collections.defaultdict(list)
    for r in rows:
        if r['method'] == method and r['condition'] in names:
            by_key[r['k']].append(r)
    result = []
    for k in sorted(by_key):
        items = by_key[k]
        if {r['condition'] for r in items} != set(names):
            raise RuntimeError('Incomplete pooled conditions')
        rec = {'k': k, 'cohort': items[0]['cohort']}
        for field in set(f for _, n, d in COUNT_PAIRS for f in (n, d)):
            rec[field] = sum(r[field] for r in items)
        result.append(rec)
    return result


def deltas(base, alt, indices):
    result = paired_deltas(base, alt, indices)
    for name, num, den in COUNT_PAIRS:
        if name in result:
            continue
        b = {r['k']: r for r in base}
        a = {r['k']: r for r in alt}
        keys = sorted(b)
        if keys != sorted(a):
            raise RuntimeError('Bootstrap pair mismatch')
        n = np.array([a[k][num] - b[k][num] for k in keys], float)
        d = np.array([b[k][den] for k in keys], float)
        boot = n[indices].sum(1) / np.maximum(1, d[indices].sum(1))
        result[name] = {'delta_pp': float(100*n.sum()/max(1,d.sum())),
                       'ci95_pp': list(map(float, 100*np.percentile(boot, (2.5,97.5))))}
    return result


def report(out, plan):
    selection = json.loads((out / 'selection.json').read_text())
    frozen = selection['threshold']
    source = read_results(out, 'test', plan)
    rows = []
    for r in source:
        rows.append(apply_threshold(r, plan['default_threshold'],
                                    'baseline' if r['method'] == 'baseline' else 'targeted_default'))
        if r['method'] == 'targeted':
            rows.append(apply_threshold(r, frozen, 'targeted_calibrated'))
    methods = ['baseline', 'targeted_default', 'targeted_calibrated']
    values = {c: {m: summarize([r for r in rows if r['condition'] == c and r['method'] == m])
                  for m in methods} for c in CONDITIONS}
    ordered = sorted((r for r in rows if r['method'] == 'baseline' and r['condition'] == 'clean'),
                     key=lambda r:r['k'])
    cohorts = np.array([r['cohort'] for r in ordered])
    rng = np.random.RandomState(plan['seed'] + 37)
    indices = np.concatenate([rng.choice(np.where(cohorts == group)[0],
        size=(plan['bootstraps'], int((cohorts == group).sum())), replace=True)
        for group in ('normal', 'error')], axis=1)
    compared = {m: {c: deltas(pool(rows, 'baseline', [c]), pool(rows, m, [c]), indices)
                        for c in CONDITIONS} for m in methods[1:]}
    for m in methods[1:]:
        compared[m]['primary'] = deltas(pool(rows, 'baseline', PRIMARY), pool(rows, m, PRIMARY), indices)
    calibrated = compared['targeted_calibrated']
    checks = {'calibration_feasible': selection['calibration_feasible'],
        'each_condition_no_point_false_flag_increase': all(calibrated[c]['api_frr']['delta_pp'] <= 0
                                                          for c in CONDITIONS),
        'each_condition_frr_upper_ci_at_most_0_3pp': all(calibrated[c]['api_frr']['ci95_pp'][1] <= 0.3
                                                       for c in CONDITIONS),
        'primary_recall_gain_ci_positive': calibrated['primary']['api_recall']['ci95_pp'][0] > 0,
        'primary_display_preservation_gain_ci_positive':
            calibrated['primary']['display_preservation']['ci95_pp'][0] > 0,
        'clean_recall_ci_loss_at_most_2pp': calibrated['clean']['api_recall']['ci95_pp'][0] >= -2}
    accepted = all(checks.values())
    decision = {'threshold': frozen, 'checks': checks,
        'verdict': 'synthetic_display_candidate' if accepted else 'no_safe_uniform_threshold_candidate',
        'test_retuned': False, 'deployment': False, 'metrics': values, 'paired_vs_baseline': compared}
    write_json(out / 'decision.json', decision)
    names = {'baseline': '기존(60)', 'targeted_default': '개선 모델(60)',
             'targeted_calibrated': '개선 모델(%d)' % frozen}
    lines = ['고정 모델 오류 표시 문턱 보정 — 2026-10-10', '',
        '결론: ' + ('합성 소음에서 헛지적 제한과 검출 개선을 함께 충족한 후보.' if accepted else
                    '하나의 문턱 조정으로 사전 지정 안전 조건을 모두 충족하지 못함.'),
        '보정 %d개 / 새 시험 %d개; 모델 step_%d 고정; 문턱 60 → %d.' %
            (len(plan['valid_selection']), len(plan['test_selection']), plan['fixed_step'], frozen),
        '문턱은 보정 데이터에서만 선택. 이전 평가 녹음을 제외했고 새 시험으로 재조정하지 않았다.',
        '', '조건별 시험 결과 (음소 자리 비율; 정확히 보존=사람 라벨과 같은 잘못된 음소를 인식)',
        '| 조건 | 모델/문턱 | 표시 검출 | 정상 자리 헛지적 | 정확히 보존 | 정확한 오류 표시 | 자모 오류율 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for m in methods:
            v = values[c][m]
            lines.append('| %s | %s | %s |' % (c, names[m], ' | '.join('%.2f%%' % (100*v[k])
                for k in ('api_recall','api_frr','preservation','display_preservation','jer'))))
    lines += ['', '조정한 모델 − 기존 모델; 발화 단위 짝 bootstrap 3,000회 95% 구간']
    for c in ['primary'] + CONDITIONS:
        lines.append(c)
        for metric in ('api_recall','api_frr','display_preservation'):
            v = calibrated[c][metric]
            lines.append('  %s %+.2f%%p [%+.2f, %+.2f]' % (metric, v['delta_pp'], *v['ci95_pp']))
    lines += ['', '사전 지정 판정:']
    lines += ['- %s: %s' % (k, v) for k,v in checks.items()]
    lines += ['', '보정 단계 (조건별 기존 헛지적을 넘지 않는 가장 높은 단일 문턱)',
              '보정 가능: %s' % selection['calibration_feasible']]
    for c in CONDITIONS:
        v = selection['chosen']['metrics'][c]
        lines.append('- %s 헛지적 %.2f%% → %.2f%%, 점수 없음 헛지적 바닥 %.2f%%' %
                     (c,100*selection['baseline'][c]['api_frr'],100*v['api_frr'],
                      100*v['missing_score_false_flag_floor']))
    lines += ['', '범위와 한계',
        '- 추가 학습 없음. 원래 보정표를 고정하고 표시 기준만 조정했다. 실제 인식/보존율은 문턱으로 바뀌지 않는다.',
        '- 삭제/점수 없음 오류 표시는 유지한다. 문턱 0에서도 남는 헛지적 바닥을 별도 계산했다.',
        '- 이전 시험 800개 및 모든 이전 평가/스모크/보정 녹음을 제외했다. 기존 학습 문항의 새 녹음이다.',
        '- 깨끗한 입력에서 기본 말소리 검사를 통과한 1.5~10초 녹음, 정상/오발음 50:50 표본이다.',
        '- 선택한 입력은 소음 조건에서도 모두 채점했다. 재녹음 거절 후보를 성능 계산에서 빼지 않았다.',
        '- 합성 소음 결과. 실제 부스/앱 녹음은 사용하지 않았다. 화자 메타를 읽지 않아 화자 중복 미검사.',
        '- 신뢰구간은 화자 단위가 아닌 발화 단위. 조건별 다중 비교 보정은 하지 않았다.',
        '- 문턱은 0..60의 정수 하나뿐이다. 음소별 보정/삭제 억제/품질 거절로 결과를 우회하지 않았다.',
        '- 검출에는 엉뚱한 다른 음소로 짚은 사례도 포함하므로 정확한 오류 표시를 함께 보고했다.',
        '- 운영 API/모델/보정표 변경, 서버 재시작, 배포 없음.']
    (out / 'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    check_hashes(plan)
    write_json(out / 'verification.json', {'status': 'complete',
        'raw_evaluation_records': len(source) + len(plan['valid_selection'])*len(CONDITIONS)*2,
        'paired_audio_and_labels_identical': True, 'all_hashes_unchanged': True,
        'frozen_selection_sha256': digest(out / 'selection.json'), 'production_changed': False,
        'production_restarted': False, 'test_retuned': False})
    print('[decision] %s threshold=%d' % (decision['verdict'],frozen), flush=True)
    print(json.dumps(calibrated['primary'], ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'smoke', 'calibrate', 'test', 'report'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--previous-out', default=str(MDD_OUTPUTS/'analysis/phone_consistency_2026-10-10/pilot'))
    ap.add_argument('--seed', type=int, default=20261030)
    ap.add_argument('--calibration-per-cohort', type=int, default=300)
    ap.add_argument('--test-per-cohort', type=int, default=500)
    ap.add_argument('--bootstraps', type=int, default=3000)
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare(args)
        return
    out = _pl.Path(args.out)
    plan = json.loads((out / 'plan.json').read_text())
    if args.action == 'smoke':
        for method in ('baseline','targeted'):
            evaluate(out,plan,'valid',method,limit=2)
        paired = {}
        for method in ('baseline', 'targeted'):
            path = out/'smoke'/(method+'.jsonl')
            for line in path.read_text().splitlines():
                row = json.loads(line)
                identity = (row['k'], row['condition'])
                invariant = tuple(row[k] for k in ('source_audio_sha256', 'processed_audio_sha256', 'n_err', 'n_ok'))
                if paired.setdefault(identity, invariant) != invariant:
                    raise RuntimeError('Smoke audio/label pair mismatch')
                tp = fp = -1
                for threshold in plan['threshold_grid']:
                    transformed = apply_threshold(row, threshold)
                    if transformed['api_tp'] < tp or transformed['api_fp'] < fp:
                        raise RuntimeError('Non-monotone threshold curve')
                    tp, fp = transformed['api_tp'], transformed['api_fp']
                floor = apply_threshold(row, 0)
                if floor['api_fp'] != row['always_flag']['fp']:
                    raise RuntimeError('Deletion/missing-score floor changed')
        write_json(out/'smoke/checks.json', {'production_flag_histogram_match': True,
            'thresholds_0_to_60_respect_unchanged_deletion_semantics': True,
            'test_data_not_opened': True})
    elif args.action == 'calibrate':
        for method in ('baseline','targeted'):
            evaluate(out,plan,'valid',method)
        calibrate(out,plan)
    elif args.action == 'test':
        selected = out/'selection.json'
        selection_hash = digest(selected)
        for method in ('baseline','targeted'):
            evaluate(out,plan,'test',method)
        if digest(selected) != selection_hash:
            raise RuntimeError('Selection changed during test')
        report(out,plan)
    else:
        report(out,plan)


if __name__ == '__main__':
    main()

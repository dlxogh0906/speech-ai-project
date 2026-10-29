"""TRAIN-only reliability/coverage check for existing harmonic measurement code.

All measurement uses the original runtime ROI. TAG V bounds are read only after
measurement to audit support/early coverage; they are not estimator inputs or
physical resonance/voicing-onset truth. No new pronunciation labels are made.
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
import hashlib
import importlib.util
import json
import math
import time
import numpy as np
import parselmouth
import eval_sibilant_directional as service
from inspect_sibilant_evidence import praat_tracks
from mdd.sz_detector import yin


def auc(positive, negative):
    if not positive or not negative:
        return None
    a, b = np.asarray(positive), np.asarray(negative)
    delta = a[:, None]-b[None, :]
    return float(((delta > 0).sum()+.5*(delta == 0).sum())/delta.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen harmonic audit source changed: '+path)
    if parselmouth.PRAAT_VERSION != '6.1.38':
        raise ValueError('audited Praat6.1.38 required')
    cache = Path(p['cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('harmonic preflight must use TRAIN only')
    selected = json.loads(Path(p['selection']).read_text())['cases']
    features = service.read_jsonl(p['features'])
    wanted = {z['uid'] for z in selected}
    rows = {}
    for line in (cache/'data.responses.jsonl').open():
        z = json.loads(line)
        if z['row']['uid'] in wanted:
            rows[z['row']['uid']] = z['row']
    spec = importlib.util.spec_from_file_location('pinned_harmonic_correction', p['correction'])
    correction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(correction)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(args.protocol).read_text())
    cases, frames, reasons = [], [], collections.Counter()
    tick = time.monotonic()
    for j, case in enumerate(selected):
        f = features[case['feature_index']]
        if (f['uid'], f['q_runtime']) != (case['uid'], case['q_runtime']):
            raise ValueError('original selection/runtime identity differs')
        source = service.load_audio(rows[f['uid']])[:320000]
        wave = source[f['left']:f['right']]
        before = hashlib.sha256(wave.tobytes()).hexdigest()
        if before != f['clip_sha256']:
            raise ValueError('harmonic original PCM differs')
        # Estimation sees every supported point of the runtime clip, no TAG.
        measured = praat_tracks(wave, 16000, correction.correction_iseli_i)
        offset = f['left']/16000
        vlo, vhi = case['V_interval'][:2]
        prefix_end = min(vhi, vlo+.12)
        early_end = min(prefix_end, vlo+.04)
        raw_early, corrected_early, raw_late, corrected_late = [], [], [], []
        counts = collections.Counter()
        for row in measured:
            t, f0, strength = row['time'], row['pitch_hz'], row['pitch_strength']
            center = int(round(t*16000))
            estimate = yin(wave[center-320:center+320], fmin=70., fmax=500.) if 320 <= center <= len(wave)-320 else None
            agreement = (f0 is not None and estimate is not None
                and abs(math.log2(estimate/f0)) <= p['F0_agreement_octaves']
                and np.isfinite(strength) and strength >= p['minimum_CC_strength'])
            if not f0:
                state = 'no_CC_candidate'
            elif estimate is None:
                state = 'no_local_YIN_candidate'
            elif not agreement:
                state = 'candidate_disagreement_or_weak_CC'
            else:
                state = 'two_estimators_agree_proxy_only'
            counts[state] += 1
            ceilings = row['ceilings']
            harmonic = (ceilings.get('5500') or {}).get('harmonics')
            raw_value = harmonic.get('h1h2_raw') if harmonic else None
            raw_valid = agreement and raw_value is not None and np.isfinite(raw_value)
            values = [(ceilings.get(str(c)) or {}).get('harmonics') for c in (5000, 5500, 6000)]
            corrected = [z.get('h1h2_corrected') if z else None for z in values]
            corrected_valid = raw_valid and all(v is not None and np.isfinite(v) for v in corrected)
            spread = float(np.ptp(corrected)) if corrected_valid else None
            corrected_valid = corrected_valid and spread <= p['maximum_correction_spread_db']
            corrected_valid = corrected_valid and t >= .025 and t+.025 <= len(wave)/16000
            raw_support = ([offset+min(t-.020, t-(1/70+2/16000), harmonic['window_start']),
                            offset+max(t+.020, t+(1/70+2/16000), harmonic['window_end'])] if harmonic else None)
            corrected_support = ([min(raw_support[0], offset+t-.025),
                                  max(raw_support[1], offset+t+.025)] if raw_support else None)
            source_t = offset+t
            early = vlo <= source_t <= early_end
            raw_in_V = raw_support is not None and raw_support[0] >= vlo and raw_support[1] <= prefix_end
            corrected_in_V = corrected_support is not None and corrected_support[0] >= vlo and corrected_support[1] <= prefix_end
            if raw_valid and raw_in_V:
                (raw_early if early else raw_late).append(float(raw_value))
            if corrected_valid and corrected_in_V:
                (corrected_early if early else corrected_late).append(float(corrected[1]))
            frames.append({'feature_index': case['feature_index'], 'source_center_s': source_t,
                'state': state, 'CC_hz': f0, 'local_YIN_hz': estimate,
                'CC_strength': float(strength) if np.isfinite(strength) else None,
                'H1_H2_raw': float(raw_value) if raw_value is not None and np.isfinite(raw_value) else None,
                'H1_H2_corrected_5000_5500_6000': corrected,
                'correction_spread_db': spread, 'raw_quality_proxy': bool(raw_valid),
                'corrected_quality_proxy': bool(corrected_valid), 'raw_support_s': raw_support,
                'corrected_support_s': corrected_support, 'raw_supported_in_TAG_Vprefix': raw_in_V,
                'corrected_supported_in_TAG_Vprefix': corrected_in_V,
                'in_first40ms_of_TAG_V': early})
        if hashlib.sha256(wave.tobytes()).hexdigest() != before:
            raise ValueError('harmonic estimator modified original PCM')
        reasons.update(counts)
        result = dict(case, actual_TAG_V=case['V_interval'][2], frame_states=dict(counts),
            raw_early_frames=len(raw_early), corrected_early_frames=len(corrected_early),
            raw_late_frames=len(raw_late), corrected_late_frames=len(corrected_late),
            raw_early_median=float(np.median(raw_early)) if len(raw_early) >= p['minimum_early_frames'] else None,
            corrected_early_median=float(np.median(corrected_early)) if len(corrected_early) >= p['minimum_early_frames'] else None)
        cases.append(result)
        if (j+1) % 16 == 0:
            print(json.dumps({'cases': j+1, 'total': len(selected), 'seconds': time.monotonic()-tick}), flush=True)
    grouped = {}
    for vowel in ('ㅣ', 'ㅡ', 'ㅜ', 'other'):
        subset = [c for c in cases if (c['observed_V'] if c['observed_V'] in ('ㅣ', 'ㅡ', 'ㅜ') else 'other') == vowel]
        grouped[vowel] = {'all_cases': len(subset), 'TAG_V_differs_from_runtime_V': sum(c['actual_TAG_V'] != c['observed_V'] for c in subset)}
        for key in ('raw_early_median', 'corrected_early_median'):
            a = [c[key] for c in subset if c['actual'] == 'ㅅ' and c[key] is not None]
            b = [c[key] for c in subset if c['actual'] == 'ㅆ' and c[key] is not None]
            grouped[vowel][key] = {'S_available': len(a), 'SS_available': len(b),
                'S_median': float(np.median(a)) if a else None, 'SS_median': float(np.median(b)) if b else None,
                'S_above_SS_AUC_TRAIN_only': auc(a, b)}
            matching = [c for c in subset if c['actual_TAG_V'] == c['observed_V']]
            a = [c[key] for c in matching if c['actual'] == 'ㅅ' and c[key] is not None]
            b = [c[key] for c in matching if c['actual'] == 'ㅆ' and c[key] is not None]
            grouped[vowel][key]['matched_TAG_V_only'] = {'S_available': len(a), 'SS_available': len(b),
                'S_above_SS_AUC_TRAIN_only': auc(a, b)}
    passing = []
    for v in ('ㅣ', 'ㅡ', 'ㅜ'):
        for key in ('raw_early_median', 'corrected_early_median'):
            z = grouped[v][key]['matched_TAG_V_only']
            if (min(z['S_available'], z['SS_available']) >= p['minimum_cases_per_actual']
                    and z['S_above_SS_AUC_TRAIN_only'] >= p['minimum_TRAIN_proxy_AUC']):
                passing.append(v+'/'+key)
    report = {'scope': p['scope'], 'all_cases': len(cases), 'frame_states': dict(reasons),
        'groups': grouped, 'passing_small_pilot_signals': passing,
        'measurement_expansion_warranted': bool(passing),
        'versions': {'parselmouth': parselmouth.__version__, 'praat': parselmouth.PRAAT_VERSION},
        'seconds': time.monotonic()-tick, 'deployment': False, 'goal_complete': False}
    (out/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2, allow_nan=False))
    (out/'frames.json').write_text(json.dumps(frames, ensure_ascii=False, allow_nan=False))
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    (out/'finished.json').write_text(json.dumps({'report_sha256': service.sha(out/'report.json')}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

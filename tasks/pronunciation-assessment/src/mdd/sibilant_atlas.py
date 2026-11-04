"""Descriptive SS acoustics, with explicit window support and no learned rules.

TAG intervals are annotation coordinates, not verified acoustic boundaries.
High-band dominance and periodicity are numerical proxies, not frication,
aspiration, glottal onset, or a pronunciation classifier.
"""
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mdd.phone_atlas import configure_font, read_audio, spectrum, audio_html
from mdd.sibilant_observations import noise_observations

FEATURES = {
    'tag_duration_ms': ('TAG 자음 구간 길이', 'ms'),
    'high_run_ms': ('고주파 우세 연속 구간', 'ms · 자동 후보'),
    'centroid_khz': ('자음 중앙의 주파수 무게중심', 'kHz · 1~7.9kHz'),
    'vowel_rise_db': ('뒤 모음 0~30 → 60~90ms 크기 증가', 'dB · 상대 RMS'),
    'vowel_periodicity': ('뒤 모음 처음 40ms의 주기성', '0~1 · 자동 지표'),
}
COLORS = {'ㅅ': '#1976b5', 'ㅆ': '#da6b18'}


def context_key(c):
    # Word position is inferred from existing TAG word/phone times, not spelling.
    return (c['previous'], c['following'], c['position'], c['word_position'])


def context_text(key):
    return '%s _ %s · %s · %s' % tuple(key)


def rank(c):
    return hashlib.sha256(('ss-acoustic-v1:' + c['case_id']).encode()).hexdigest()


def paired_cases(cases):
    """At most one hash-selected pair per speaker x exact context; no predictions."""
    groups = defaultdict(lambda: defaultdict(list))
    for c in cases:
        groups[(c['speaker'], context_key(c))][c['actual']].append(c)
    pairs = []
    for (speaker, key), group in sorted(groups.items()):
        if all(p in group for p in COLORS):
            a, b = [min(group[p], key=rank) for p in COLORS]
            pairs.append(dict(speaker=speaker, context=list(key), s=a['case_id'], ss=b['case_id']))
    return pairs


def _longest_run(frames, mask, min_span=.010):
    groups = []; current = []
    for i, ok in enumerate(mask):
        if ok:
            current.append(i)
        elif current:
            groups.append(current); current = []
    if current:
        groups.append(current)
    # >=10ms of consecutive frame centres, rather than one 10ms window.
    groups = [g for g in groups if frames[g[-1]]['center_s'] - frames[g[0]]['center_s'] >= min_span - 1e-9]
    if not groups:
        return None
    g = max(groups, key=lambda g: (len(g), -g[0]))
    return [frames[g[0]]['support_s'][0], frames[g[-1]]['support_s'][1]]


def periodicity_curve(x, sr):
    """Maximum normalized lag correlation, 80..500Hz, 30ms raw PCM windows.

    No pitch path or frequency estimate. Silence returns None, never voiced.
    Scores can be high for periodic non-speech too; this is not a voicing truth.
    """
    n, hop = round(.030 * sr), round(.005 * sr)
    lags = np.arange(int(np.ceil(sr / 500.)), int(np.floor(sr / 80.)) + 1)
    result = []
    for start in range(0, len(x) - n + 1, hop):
        y = np.array(x[start:start + n], dtype=float, copy=True)
        y -= y.mean(); energy = float(np.mean(y * y))
        value = None
        if energy >= 1e-6:  # -60dB RMS relative to PCM 1.0; fixed display policy
            corr = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * n)) ** 2, 2 * n)[:n]
            cs = np.concatenate(([0.], np.cumsum(y * y)))
            denom = np.sqrt(cs[n - lags] * (cs[n] - cs[lags]))
            valid = denom > 1e-15
            if valid.any():
                value = float(np.clip(np.max(corr[lags[valid]] / denom[valid]), 0., 1.))
        result.append(dict(center_s=(start + (n - 1) / 2) / sr,
                           support_s=[start / sr, (start + n) / sr], value=value))
    return result


def extract(x, sr, consonant, vowel):
    """Times relative to PCM crop. No expected/TAG identity or prediction input."""
    start, end = consonant
    if not 0 <= start < end <= len(x) / sr + 1e-8:
        raise ValueError('Consonant interval outside audio')
    noise = noise_observations(x, sr)
    frames = noise['frames']; pc = periodicity_curve(x, sr)
    contained = lambda r, a, b: r['support_s'][0] >= a - 1e-8 and r['support_s'][1] <= b + 1e-8
    middle = (start + .25 * (end - start), start + .75 * (end - start))
    middle_frames = [r for r in frames if contained(r, *middle) and r['rms_db'] >= -60]
    centroids = [r['centroid_1000_to_top_hz'] for r in middle_frames if r['centroid_1000_to_top_hz'] is not None]
    high_mask = [contained(r, start, end) and r['rms_db'] >= -60
                 and r['upper_lower_3000_db'] is not None and r['upper_lower_3000_db'] >= 0 for r in frames]
    high = _longest_run(frames, high_mask)
    result = dict(tag_duration_ms=(end - start) * 1000,
                  high_run_ms=(high[1] - high[0]) * 1000 if high else None,
                  centroid_khz=float(np.median(centroids)) / 1000 if centroids else None,
                  vowel_rise_db=None, vowel_periodicity=None)
    missing = {}
    if high is None: missing['high_run_ms'] = '고정 조건을 통과한 연속 후보 없음'
    if not centroids: missing['centroid_khz'] = '중앙 50% 안의 유효 10ms 창 없음'
    periodic_run = None
    if vowel is not None:
        va, vb = vowel
        if 0 <= va < vb <= len(x) / sr + 1e-8:
            if vb - va >= .090 - 1e-8:
                chunks = [x[round((va + a) * sr):round((va + b) * sr)] for a, b in ((0., .030), (.060, .090))]
                powers = [float(np.mean((y - y.mean()) ** 2)) for y in chunks]
                if min(powers) >= 1e-10:
                    result['vowel_rise_db'] = float(10 * np.log10(powers[1] / powers[0]))
            pv = [r['value'] for r in pc if contained(r, va, min(vb, va + .040)) and r['value'] is not None]
            if pv: result['vowel_periodicity'] = float(np.median(pv))
            # Search around the annotation transition, without deciding aspiration.
            mask = [contained(r, max(start, va - .080), min(vb, va + .150))
                    and r['value'] is not None and r['value'] >= .7 for r in pc]
            periodic_run = _longest_run(pc, mask, .010)
    if result['vowel_rise_db'] is None: missing['vowel_rise_db'] = '연속된 뒤 모음 90ms 또는 유효 에너지 없음'
    if result['vowel_periodicity'] is None: missing['vowel_periodicity'] = '뒤 모음의 유효 30ms 창 없음'
    return dict(features=result, missing=missing, high_run=high, periodic_run=periodic_run,
                noise=frames, periodicity=pc, middle=list(middle))


class SibilantAtlas:
    def __init__(self, assets):
        self.assets = Path(assets)
        self.report = json.loads((self.assets / 'report.json').read_text())
        self.rows = json.loads((self.assets / 'cases.json').read_text())
        self.by_id = {r['case_id']: r for r in self.rows}
        configure_font()

    def subset(self, context=None, normal=None):
        rows = self.rows
        if context is not None:
            rows = [r for r in rows if list(context_key(r)) == list(context)]
        if normal is not None:
            rows = [r for r in rows if (r['actual'] == r['expected']) == normal]
        return rows

    def support_table(self):
        import pandas as pd
        return pd.DataFrame(self.report['contexts']).rename(columns={
            'label': '앞뒤 소리·위치', 's': 'TAG ㅅ', 'ss': 'TAG ㅆ',
            'pairs': '동일 화자 쌍', 'speakers': '화자 수', 'items': '문항 수',
            's_errors': 'ㅅ 오발음', 'ss_errors': 'ㅆ 오발음'})[
                ['앞뒤 소리·위치', 'TAG ㅅ', 'TAG ㅆ', 'ㅅ 오발음', 'ㅆ 오발음', '동일 화자 쌍', '화자 수', '문항 수']]

    def feature_table(self, context=None):
        import pandas as pd
        rows = self.subset(context)
        table = []
        for feature, (name, unit) in FEATURES.items():
            for phone in COLORS:
                rs = [r for r in rows if r['actual'] == phone]
                vals = [r['features'][feature] for r in rs if r['features'][feature] is not None]
                q = np.percentile(vals, [25, 50, 75]) if vals else [np.nan] * 3
                table.append({'특징': name, '단위': unit, 'TAG': phone,
                    '유효/전체': '%d/%d' % (len(vals), len(rs)), '중앙값': q[1],
                    '25%': q[0], '75%': q[2]})
        return pd.DataFrame(table).round(2)

    def plot_distribution(self, context, normal=None):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        rows = self.subset(context, normal)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), layout='constrained')
        for ax, (feature, (name, unit)) in zip(axes.flat, FEATURES.items()):
            for x, phone in enumerate(COLORS):
                rs = [r for r in rows if r['actual'] == phone and r['features'][feature] is not None]
                values = [r['features'][feature] for r in rs]
                if values:
                    # Deterministic horizontal jitter, with no clipping of outliers.
                    jitter = np.array([(int(rank(r)[:8], 16) / (2**32 - 1) - .5) * .36 for r in rs])
                    for correct, marker in ((True, 'o'), (False, 'x')):
                        take = [i for i, r in enumerate(rs) if (r['predicted'] == r['actual']) == correct]
                        ax.scatter(x + jitter[take], np.array(values)[take], c=COLORS[phone],
                                   marker=marker, s=18 if correct else 42, alpha=.4 if correct else .95,
                                   linewidths=.8, zorder=3 if correct else 5)
                    q = np.percentile(values, [25, 50, 75])
                    ax.plot([x - .27, x - .27], [q[0], q[2]], color='black', lw=3)
                    ax.plot([x - .31, x - .23], [q[1], q[1]], color='black', lw=2)
            ax.set_xticks([0, 1], ['%s\n유효 %d/%d' % (p, sum(r['features'][feature] is not None for r in rows if r['actual']==p), sum(r['actual']==p for r in rows)) for p in COLORS])
            ax.set_title(name, fontsize=11); ax.set_ylabel(unit); ax.grid(axis='y', alpha=.2)
        axes.flat[-1].axis('off')
        status = []
        for p in COLORS:
            rs = [r for r in rows if r['actual'] == p]
            status.append('%s: %d자리 / 전사 불일치 %d\n    LAB≠TAG 오발음 %d' % (
                p, len(rs), sum(r['predicted'] != p for r in rs), sum(r['expected'] != p for r in rs)))
        axes.flat[-1].legend(handles=[Line2D([0], [0], marker='o', color='gray', ls='', label='○ M3 전사 일치'),
            Line2D([0], [0], marker='x', color='gray', ls='', label='× M3 전사 불일치'),
            Line2D([0], [0], color='black', lw=3, label='검정선: 25~75% / 중앙값')], loc='upper left')
        axes.flat[-1].text(0, .58, '\n\n'.join(status) + '\n\n누락은 0으로 채우지 않음\n색 = 실제 TAG / 경고 성능 아님', va='top')
        kind = '전체' if normal is None else 'LAB=TAG 정상만' if normal else 'LAB≠TAG 오발음만'
        fig.suptitle('ㅅ·ㅆ 특징 분포 — ' + context_text(context) + ' · ' + kind + '\n점 하나 = 녹음의 자음 한 자리. 집단이 겹치는 정도와 ×의 위치를 보세요.', fontsize=13)
        return fig

    def plot_pairs(self, context):
        import matplotlib.pyplot as plt
        pairs = [p for p in self.report['pairs'] if p['context'] == list(context)]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
        for ax, key in zip(axes, ('tag_duration_ms', 'high_run_ms', 'vowel_rise_db')):
            n = 0
            for pair in pairs:
                a, b = self.by_id[pair['s']], self.by_id[pair['ss']]
                values = [a['features'][key], b['features'][key]]
                if None in values: continue
                n += 1; ax.plot([0, 1], values, color='#818b96', alpha=.35, lw=.8)
                for x, c in enumerate((a, b)):
                    ax.scatter(x, values[x], color=COLORS[c['actual']], s=20,
                        marker='o' if c['predicted']==c['actual'] else 'x', zorder=3)
            ax.set_xticks([0, 1], ['ㅅ', 'ㅆ']); ax.set_ylabel(FEATURES[key][1]); ax.grid(axis='y', alpha=.2)
            ax.set_title('%s\n유효 %d/%d쌍' % (FEATURES[key][0], n, len(pairs)), fontsize=11)
        fig.suptitle('동일 화자·앞뒤 소리·위치의 쌍 — ' + context_text(context) + '\n문항·발화 속도까지 같지는 않습니다. 선 하나 = 화자 한 명의 고정 해시 선정 쌍.', fontsize=12)
        return fig

    def plot_examples(self, case_ids):
        import matplotlib.pyplot as plt
        cases = [self.by_id[k] for k in case_ids]
        loaded = [read_audio(c['audio_path'], c['clip_start'], c['clip_end']) for c in cases]
        limit = max(.02, max(np.max(np.abs(x)) for x, _, _ in loaded) * 1.05)
        fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3.3 * len(cases)), layout='constrained', squeeze=False)
        xmax = max(c['clip_end'] - c['tag_start'] for c in cases) * 1000
        for i, (c, (x, sr, offset)) in enumerate(zip(cases, loaded)):
            start, end = c['tag_start'] - offset, c['tag_end'] - offset
            vowel = [v - offset for v in c['vowel_span']] if c['vowel_span'] else None
            obs = extract(x, sr, [start, end], vowel)
            shift = (offset - c['tag_start']) * 1000
            ax = axes[i, 0]; ax.plot(np.arange(len(x)) / sr * 1000 + shift, x, lw=.5, color=COLORS[c['actual']]); ax.set_ylim(-limit, limit)
            ax.set_ylabel('원래 PCM 진폭')
            freq, t, db = spectrum(x, sr)
            im = axes[i, 1].pcolormesh(t * 1000 + shift, freq / 1000, db, cmap='magma', vmin=-100, vmax=0, shading='auto')
            axes[i, 1].set_ylim(0, 8); axes[i, 1].set_ylabel('kHz')
            noise, pc = obs['noise'], obs['periodicity']
            ax = axes[i, 2]
            ax.plot([r['center_s'] * 1000 + shift for r in noise],
                    [r['upper_lower_3000_db'] if r['rms_db'] >= -60 else np.nan for r in noise], color='#a5540b', label='고주파/저주파 비')
            ax.axhline(0, color='#a5540b', ls=':', lw=.7); ax.set_ylabel('고주파/저주파 (dB)'); ax.set_ylim(-60, 60)
            twin = ax.twinx(); twin.plot([r['center_s'] * 1000 + shift for r in pc], [r['value'] for r in pc], color='#722e9b', lw=1, label='주기성')
            twin.set_ylim(0, 1.05); twin.set_ylabel('주기성 (보라)')
            for ax in axes[i]:
                ax.axvspan(0, c['duration_ms'], color='#31a26b', alpha=.09)
                ax.axvline(0, color='#31a26b', ls='--', lw=.8); ax.axvline(c['duration_ms'], color='#31a26b', ls='--', lw=.8)
                if vowel:
                    ax.axvspan(vowel[0]*1000+shift, min(vowel[1],vowel[0]+.030)*1000+shift, color='#1976b5', alpha=.13)
                    if vowel[1]-vowel[0] >= .09-1e-8:
                        ax.axvspan((vowel[0]+.060)*1000+shift, (vowel[0]+.090)*1000+shift, color='#1976b5', alpha=.13)
                if obs['high_run']:
                    a, b = np.array(obs['high_run']) * 1000 + shift
                    ax.plot([a, b], [.93, .93], transform=ax.get_xaxis_transform(), color='#e88d16', lw=4)
                if obs['periodic_run']:
                    a, b = np.array(obs['periodic_run']) * 1000 + shift
                    ax.plot([a, b], [.84, .84], transform=ax.get_xaxis_transform(), color='#722e9b', lw=4)
                ax.set_xlim(-100, xmax); ax.set_xlabel('TAG 자음 시작 기준 (ms)')
            axes[i, 0].set_title('기대 %s → TAG %s → M3 %s\n%s' % (c['expected'], c['actual'], c['predicted'] or '∅', c['prompt']), fontsize=10)
            axes[i, 1].set_title('녹색: TAG / 파란 음영: 모음 측정 창', fontsize=10)
            axes[i, 2].set_title('주황 막대: 고주파 우세 후보\n보라 막대: 주기성 높은 후보', fontsize=10)
        fig.colorbar(im, ax=axes[:, 1], shrink=.75, label='고정 STFT dB · 음압 아님')
        fig.suptitle('같은 화자·앞뒤 소리·위치 — 파형에서 측정 구간 찾기\n자동 막대는 기식/VOT/성대 진동의 확정 경계가 아닙니다.', fontsize=12)
        return fig

    def example_audio(self, case_ids):
        result = []
        for key in case_ids:
            c = self.by_id[key]
            result.append('<p><b>TAG %s / M3 %s</b> · %s · 화자 %s<br>%s<br>%s</p>' % (
                c['actual'], c['predicted'] or '∅', html.escape(c['prompt']), c['speaker'],
                html.escape(context_text(context_key(c))), audio_html(c)))
        return ''.join(result)

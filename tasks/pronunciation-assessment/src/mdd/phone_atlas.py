"""Read-only audio atlas and TAG-versus-model summaries. No model inference."""
import base64
import csv
import hashlib
import html
import io
import json
import wave
from collections import Counter
import os
from pathlib import Path

import numpy as np

CONSONANTS = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
COMPOUNDS = list('ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ')
VOWELS = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
PHONES = CONSONANTS + VOWELS + COMPOUNDS
GROUPS = ('ㄱㄲㅋ', 'ㄷㄸㅌ', 'ㅂㅃㅍ', 'ㅅㅆ', 'ㅈㅉㅊ', 'ㄴㄹㅁㅇ', 'ㅏㅓㅗㅜㅡㅣ')
DEL = '∅'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pct(k, n):
    return 100.0 * k / n if n else None


def summarize(sites, vocab):
    """Fixed saved canonical-site denominator; actual-label and expected-label views."""
    actual, expected, confusion = [], [], []
    for phone in PHONES:
        rows = [s for s in sites if s['actual'] == phone]
        normal = [s for s in rows if s['expected'] == phone]
        errors = [s for s in rows if s['expected'] != phone]
        hit = lambda rs: sum(s['predicted'] == s['actual'] for s in rs)
        wrong = Counter((s['predicted'] or DEL) for s in rows if s['predicted'] != phone)
        support = '표본 없음' if not rows else '희소 표본 (<30)' if len(rows) < 30 else '평가 가능'
        actual.append(dict(phone=phone, in_model_vocab=phone in vocab, support=support,
            n=len(rows), correct=hit(rows), accuracy_pct=pct(hit(rows), len(rows)),
            normal_n=len(normal), normal_correct=hit(normal), normal_accuracy_pct=pct(hit(normal), len(normal)),
            error_n=len(errors), error_correct=hit(errors), error_accuracy_pct=pct(hit(errors), len(errors)),
            errors_warned_exactly=sum(s['warning'] and s['predicted'] == phone for s in errors),
            false_warnings=sum(s['warning'] for s in normal),
            items=len({s['item'] for s in rows}), speakers=len({s['speaker'] for s in rows}),
            top_confusions=' / '.join('%s %d건' % (k, v) for k, v in wrong.most_common(3))))
        for pred, count in Counter(s['predicted'] or DEL for s in rows).items():
            confusion.append(dict(actual=phone, predicted=pred, count=count,
                                  row_pct=pct(count, len(rows))))
        rs = [s for s in sites if s['expected'] == phone]
        er = [s for s in rs if s['actual'] != phone]
        expected.append(dict(phone=phone, n=len(rs), normal_n=len(rs)-len(er), error_n=len(er),
            exact_transcriptions=hit(er), exact_transcription_pct=pct(hit(er), len(er)),
            exact_warnings=sum(s['warning'] and s['predicted'] == s['actual'] for s in er),
            exact_warning_pct=pct(sum(s['warning'] and s['predicted'] == s['actual'] for s in er), len(er)),
            expected_restored=sum(s['predicted'] == phone for s in er),
            false_warnings=sum(s['warning'] for s in rs if s['actual'] == phone)))
    return actual, expected, confusion


def read_audio(path, start=None, end=None):
    with wave.open(str(path), 'rb') as f:
        sr, frames = f.getframerate(), f.getnframes()
        if (sr, f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Expected 16kHz mono PCM16: ' + str(path))
        a = 0 if start is None else max(0, min(frames, int(round(start * sr))))
        b = frames if end is None else max(a, min(frames, int(round(end * sr))))
        f.setpos(a)
        raw = f.readframes(b-a)
    return np.frombuffer(raw, '<i2').astype(np.float64) / 32768.0, sr, a / sr


def wav_bytes(path, start=None, end=None):
    # Copy original PCM, without per-clip loudness normalization.
    with wave.open(str(path), 'rb') as f:
        sr = f.getframerate()
        a = 0 if start is None else max(0, min(f.getnframes(), int(round(start * sr))))
        b = f.getnframes() if end is None else min(f.getnframes(), int(round(end * sr)))
        f.setpos(a)
        raw = f.readframes(max(0, b-a)); params = f.getparams()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as g:
        g.setparams(params); g.writeframes(raw)
    return buf.getvalue()


def audio_html(case, full=False):
    args = () if full else (case['clip_start'], case['clip_end'])
    data = base64.b64encode(wav_bytes(case['audio_path'], *args)).decode('ascii')
    return '<audio controls preload="none" src="data:audio/wav;base64,%s"></audio>' % data


def configure_font():
    import matplotlib
    from matplotlib import font_manager
    # 한글 글꼴 경로는 환경변수 MDD_FONT (예: NanumGothic-Regular.ttf). 없으면 기본 글꼴
    p = Path(os.environ.get('MDD_FONT', ''))
    if os.environ.get('MDD_FONT') and p.is_file():
        font_manager.fontManager.addfont(str(p))
        matplotlib.rcParams['font.family'] = [font_manager.FontProperties(fname=str(p)).get_name(), 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


def spectrum(x, sr):
    from scipy.signal import stft
    n = min(400, len(x))
    if n < 2:
        raise ValueError('Audio crop too short')
    f, t, z = stft(x, fs=sr, window='hann', nperseg=n, noverlap=max(0, n-80),
                  nfft=512, detrend=False, boundary=None, padded=False)
    # Fixed numerical reference: STFT magnitude of PCM normalized to [-1,1].
    # Not calibrated SPL, and no independent maximum normalization per example.
    return f, t, 20 * np.log10(np.maximum(np.abs(z), 1e-6))


def render_cases(cases, title='', shared_wave_limit=None):
    import matplotlib.pyplot as plt
    configure_font()
    if not cases:
        raise ValueError('No annotated audio examples available')
    loaded = [read_audio(c['audio_path'], c['clip_start'], c['clip_end']) for c in cases]
    amp = shared_wave_limit or max(.02, max(float(np.abs(x).max()) for x, _, _ in loaded) * 1.05)
    xmin = min(offset-c['tag_start'] for c, (_, _, offset) in zip(cases, loaded))
    xmax = max(offset+len(x)/sr-c['tag_start'] for c, (x, sr, offset) in zip(cases, loaded))
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 2.9*len(cases)), squeeze=False,
                             layout='constrained')
    mesh = None
    for i, (c, (x, sr, offset)) in enumerate(zip(cases, loaded)):
        shift = offset-c['tag_start']
        times = np.arange(len(x))/sr + shift
        axes[i, 0].plot(times*1000, x, lw=.5, color='#20618a')
        axes[i, 0].set_ylim(-amp, amp)
        axes[i, 0].set_ylabel('파형 · PCM 진폭')
        freq, t, db = spectrum(x, sr)
        mesh = axes[i, 1].pcolormesh((t+shift)*1000, freq/1000, db, vmin=-100, vmax=0,
                                    cmap='magma', shading='auto')
        axes[i, 1].set_ylim(0, 8); axes[i, 1].set_ylabel('주파수 (kHz)')
        for ax in axes[i]:
            ax.axvline(0, color='#24ad72', ls='--', lw=1)
            ax.axvline(c['duration_ms'], color='#24ad72', ls='--', lw=1)
            ax.axvspan(0, c['duration_ms'], color='#24ad72', alpha=.08)
            ax.set_xlim(xmin*1000, xmax*1000)
            ax.set_xlabel('TAG 시작 기준 시간 (ms) · 녹음 길이 유지')
        output = (c['predicted'] or DEL) if c.get('has_prediction') else '미측정'
        warning = ('경고' if c['warning'] else '경고 없음') if c.get('has_prediction') else '미측정'
        axes[i, 0].set_title('%s | 기대 %s → TAG %s → M3 %s | %s' %
                            (c['case_id'], c['expected'] or '미확정', c['actual'], output, warning), fontsize=10)
        axes[i, 1].set_title('%s · %s · 다음음 %s · TAG %.0fms' %
                            (c['item'], c['position'], c['following'], c['duration_ms']), fontsize=10)
    fig.colorbar(mesh, ax=axes[:, 1], label='STFT 크기 (dB, 고정 기준 / 음압 dB SPL 아님)', shrink=.75)
    fig.suptitle(title+'\n초록선 = 기존 TAG 구간. 독립 판정한 물리적 경계가 아닙니다.', fontsize=12)
    return fig


class Atlas:
    def __init__(self, assets):
        self.assets = Path(assets)
        self.report = json.loads((self.assets/'report.json').read_text())
        self.examples = json.loads((self.assets/'examples.json').read_text())
        self.groups = json.loads((self.assets/'comparison_groups.json').read_text())

    def table(self, kind='actual'):
        import pandas as pd
        return pd.read_csv(self.assets / ('per_phone_%s.csv' % kind), keep_default_na=True)

    def show_phone(self, phone, n=2, mode='전체', following=None, position=None, full_audio=False):
        from IPython.display import display, HTML
        import matplotlib.pyplot as plt
        if phone not in PHONES:
            raise ValueError('현대 한글 자모 한 개를 입력하세요: 예) ㅅ, ㅏ, ㄺ')
        rows = [r for r in self.examples if r['actual'] == phone]
        if mode == '정상': rows = [r for r in rows if r['expected'] == r['actual']]
        if mode == '오발음': rows = [r for r in rows if r['expected'] is not None and r['expected'] != r['actual']]
        if following is not None: rows = [r for r in rows if r['following'] == following]
        if position is not None: rows = [r for r in rows if r['position'] == position]
        display(HTML(self.phone_summary_html(phone)))
        if not rows:
            display(HTML('<p>이 조건에서 확인 가능한 TAG 시간·음성 사례가 없습니다. 0%라는 뜻이 아닙니다.</p>'))
            return
        selected = rows[:n]
        fig = render_cases(selected, '실제 TAG '+phone+'의 음성 사례')
        display(fig); plt.close(fig)
        for c in selected:
            display(HTML(self.case_html(c, full_audio)))

    def phone_summary_html(self, phone):
        r = next(r for r in self.report['actual_phone_metrics'] if r['phone'] == phone)
        fmt = lambda k, n: '%d/%d (%.2f%%)' % (k, n, 100*k/n) if n else '자료 없음'
        return '<h3>실제 TAG: %s</h3><p>전체 전사 일치 %s · 정상 %s · 오발음 %s</p><p>자주 다른 출력: %s</p>' % (
            phone, fmt(r['correct'], r['n']), fmt(r['normal_correct'], r['normal_n']),
            fmt(r['error_correct'], r['error_n']), html.escape(r['top_confusions'] or '없음'))

    def case_html(self, c, full_audio=False):
        seq = '기대 LAB: %s<br>실제 TAG: %s' % (html.escape(c['canonical_sequence']), html.escape(c['tag_sequence']))
        out = (c['predicted'] or DEL) if c.get('has_prediction') else '미측정'
        info = '<p><b>%s</b> %s<br>기대 %s / TAG %s / 모델 %s · %s · 다음음 %s<br>%s<br><small>%s</small></p>' % (
            c['case_id'], html.escape(c['prompt']), c['expected'] or '미확정', c['actual'], out,
            c['position'], c['following'], seq, html.escape(c['uid']))
        info += '<p>주변 구간 '+audio_html(c)+'</p>'
        if full_audio: info += '<p>원래 녹음 '+audio_html(c, full=True)+'</p>'
        return info

    def show_group(self, group):
        from IPython.display import display, HTML
        import matplotlib.pyplot as plt
        g = self.groups.get(group)
        if not g or not g['case_ids']:
            display(HTML('<p>선정된 비교 사례가 없습니다.</p>')); return
        rows = {c['case_id']: c for c in self.examples}
        cases = [rows[k] for k in g['case_ids']]
        display(HTML('<p>'+html.escape(g['description'])+'</p>'))
        fig = render_cases(cases, group+' 비교')
        display(fig); plt.close(fig)
        for c in cases: display(HTML(self.case_html(c)))

    def widgets(self):
        from IPython.display import display, HTML
        try:
            import ipywidgets as w
        except ImportError:
            display(HTML('<p>선택 위젯이 없는 환경입니다. 아래 <code>atlas.show_phone("ㅅ")</code>를 실행하거나 함께 만든 HTML의 선택 메뉴를 사용하세요.</p>'))
            return
        phone = w.Dropdown(options=PHONES, value='ㅅ', description='실제 TAG')
        mode = w.Dropdown(options=['전체','정상','오발음'], description='사례')
        n = w.IntSlider(value=2, min=1, max=8, description='표시 수')
        output = w.interactive_output(self.show_phone, {'phone':phone, 'mode':mode, 'n':n})
        display(w.VBox([w.HBox([phone,mode,n]), output]))

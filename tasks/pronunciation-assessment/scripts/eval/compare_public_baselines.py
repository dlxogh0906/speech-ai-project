"""같은 시험 녹음에서 여러 모델의 자리별 채점 결과(`*.utt.jsonl`)를 나란히 집계한다.

`eval_actual_ctc.py score` 가 모델마다 남긴 `<이름>.utt.jsonl` 을 읽어, 발표 자료에 쓴 기준을 한 번에 다시 계산한다.

- 오류가 있는 녹음의 인식률: 정답 레이블 ≠ 표준 발음인 녹음에서 1 − PER (전체 · 단어 · 문장 · 모국어별)
- 검출 재현율 · 정밀도 · F1 · 헛지적(FRR) · 진단 정확도(1 − DER)
- 자음 계열별(ㄱ·ㄲ·ㅋ 등) 오류가 있는 단어 녹음의 인식률
- 두 모델 판정이 갈린 오류 자리에서 맞은 쪽 (검출 기준 · 소리까지 기준 · 정상 자리 포함 전체)

예 (재현 명령은 results/24 에 있다):
    $PY compare_public_baselines.py --utt slplab=<...>/slplab_test_unseen.utt.jsonl \
        --utt m3=<...>/m3_test_unseen.utt.jsonl --pair slplab m3 --out cmp.json
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
from shared.paths import MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)

import argparse
import collections
import json

SERIES = [('ㄱ·ㄲ·ㅋ', 'ㄱㄲㅋ'), ('ㅂ·ㅃ·ㅍ', 'ㅂㅃㅍ'), ('ㄷ·ㄸ·ㅌ', 'ㄷㄸㅌ'), ('ㅈ·ㅉ·ㅊ', 'ㅈㅉㅊ'), ('ㅅ·ㅆ', 'ㅅㅆ')]


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def acc(evs):
    n = sum(e['N'] for e in evs)
    if not n: return None
    return round(100 - 100 * sum(e['S'] + e['D'] + e['I'] for e in evs) / n, 1)


def mdd(evs):
    mis = [m for e in evs for m in e['mis']]
    cls = collections.Counter(m[4] for m in mis)
    n_normal = sum(e['n_normal'] for e in evs)
    fr = sum(1 for e in evs for x in e['nerr'] if x[2] != x[1])
    tr = cls['preserve'] + cls['other']; fa = cls['revert']
    pr = tr / (tr + fr) if tr + fr else 0.0; rc = tr / (tr + fa) if tr + fa else 0.0
    return {'recall': round(100 * rc, 1), 'precision': round(100 * pr, 1),
            'f1': round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0,
            'frr': round(100 * len([x for e in evs for x in e['nerr']]) / n_normal, 2) if n_normal else None,
            'diag': round(100 * cls['preserve'] / tr, 1) if tr else None,
            'per': round(100 * sum(e['S'] + e['D'] + e['I'] for e in evs) / sum(e['N'] for e in evs), 2) if evs else None,
            'error_sites': len(mis), 'detected': tr}


def in_series(x, s):
    return x is not None and x in s


def disagreement(a, b):
    """a, b: 같은 순서의 utt 목록. 오류 자리에서 한쪽만 잡은 곳, 소리까지 맞힌 곳, 정상 자리 포함 전체."""
    det = [0, 0]; exact = [0, 0]; every = [0, 0]
    for ea, eb in zip(a, b):
        assert ea['i'] == eb['i']
        mb = {x[0]: x[4] for x in eb['mis']}
        for x in ea['mis']:
            j = x[0]; ca, cb = x[4], mb[j]
            da, db = ca != 'revert', cb != 'revert'
            if da != db: det[0 if da else 1] += 1
            pa, pb = ca == 'preserve', cb == 'preserve'
            if pa != pb: exact[0 if pa else 1] += 1; every[0 if pa else 1] += 1
        na = {x[0] for x in ea['nerr']}; nb = {x[0] for x in eb['nerr']}
        for j in na ^ nb:
            every[1 if j in na else 0] += 1   # 틀린 쪽의 반대가 맞은 쪽
    r = lambda p: {'a': p[0], 'b': p[1], 'b_rate': round(100 * p[1] / sum(p), 1) if sum(p) else None}
    return {'error_sites_detect': r(det), 'error_sites_exact_sound': r(exact), 'all_sites': r(every)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, default=MDD_OUTPUTS / 'manifest_foreign_clean_v2')
    ap.add_argument('--split', default='test_unseen')
    ap.add_argument('--utt', action='append', required=True, metavar='이름=경로', help='모델별 utt.jsonl (여러 번)')
    ap.add_argument('--pair', nargs=2, metavar=('기준', '우리'), help='판정이 갈린 자리 비교에 쓸 두 모델 이름')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    M = a.manifest
    meta = load_jsonl(M / (a.split + '.meta.jsonl'))
    ltr = [l.split() for l in open(M / (a.split + '.ltr'), encoding='utf-8').read().splitlines()]
    can = [l.split() for l in open(M / (a.split + '.canon'), encoding='utf-8').read().splitlines()]
    has_err = [x != y for x, y in zip(ltr, can)]
    word = [m['task'] == 'RW' for m in meta]
    lang = [m['language'] for m in meta]
    U = {}
    for spec in a.utt:
        name, path = spec.split('=', 1); U[name] = load_jsonl(path)
        assert len(U[name]) == len(meta), (name, len(U[name]), len(meta))

    out = {'manifest': str(M), 'split': a.split, 'utterances': len(meta),
           'error_recordings': sum(has_err), 'error_word_recordings': sum(1 for e, w in zip(has_err, word) if e and w),
           'models': {}, 'series': {}, 'by_language': {}}
    langs = sorted(set(lang))
    for name, evs in U.items():
        sel = lambda f: [e for e in evs if f(e['i'])]
        out['models'][name] = dict(mdd(evs),
            err_acc_all=acc(sel(lambda i: has_err[i])),
            err_acc_word=acc(sel(lambda i: has_err[i] and word[i])),
            err_acc_sentence=acc(sel(lambda i: has_err[i] and not word[i])))
        for g in langs:
            d = out['by_language'].setdefault(g, {})
            d[name] = {'err_acc_word': acc(sel(lambda i: has_err[i] and word[i] and lang[i] == g)),
                       'diag': mdd(sel(lambda i: lang[i] == g))['diag']}
    first = next(iter(U.values()))
    for label, s in SERIES:
        rw = [e['i'] for e in first if word[e['i']] and any(in_series(x[1], s) and in_series(x[2], s) for x in e['mis'])]
        def sdiag(evs):
            c = collections.Counter(x[4] for e in evs for x in e['mis'] if in_series(x[1], s) and in_series(x[2], s))
            t = c['preserve'] + c['other']
            return round(100 * c['preserve'] / t, 1) if t else None
        out['series'][label] = {'word_recordings': len(rw),
                                'err_acc_word': {n: acc([U[n][i] for i in rw]) for n in U},
                                'diag': {n: sdiag(U[n]) for n in U}}
        if a.pair:
            pa, pb = U[a.pair[0]], U[a.pair[1]]
            sub_a = [dict(e, mis=[x for x in e['mis'] if in_series(x[1], s) and in_series(x[2], s)], nerr=[]) for e in pa]
            sub_b = [dict(e, mis=[x for x in e['mis'] if in_series(x[1], s) and in_series(x[2], s)], nerr=[]) for e in pb]
            out['series'][label]['disagreement_detect'] = disagreement(sub_a, sub_b)['error_sites_detect']
    if a.pair:
        out['pair'] = {'a': a.pair[0], 'b': a.pair[1], **disagreement(U[a.pair[0]], U[a.pair[1]])}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[cmp]', a.out, {n: (v['err_acc_word'], v['diag'], v['f1']) for n, v in out['models'].items()})


if __name__ == '__main__':
    main()

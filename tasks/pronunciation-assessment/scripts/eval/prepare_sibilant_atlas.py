"""Build a fixed-context SS acoustic census from saved M3 output and real PCM."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
from shared.paths import DATA_ROOT, MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict

from prepare_phone_atlas import normalized_timing, make_case, write_json, write_csv
from mdd.phone_atlas import sha, read_audio, VOWELS
from mdd.h0_phone_audit import optimal_mapping
from mdd.sibilant_atlas import FEATURES, context_key, context_text, paired_cases, extract, rank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--atlas', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    manifest = MDD_OUTPUTS/'manifest_full_tag_replay10'
    protocol = dict(version='SS_acoustic_atlas_v1', model='M3 saved ft100_ds50m',
        training_updates=0, model_inference_runs=0, audio_listened=False,
        population='All accepted saved sites whose actual TAG is ㅅ or ㅆ; missing mappings retained in inventory.',
        contexts='Immediate original TAG neighbors, inferred onset/coda and TAG word-initial/internal position. Unknown/gapped neighbor is not treated as adjacent.',
        default_context='Among known onset vowel contexts with >=1 shared speaker pair, maximize min(n_s,n_ss), then pair count, then total n. No feature/model-output selection.',
        examples='One pair per speaker x context chosen by SHA256; displayed pair is minimum combined case hash in default context. No favorable acoustic selection.',
        noise='Original PCM; DC removal per frame; Hann 10ms / step2ms / FFT1024; band 100..3000 versus 3000..7900Hz.',
        high_run='Longest run: full frame inside TAG consonant, RMS >= -60dB relative PCM1, high/low >=0dB, >=10ms between centres. Missing if absent, not zero. Not frication duration.',
        centroid='Median 1..7.9kHz power centroid of 10ms frames wholly within middle50% of TAG consonant, RMS >= -60dB.',
        vowel_rise='10log10(DC-removed PCM power[60..90ms]/power[0..30ms]) relative to contiguous following TAG vowel start; both windows must exist and power>=1e-10.',
        periodicity='Max normalized autocorrelation at 80..500Hz lags, raw 30ms frames/hop5ms; RMS >= -60dB. Median fully contained within first40ms of TAG vowel. Not validated F0.',
        periodic_run='Around TAG transition: fully contained 30ms windows from vowel_start-80ms to vowel_start+150ms; score>=.7 for >=10ms centres. Display candidate only.',
        limitations=['No physical frication/aspiration segmentation, F0/F1 or independent listening.',
                     'No classifier or threshold fitted; no causal claim from descriptive separation.',
                     'Known training prompts; word/stress/rate/environment not fully controlled.',
                     'TAG conditioning/annotation timing is unavailable to a production detector.'])
    write_json(args.out / 'protocol.json', protocol)
    with gzip.open(args.atlas / 'sites.jsonl.gz', 'rt', encoding='utf-8') as f:
        sites = [json.loads(l) for l in f]
    by_uid = defaultdict(list)
    for s in sites:
        if s['accepted']: by_uid[s['uid']].append(s)
    target_uids = {s['uid'] for s in sites if s['accepted'] and s['actual'] in ('ㅅ', 'ㅆ')}
    source_count = sum(s['accepted'] and s['actual'] in ('ㅅ', 'ㅆ') for s in sites)
    audio_paths = {}
    with (manifest / 'test.tsv').open() as f:
        audio_root = Path(next(f).strip())
        for line in f:
            rel, frames = line.rstrip().split('\t'); uid = Path(rel).stem
            if uid in target_uids: audio_paths[uid] = (str(audio_root / rel), int(frames))
    metas = {}
    with (manifest / 'test.meta.jsonl').open() as f:
        for line in f:
            m = json.loads(line)
            if m['stem'] in target_uids: metas[m['stem']] = m
    assert set(metas) == set(audio_paths) == target_uids
    print('Loaded %d SS sites, %d recordings' % (source_count, len(target_uids)), flush=True)
    cases = []; excluded = []; audio_hashes = {}; checked_labels = {}; count = Counter()
    bases = {p.name[:3]: p for p in (DATA_ROOT/'aihub').glob('04[0-3].*')}
    sys.path.insert(0, str(ROOT / 'tasks/pronunciation-assessment/scripts/data'))
    from build_foreign_clean_v2 import parse_one
    for number, (uid, m) in enumerate(sorted(metas.items()), 1):
        rs = by_uid[uid]
        tag, spans = normalized_timing(m['tag_phones']); canon, _ = normalized_timing(m['lab_phones'])
        smap = optimal_mapping(tuple(s['expected'] for s in rs), tuple(canon))
        tmap = optimal_mapping(tuple(canon), tuple(tag))
        wav, frames = audio_paths[uid]
        m.update(_tag=tag, _canon=canon, _spans=spans, _wav=wav, _frames=frames)
        # Map flat token indices back to their original interval indices so an
        # unknown or silent interval cannot disappear from the context check.
        original_indices = []
        for j, (_, _, value) in enumerate(m['tag_phones']):
            original_indices.extend([j] * sum('\u3131' <= c <= '\u3163' for c in value.strip()))
        assert len(original_indices) == len(tag)
        for i, s in enumerate(rs):
            if s['actual'] not in ('ㅅ', 'ㅆ'): continue
            c = None; reason = None
            if len(smap[i]) != 1 or None in smap[i]: reason = 'saved_to_LAB_ambiguous'
            else:
                q = next(iter(smap[i])); ts = tmap[q]
                if canon[q] != s['expected']: reason = 'saved_to_LAB_disagreement'
                elif len(ts) != 1 or None in ts: reason = 'LAB_to_TAG_ambiguous_or_deleted'
                else:
                    t = next(iter(ts))
                    if tag[t] != s['actual']: reason = 'saved_TAG_disagreement'
                    else:
                        c = make_case(m, q, t, s)
                        if c is None: reason = 'no_single_valid_TAG_time'
            if reason:
                excluded.append(dict(uid=uid, mapped_index=s['mapped_index'], actual=s['actual'], reason=reason)); continue
            t = c['tag_index']; j = original_indices[t]
            def neighbor(delta):
                k = j + delta
                if k < 0: return '<시작>'
                if k >= len(m['tag_phones']): return '<끝>'
                a, b, label = m['tag_phones'][k]
                gap = c['tag_start'] - b if delta < 0 else a - c['tag_end']
                return label if len(label) == 1 and label in 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ' + ''.join(VOWELS) and abs(gap) <= .0020001 else '<불명/틈>'
            c['previous'], c['following'] = neighbor(-1), neighbor(1)
            words = [(a, b, w) for a, b, w in m['tag_words']
                     if a <= c['tag_start'] + .002 and b >= c['tag_end'] - .002]
            c['word_position'] = ('어절 처음' if abs(words[0][0] - c['tag_start']) <= .0020001 else '어절 내부') if len(words) == 1 else '어절 미확정'
            c['word'] = words[0][2] if len(words) == 1 else None
            c['vowel_span'] = None
            if c['following'] in VOWELS and t + 1 < len(spans) and spans[t + 1] is not None and original_indices[t + 1] == j + 1:
                a, b = spans[t + 1]
                if a < c['clip_end']: c['vowel_span'] = [a, min(b, c['clip_end'])]
            x, sr, offset = read_audio(wav, c['clip_start'], c['clip_end'])
            result = extract(x, sr, [c['tag_start'] - offset, c['tag_end'] - offset],
                             [v - offset for v in c['vowel_span']] if c['vowel_span'] else None)
            c['features'] = result['features']; c['feature_missing'] = result['missing']
            c['high_run'] = [v + offset for v in result['high_run']] if result['high_run'] else None
            c['periodic_run'] = [v + offset for v in result['periodic_run']] if result['periodic_run'] else None
            if wav not in audio_hashes: audio_hashes[wav] = sha(wav)
            c['audio_sha256'] = audio_hashes[wav]
            if uid not in checked_labels:
                base = bases[m['ds']] / '01-1.정식개방데이터' / ('Training' if m['part'] == 'TL' else 'Validation') / '02.라벨링데이터'
                lab = base / (m['part'] + '_pronunciation_lab') / (uid + '.json')
                tg = base / (m['part'] + '_pronunciation_tag') / (uid + '.TextGrid')
                raw = parse_one((uid, m['ds'], m['part'], str(tg), str(lab)))
                assert raw['tag_phones'] == m['tag_phones'] and raw['lab_phones'] == m['lab_phones']
                assert raw['tag_words'] == m['tag_words']
                checked_labels[uid] = dict(lab=str(lab), tag=str(tg), lab_sha256=sha(lab), tag_sha256=sha(tg))
            cases.append(c)
        if number % 400 == 0: print('Processed %d/%d recordings; %d cases' % (number, len(metas), len(cases)), flush=True)
    assert len(cases) + len(excluded) == source_count
    groups = defaultdict(list)
    for c in cases: groups[context_key(c)].append(c)
    pairs = paired_cases(cases)
    contexts = []
    for key, rs in groups.items():
        pc = [p for p in pairs if tuple(p['context']) == key]
        contexts.append(dict(context=list(key), label=context_text(key), s=sum(r['actual']=='ㅅ' for r in rs),
            ss=sum(r['actual']=='ㅆ' for r in rs), pairs=len(pc), speakers=len({r['speaker'] for r in rs}),
            items=len({r['item'] for r in rs}), s_errors=sum(r['actual']=='ㅅ' and r['expected']!='ㅅ' for r in rs),
            ss_errors=sum(r['actual']=='ㅆ' and r['expected']!='ㅆ' for r in rs)))
    contexts.sort(key=lambda r: (-min(r['s'], r['ss']), -r['pairs'], -(r['s']+r['ss']), r['label']))
    eligible = [r for r in contexts if r['pairs'] and r['context'][1] in VOWELS
                and r['context'][2] == '초성 추정' and r['context'][3] != '어절 미확정' and '<불명/틈>' not in r['context']]
    default = eligible[0]['context'] if eligible else None
    default_pairs = [p for p in pairs if p['context'] == default]
    pair = min(default_pairs, key=lambda p: hashlib_key(p['s'] + p['ss'])) if default_pairs else None
    examples = [pair['s'], pair['ss']] if pair else []
    # Export every site/missing feature, not just a clean complete-case subset.
    flattened = []
    for c in cases:
        row = {k: c[k] for k in ('case_id','uid','speaker','item','expected','actual','predicted','warning','previous','following','position','word_position','tag_start','tag_end')}
        row.update(c['features']); row['transcription_correct'] = c['predicted'] == c['actual']
        row['normal_TAG'] = c['expected'] == c['actual']; flattened.append(row)
    write_csv(args.out / 'features.csv', flattened)
    write_json(args.out / 'cases.json', cases)
    write_json(args.out / 'excluded.json', excluded)
    report = dict(version=protocol['version'], source_sites=source_count, timed_cases=len(cases),
        excluded=len(excluded), excluded_reasons=dict(Counter(r['reason'] for r in excluded)),
        actual_counts=dict(Counter(c['actual'] for c in cases)), recordings=len(audio_hashes),
        normal_cases=sum(c['expected']==c['actual'] for c in cases),
        contexts=contexts, default_context=default, pairs=pairs, example_ids=examples,
        missing_features={k: dict(Counter(c['feature_missing'].get(k,'available') for c in cases)) for k in FEATURES},
        sources_sha256={str(p): sha(p) for p in (args.atlas/'sites.jsonl.gz', args.atlas/'report.json', manifest/'test.tsv', manifest/'test.meta.jsonl')},
        source_code_sha256={str(p): sha(p) for p in (Path(__file__), ROOT/'tasks/pronunciation-assessment/src/mdd/sibilant_atlas.py', ROOT/'tasks/pronunciation-assessment/src/mdd/sibilant_observations.py')},
        verified_annotations=checked_labels, model_inference_runs=0, training_updates=0)
    write_json(args.out / 'report.json', report)
    write_json(args.out / 'finished.json', dict(status='complete', source_sites=source_count, timed_cases=len(cases),
        excluded=len(excluded), outputs_sha256={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in report.items() if k in ('source_sites','timed_cases','excluded_reasons','default_context','actual_counts','missing_features')}, ensure_ascii=False), flush=True)


def hashlib_key(text):
    import hashlib
    return hashlib.sha256(('ss-pair-v1:' + text).encode()).hexdigest()


if __name__ == '__main__': main()

"""Prepare small, reproducible M3 all-jamo audio atlas from saved evaluations."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src')]
from shared.paths import DATA_ROOT, MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import wave
from collections import Counter, defaultdict

from mdd.phone_atlas import PHONES, VOWELS, CONSONANTS, COMPOUNDS, GROUPS, sha, summarize
from mdd.h0_phone_audit import optimal_mapping


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def normalized_timing(intervals):
    tokens, spans = [], []
    for a, b, value in intervals:
        value = value.strip()
        chars = [c for c in value if '\u3131' <= c <= '\u3163']
        # A mixed unknown/jamo interval never supplies a confirmed phone time.
        valid = len(value) == 1 and value in PHONES and a is not None and b is not None
        valid = valid and math.isfinite(a) and math.isfinite(b) and 0 <= a < b
        tokens.extend(chars)
        spans.extend([(a, b) if valid else None] * len(chars))
    return tokens, spans


def label_position(actual, i):
    if actual[i] in VOWELS: return '중성'
    if actual[i] in COMPOUNDS: return '겹받침 표기'
    if i+1 < len(actual) and actual[i+1] in VOWELS: return '초성 추정'
    return '종성 추정'


def make_case(meta, q, t, saved=None):
    canon, tag, spans = meta['_canon'], meta['_tag'], meta['_spans']
    if spans[t] is None: return None
    a, b = spans[t]
    frames = meta['_frames']; duration = frames/16000.
    if a >= duration or b > duration+.005: return None
    b = min(b, duration)
    # Preserve entire labelled phone and the beginning of the next vowel.
    # No time stretching, VAD re-trimming, or per-recording amplitude scaling.
    following = tag[t+1] if t+1 < len(tag) else '<끝>'
    right = min(duration, b+.2)
    if following in VOWELS and spans[t+1] is not None:
        right = min(duration, max(right, min(spans[t+1][1], spans[t+1][0]+.2)))
    left = max(0., a-.1)
    uid = meta['stem']
    source_position = label_position(tag, t)
    case = dict(case_id=hashlib.sha256((uid+':'+str(t)).encode()).hexdigest()[:12],
        uid=uid, speaker=meta['spk'], item=meta['item'], prompt=meta['prompt'],
        dataset=meta['ds'], language=meta['language'], proficiency=meta.get('proficiency'),
        audio_path=meta['_wav'], expected=canon[q] if q is not None else None,
        actual=tag[t], predicted=saved['predicted'] if saved is not None else None,
        warning=saved['warning'] if saved is not None else None, has_prediction=saved is not None,
        position=source_position, following=following,
        previous=tag[t-1] if t else '<시작>', tag_index=t, canonical_index=q,
        tag_start=a, tag_end=b, duration_ms=1000*(b-a), clip_start=left, clip_end=right,
        canonical_sequence=''.join(canon), tag_sequence=''.join(tag),
        audio_listened=False, independent_phonetic_judgment=None,
        example_source='primary_evaluation' if saved is not None else 'additional_test_audio_no_M3_prediction',
        source_part=meta['part'], annotation_has_dropped=bool(meta.get('dropped')))
    if saved is not None:
        case['saved_mapped_index'] = saved['mapped_index']
        case['runtime_canonical'] = saved['canonical']
        case['state'] = ('정상' if case['expected']==case['actual'] else '오발음')+' / '+(
            '전사 일치' if case['predicted']==case['actual'] else '전사 불일치')
    else: case['state'] = '추가 음성 · M3 미측정'
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, default=MDD_OUTPUTS/'analysis')
    ap.add_argument('--manifest', type=Path, default=MDD_OUTPUTS/'manifest_full_tag_replay10')
    ap.add_argument('--prior-report', type=Path, required=True, help='m3_weakness_review_2026-10-31_v1/phones/report.json (진단 산출물)')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    sources = {}
    prior_path=a.prior_report
    prior=json.loads(prior_path.read_text()); sources[str(prior_path)]=sha(prior_path)
    vocab_path=DATA_ROOT/'checkpoints/w2vbert_synth_tense_v2/step_3000/jamo_vocab.json'
    vocab=json.loads(vocab_path.read_text())['vocab']; sources[str(vocab_path)]=sha(vocab_path)
    rows={}; primary=[]; all_sites=[]; inventory=Counter()
    for p in sorted((a.analysis/'error_types_2026-10-19/ft100_ds50m').glob('sites.*.jsonl')):
        digest=sha(p)
        assert prior['sources_sha256'][str(p)]==digest, 'Saved source changed'
        sources[str(p)]=digest
        for number, line in enumerate(p.open(), 1):
            r=json.loads(line); uid=r['stem']; assert uid not in rows
            saved=r['sites']['new']
            accepted=bool(saved) and r['voiced_sec']>=min(1.,max(.2,.08*sum(s[0] in VOWELS for s in saved))) and r['voiced_ratio']>=.3
            rows[uid]=dict(record=r,accepted=accepted)
            inventory['recordings']+=1
            inventory['accepted_recordings' if accepted else 'rejected_recordings']+=1
            for i, s in enumerate(saved or []):
                L,A,C,H,shown,score,prob,pos,*_=s
                out=dict(uid=uid, speaker=uid.split('-')[0], item=r['item'], mapped_index=i,
                    expected=L, actual=A, canonical=C, predicted=H, warning=bool(shown),
                    score=score, prediction_probability=prob, saved_position=pos,
                    accepted=accepted, source_file=p.name, source_line=number)
                all_sites.append(out)
                if accepted: primary.append(out)
    metrics, expected, confusion=summarize(primary, vocab)
    all_metrics, _, _=summarize(all_sites, vocab)
    check=dict(sites=len(primary), normal=sum(s['expected']==s['actual'] for s in primary),
        errors=sum(s['expected']!=s['actual'] for s in primary),
        false_warnings=sum(s['warning'] and s['expected']==s['actual'] for s in primary),
        exact_warning=sum(s['warning'] and s['expected']!=s['actual'] and s['actual']==s['predicted'] for s in primary),
        annotated_actual_output=sum(s['expected']!=s['actual'] and s['actual']==s['predicted'] for s in primary))
    assert all(v==prior['counts'][k] for k,v in check.items()), check
    assert inventory['accepted_recordings']==3472 and inventory['recordings']==6720
    assert sum(m['n'] for m in metrics)+sum(s['actual'] is None for s in primary)==len(primary)
    assert sum(c['count'] for c in confusion)==sum(m['n'] for m in metrics)
    print('Saved M3 counts verified:',dict(inventory),check,flush=True)

    missing={m['phone'] for m in metrics if not m['n']}
    meta_by_uid={}; supplementary=defaultdict(list)
    paths=[a.manifest/('test.'+ext) for ext in ('tsv','ltr','canon','meta.jsonl')]
    # Streaming one pass over metadata; no whole-corpus waveform loading.
    with paths[0].open() as ft, paths[1].open() as fl, paths[2].open() as fc, paths[3].open() as fm:
        audio_root=Path(next(ft).strip())
        for n, values in enumerate(itertools.zip_longest(ft,fl,fc,fm),1):
            assert None not in values, 'Unequal manifest lengths'
            line,label,canonical,metadata=values
            rel,frames=line.rstrip().split('\t'); uid=Path(rel).stem
            tag=label.split(); can=canonical.split()
            if uid not in rows and not missing.intersection(tag): continue
            m=json.loads(metadata); assert m['stem']==uid
            parsed,spans=normalized_timing(m['tag_phones'])
            lab,_=normalized_timing(m['lab_phones'])
            assert tag==parsed and lab==can, uid
            m.update(_tag=tag,_canon=can,_spans=spans,_wav=str(audio_root/rel),_frames=int(frames),_row=n-1)
            if uid in rows: meta_by_uid[uid]=m
            if missing.intersection(tag):
                reverse=optimal_mapping(tuple(tag),tuple(can))
                for t,phone in enumerate(tag):
                    if phone not in missing: continue
                    choices=reverse[t];q=next(iter(choices)) if len(choices)==1 else None
                    case=make_case(m,q,t)
                    if case is None: continue
                    case['_rank']=hashlib.sha256(('atlas-v1:'+case['case_id']).encode()).hexdigest()
                    supplementary[phone].append(case)
                    supplementary[phone]=sorted(supplementary[phone],key=lambda c:c['_rank'])[:4]
    assert set(meta_by_uid)==set(rows)
    for p in paths: sources[str(p)]=sha(p)
    candidates=[]; timings=Counter(); fixed_maps={}; metadata_support=Counter()
    by_uid=defaultdict(list)
    for s in primary: by_uid[s['uid']].append(s)
    for uid,sites in by_uid.items():
        m=meta_by_uid[uid]; can,tag=m['_canon'],m['_tag']
        saved_sequence=tuple(s['expected'] for s in sites)
        mapping=optimal_mapping(saved_sequence,tuple(can))
        truth_map=optimal_mapping(tuple(can),tuple(tag))
        metadata_support[m['ds']]+=1
        for i,s in enumerate(sites):
            choices=mapping[i]
            if len(choices)!=1 or None in choices:
                timings['saved_to_LAB_ambiguous']+=1; continue
            q=next(iter(choices))
            if can[q]!=s['expected']:
                timings['saved_to_LAB_mismatch']+=1; continue
            ts=truth_map[q]
            if len(ts)!=1:
                timings['LAB_to_TAG_ambiguous']+=1; continue
            t=next(iter(ts))
            if t is None:
                timings['sequence_deletion_no_actual_phone']+=1; continue
            if tag[t]!=s['actual']:
                timings['saved_TAG_mapping_disagreement']+=1; continue
            c=make_case(m,q,t,s)
            if c is None:
                timings['no_valid_single_phone_time']+=1; continue
            c['_rank']=hashlib.sha256(('atlas-v1:'+c['case_id']).encode()).hexdigest()
            candidates.append(c);timings['visualizable_sites']+=1
    print('TAG timing mapped:',dict(timings),'supplementary phones:',sorted(supplementary),flush=True)

    examples={}; chosen_ids=defaultdict(list)
    for phone in PHONES:
        pool=sorted((c for c in candidates if c['actual']==phone),key=lambda c:c['_rank'])
        # First show one example from each available normal/error x match/mismatch
        # cell. The preview is illustrative, never the metric denominator.
        selected=[];used_uid=set();states=set()
        for c in pool:
            if c['state'] not in states and c['uid'] not in used_uid:
                selected.append(c);states.add(c['state']);used_uid.add(c['uid'])
        for c in pool:
            if len(selected)>=8:break
            if c['uid'] not in used_uid:selected.append(c);used_uid.add(c['uid'])
        if not selected:selected=supplementary.get(phone,[])
        for c in selected:examples[c['case_id']]=c;chosen_ids[phone].append(c['case_id'])

    groups={}
    for group in GROUPS:
        pool=[c for c in candidates if c['actual'] in group and c['expected']==c['actual']]
        contexts=defaultdict(lambda:defaultdict(list))
        for c in pool:contexts[(c['position'],c['following'])][c['actual']].append(c)
        if not contexts:groups[group]=dict(case_ids=[],description='비교 자료 없음');continue
        key=max(sorted(contexts),key=lambda k:(len(contexts[k]),min(map(len,contexts[k].values())),sum(map(len,contexts[k].values()))))
        ctx=contexts[key];common_speakers=set.intersection(*[{c['speaker'] for c in ctx[p]} for p in group if p in ctx])
        speaker=min(common_speakers) if common_speakers else None
        selected=[]
        for phone in group:
            pool=ctx.get(phone,[])
            if speaker:pool=[c for c in pool if c['speaker']==speaker]
            if pool:selected.append(min(pool,key=lambda c:c['_rank']))
        for c in selected:examples[c['case_id']]=c
        groups[group]=dict(case_ids=[c['case_id'] for c in selected],following=key[1],position=key[0],
            matched_speaker=speaker, missing_phones=[p for p in group if p not in ctx],
            description='%s, 바로 다음 TAG 음 %s를 맞춘 정상 주석 사례. %s 문항·앞 음소·녹음 환경까지 같은 대조 실험은 아닙니다.' %
                (key[0],key[1],('화자도 동일합니다.' if speaker else '화자는 서로 다릅니다.')))

    # Check actual waveform availability and original annotation provenance for
    # the selected display examples, without copying complete recordings.
    bases={p.name[:3]:p for p in (DATA_ROOT/'aihub').glob('04[0-3].*')}
    checked_audio={};checked_labels={}
    sys.path.insert(0,str(ROOT/'tasks/pronunciation-assessment/scripts/data'))
    from build_foreign_clean_v2 import parse_one
    for c in examples.values():
        p=Path(c['audio_path'])
        if str(p) not in checked_audio:
            with wave.open(str(p),'rb') as f:
                assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2)
                assert c['clip_end']<=f.getnframes()/16000.+1e-8
            checked_audio[str(p)]=sha(p)
        c['audio_sha256']=checked_audio[str(p)]
        if c['uid'] not in checked_labels:
            base=bases[c['dataset']]/'01-1.정식개방데이터'/('Training' if c['source_part']=='TL' else 'Validation')/'02.라벨링데이터'
            lab=base/(c['source_part']+'_pronunciation_lab')/(c['uid']+'.json')
            tag_path=base/(c['source_part']+'_pronunciation_tag')/(c['uid']+'.TextGrid')
            parsed=parse_one((c['uid'],c['dataset'],c['source_part'],str(tag_path),str(lab)))
            assert ''.join(parsed['canon'])==c['canonical_sequence'] and ''.join(parsed['ltr'])==c['tag_sequence']
            parsed_tokens,spans=normalized_timing(parsed['tag_phones'])
            checked_labels[c['uid']]=dict(lab=str(lab),tag=str(tag_path),lab_sha256=sha(lab),tag_sha256=sha(tag_path),spans=spans)
        raw=checked_labels[c['uid']]
        span=raw['spans'][c['tag_index']]
        assert span is not None and abs(span[0]-c['tag_start'])<1e-9 and abs(span[1]-c['tag_end'])<=.005
        c['source_lab']=raw['lab'];c['source_tag']=raw['tag']
    for r in checked_labels.values():r.pop('spans')

    # Membership means listed in a manifest, not observed at each optimizer step.
    train_membership={}; test_speakers={m['spk'] for m in meta_by_uid.values()}
    for name,p in [('M2',a.manifest/'train.tsv'),('M3',MDD_OUTPUTS/'manifest_synth_tense_v2/train.tsv')]:
        uids=set();speakers=set();items=set()
        with p.open() as f:
            next(f)
            for line in f:
                stem=Path(line.split('\t')[0]).stem;bits=stem.split('-')
                if stem in rows:uids.add(stem)
                if bits[0] in test_speakers:speakers.add(bits[0])
                if len(bits)>5 and bits[5].startswith(('RW','RS','RP')):items.add(bits[5])
        train_membership[name]=dict(recording_overlap=len(uids),speaker_id_overlap=len(speakers),
            primary_items_in_train=len({s['item'] for s in primary}&items),manifest=str(p))
        sources[str(p)]=sha(p)
    assert not train_membership['M2']['recording_overlap'] and not train_membership['M3']['recording_overlap']
    deletion=[s for s in primary if s['actual'] is None]
    report=dict(version='M3_phone_atlas_v1', model='w2vbert_synth_tense_v2/step_3000',
        model_remote_last_confirmed='2026-10-25', saved_predictions_date='2026-10-22',
        analysis_date='2026-11-01', model_inference_runs=0, training_updates=0,
        inventory=dict(inventory), counts=check, dataset_recordings=dict(metadata_support),
        primary_items=len({s['item'] for s in primary}),primary_speakers=len({s['speaker'] for s in primary}),
        modern_jamo_count=len(PHONES),model_jamo_count=len(vocab),
        evaluated_actual_phones=sum(m['n']>0 for m in metrics),
        missing_evaluation_phones=[m['phone'] for m in metrics if not m['n']],
        outside_model_vocab=[p for p in PHONES if p not in vocab],
        actual_phone_metrics=metrics, expected_phone_metrics=expected,
        deletion=dict(n=len(deletion),correct=sum(s['predicted'] is None for s in deletion),
                      exact_warnings=sum(s['predicted'] is None and s['warning'] for s in deletion)),
        timing=dict(timings), example_counts=dict(Counter(c['actual'] for c in examples.values())),
        membership=train_membership,sources_sha256=sources,verified_annotations=checked_labels,
        selection='SHA256 within each normal/error x transcription-match/mismatch cell; up to 8 per actual phone, diverse UIDs. Group comparisons match following TAG phone and position, same speaker when possible. Extra test audio supplies illustrations only for missing evaluation phones.',
        metric='P(saved actual_jamo == TAG | TAG=phone, saved canonical-mapped site, accepted input). Not 1-PER, not precision, not warning recall.',
        limitations=['Known M3 training prompts, test recordings; not unseen-sentence generalization or all 1.48M training recordings.',
            'Input-rejected recordings and unmapped canonical positions are outside primary rates; selection inventory and ungated table retained.',
            'Saved single edit path; alignment ambiguity can change counts. No full transcript/gap insertion accuracy is claimed.',
            'TAG and LAB are existing labels, not independently listened/adjudicated. No human acoustic boundary truth.',
            'No sample or no vocabulary entry does not mean zero acoustic ability. Compound jamo are annotation symbols, not necessarily single surface sounds.',
            'API warning absence is not a correctness guarantee. Amplitude depends on speaker/microphone; STFT is not calibrated sound pressure.',
            'No real-world population confidence interval from these selected, clustered examples.'],
        waveform=dict(sample_rate=16000,amplitude='original PCM16 /32768; common limits within figure',
            window_ms=25,hop_ms=5,nfft=512,spectrum_db_limits=[-100,0],
            spectrum_reference='20log10(abs(scipy STFT)), fixed 1.0 PCM reference, not dB SPL',
            crops='100ms before TAG start, complete TAG interval, >=200ms following context clipped to audio length'))
    for c in examples.values():c.pop('_rank',None)
    # Keep table selection order first; appended matched examples remain accessible.
    ordered=[];used=set()
    for phone in PHONES:
        for k in chosen_ids[phone]:
            if k not in used:ordered.append(examples[k]);used.add(k)
    ordered.extend(c for k,c in examples.items() if k not in used)
    write_json(a.out/'report.json',report)
    write_json(a.out/'examples.json',ordered)
    write_json(a.out/'comparison_groups.json',groups)
    write_csv(a.out/'per_phone_actual.csv',metrics)
    write_csv(a.out/'per_phone_expected.csv',expected)
    write_csv(a.out/'per_phone_all_inputs.csv',all_metrics)
    write_csv(a.out/'confusion.csv',confusion)
    with gzip.open(str(a.out/'sites.jsonl.gz'),'wt',encoding='utf-8') as f:
        for s in all_sites:f.write(json.dumps(s,ensure_ascii=False)+'\n')
    verification=dict(counts_match_prior=True,all_modern_jamo_rows=len(metrics),
        metrics_partition_complete=True,confusion_row_sums_match=True,
        selected_examples=len(examples),checked_audio=len(checked_audio),checked_original_label_pairs=len(checked_labels),
        source_code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'tasks/pronunciation-assessment/src/mdd/phone_atlas.py']},
        outputs_sha256={p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    write_json(a.out/'verification.json',verification)
    print(json.dumps({k:v for k,v in report.items() if k in ('inventory','counts','evaluated_actual_phones','missing_evaluation_phones','timing','membership','example_counts')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

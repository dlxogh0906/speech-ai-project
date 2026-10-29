"""Unscored RP source, exposure, annotation and audio-only segmentation audit."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
    str(ROOT / 'tasks/pronunciation-assessment/scripts/data')]
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import re
import time
import wave
import numpy as np
from build_foreign_clean_v2 import parse_one, split_of
from mdd.h0_evaluation_readiness import (normal_text, audio_windows, check_intervals,
                                        window_annotation)
from mdd.h0_phone_audit import project_gaps


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(str(tmp), str(path))


def emit(f, value):
    f.write(json.dumps(value, ensure_ascii=False) + '\n')


def paths(r, bases):
    base = Path(bases[r['ds']]) / '01-1.정식개방데이터' / (
        'Training' if r['part'] == 'TL' else 'Validation') / '02.라벨링데이터'
    return (base / (r['part'] + '_pronunciation_tag') / (r['stem'] + '.TextGrid'),
            base / (r['part'] + '_pronunciation_lab') / (r['stem'] + '.json'))


def audit_record(job):
    r, config = job
    tag, lab = paths(r, config['bases'])
    result = dict(uid=r['stem'], item=r['item'], speaker=r['spk'], source_dataset=r['ds'],
                  source_part=r['part'], historical_role=split_of(r['spk']),
                  audio_listened=False, independent_judgment=None)
    try:
        parsed = parse_one((r['stem'], r['ds'], r['part'], str(tag), str(lab)))
        fields = ('canon', 'ltr', 'tag_phones', 'lab_phones', 'tag_words', 'dropped', 'prompt')
        mismatch = [k for k in fields if parsed.get(k) != r.get(k)]
        raw = json.loads(lab.read_text(encoding='utf8'))
        user = raw.get('UserID')
        result.update(raw_matches_cache=not mismatch, mismatch_fields=mismatch,
            source_hashes={str(tag): sha(tag), str(lab): sha(lab)},
            user_id_sha256=hashlib.sha256(str(user).encode()).hexdigest() if user is not None else None,
            speaker_metadata_sha256=hashlib.sha256(json.dumps(raw.get('SpeakerMetadata'),
                ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            tag_phone_intervals=len(parsed.get('tag_phones', [])),
            lab_phone_intervals=len(parsed.get('lab_phones', [])),
            tag_word_intervals=len(parsed.get('tag_words', [])),
            lab_word_intervals=len(raw.get('RecordingMetadata', {}).get('phonemic', {}).get('words', [])))
        wav = Path(config['wav']) / r['stem'][:2] / (r['stem'] + '.wav')
        with wave.open(str(wav), 'rb') as w:
            sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            pcm = w.readframes(n)
        result.update(path=str(wav), sample_rate=sr, channels=ch, sample_width=width,
                      frames=n, seconds=n / sr, pcm_sha256=hashlib.sha256(pcm).hexdigest())
        if (sr, ch, width) != (16000, 1, 2):
            result['failure'] = 'unsupported_audio_format'
            return result, [], {}
        audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.
        result.update(rms=float(np.sqrt(np.mean(audio ** 2))) if n else 0.,
                      clipped_samples=int((np.abs(audio) >= 32767 / 32768).sum()))
        vocab = set(config['vocabulary'])
        result['tag_check'] = check_intervals(r['tag_phones'], n / sr, vocab)
        result['lab_check'] = check_intervals(r['lab_phones'], n / sr, vocab, True)
        # Preserve missing/uncertain labels. Dropped strings are never projected as normal.
        chunks, coverage = [], {}
        for mode in ('fixed', 'energy'):
            cc = collections.Counter()
            for k, (start, end) in enumerate(audio_windows(audio, sr, mode)):
                t = window_annotation(r['tag_phones'], start / sr, end / sr, vocab)
                c = window_annotation(r['lab_phones'], start / sr, end / sr, vocab, True)
                complete = bool(t['complete'] and c['complete'] and c['tokens'] and
                                not result['tag_check']['issues'] and not result['lab_check']['issues']
                                and not mismatch)
                entry = dict(uid=r['stem'], item=r['item'], speaker=r['spk'],
                    historical_role=result['historical_role'], mode=mode, chunk=k,
                    start_sample=start, end_sample=end, seconds=(end-start)/sr,
                    tag=t, lab=c, complete_annotation_window=complete,
                    eligible_for_primary_evaluation=False, model_scored=False)
                chunks.append(entry)
                cc['chunks'] += 1; cc['samples'] += end-start
                cc['under_300ms'] += int(end-start < .3*sr)
                cc['tag_boundary_cut_chunks'] += int(bool(t['boundary_crossings']))
                cc['lab_boundary_cut_chunks'] += int(bool(c['boundary_crossings']))
                cc['unknown_tag_chunks'] += int(bool(t['interval_counts'].get('unknown')))
                cc['unknown_lab_chunks'] += int(bool(c['interval_counts'].get('unknown')))
                cc['no_reference_tokens'] += int(not c['tokens'])
                if complete:
                    cc['complete_annotation_chunks'] += 1
                    cc['complete_annotation_samples'] += end-start
                    truth, gaps = project_gaps(c['tokens'], t['tokens'])
                    entry['conditional_annotation_counts'] = dict(
                        normal=sum(x==y for x,y in zip(c['tokens'],truth)),
                        errors=sum(x!=y for x,y in zip(c['tokens'],truth)),
                        deletion=sum(y is None for y in truth),
                        insertion_gaps=sum(bool(g) for g in gaps))
                    entry['conditional_normal_phone_counts'] = dict(collections.Counter(
                        x for x,y in zip(c['tokens'],truth) if x==y))
                    entry['conditional_error_phone_counts'] = dict(collections.Counter(
                        x for x,y in zip(c['tokens'],truth) if x!=y))
                    entry['conditional_substitutions'] = dict(collections.Counter(
                        x+'>'+y for x,y in zip(c['tokens'],truth) if y is not None and x!=y))
            if cc['samples'] != n:
                raise AssertionError('audio windows lost or duplicated samples')
            coverage[mode] = dict(cc)
        result['window_coverage'] = coverage
        return result, chunks, {}
    except Exception as e:
        result['failure'] = type(e).__name__ + ': ' + str(e)
        return result, [], {}


def alias_record(job):
    r, bases = job
    _, lab = paths(r, bases)
    try:
        raw = json.loads(lab.read_text(encoding='utf8')); user = raw.get('UserID')
        return dict(speaker=r['spk'], uid=r['stem'], source=str(lab), sha256=sha(lab),
            user_id_sha256=hashlib.sha256(str(user).encode()).hexdigest() if user is not None else None)
    except Exception as e:
        return dict(speaker=r['spk'], uid=r['stem'], failure=type(e).__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p4', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--m2', required=True); ap.add_argument('--m3', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    p4 = Path(a.p4); p = json.loads((p4/'protocol.json').read_text())
    manifest = Path(p['manifest']); stats = json.loads((manifest/'stats.json').read_text())['args']
    cache = Path(stats['cache']); native = Path(stats['native'])
    vocabulary = [s.split()[0] for s in (manifest/'dict.ltr.txt').read_text().splitlines()
                  if len(s.split()[0]) == 1 and 'ㄱ' <= s.split()[0] <= 'ㅣ']
    bases = {d.name[:3]: str(d) for d in Path(stats['aihub_root']).glob('04[0-3].*')}
    config = dict(bases=bases, wav=stats['wav16k'], vocabulary=vocabulary)
    protocol = dict(version='RP_readiness_v1', created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        p4=str(p4), manifest=str(manifest), m2=a.m2, m3=a.m3, cache=str(cache),
        items=['RP002','RP003','RP004'], candidates='all source records; no label/score selection',
        speaker_priority='historical test hash, audited against actual training and checkpoint-selection speakers',
        windows='12 s fixed and audio-only quiet-run cuts within +/-2 s; all samples and tails retained',
        window_label_policy='attach unchanged original labels only after cutting; unknown/cut intervals retained',
        conditional_counts='complete-annotation windows are descriptive only; NOT a selected evaluation set',
        no_model_scores=True, optimizer_updates=0, independent_listening=False,
        gates=['no train or checkpoint-selection speaker/UserID overlap',
               'no whole PCM duplicate of indexed train/selection assets',
               'reference and actual labels support declared denominators without unknown deletion',
               'input segmentation and partial-label scoring specified before inference',
               'three prompt families cannot replace broad new-text confirmation'],
        bindings={str(f):sha(f) for f in (cache, p4/'raw_source_candidate_inventory.json',
                  manifest/'stats.json', Path(__file__), ROOT/'tasks/pronunciation-assessment/src/mdd/h0_evaluation_readiness.py')})
    write(out/'protocol.json', protocol)
    sets, reference_uids, frame_candidates = {}, {}, {}
    groups = [('H0', manifest, ('train','valid','valid_unseen','test','test_unseen')),
              ('M2', Path(a.m2), ('train','valid','test')), ('M3', Path(a.m3), ('train','valid'))]
    for name, folder, splits in groups:
        for split in splits:
            f = folder/(split+'.tsv')
            if not f.exists(): continue
            key = name+'_'+split; speakers, uids, items = set(), set(), set(); n=0; foreign=0
            with f.open() as stream:
                base = stream.readline().strip()
                for line in stream:
                    rel, frames = line.rstrip().split('\t'); stem=Path(rel).stem; n+=1
                    parts=stem.split('-')
                    if len(parts)==7:
                        speakers.add(parts[0]); items.add(parts[5]); foreign+=1
                        reference_uids.setdefault(parts[0], stem)
                        if parts[5] in protocol['items']: uids.add(stem)
                    # Equal length is a necessary condition for exact full PCM duplication.
                    # Only retain long rows, plus short RP paths themselves; fill exact set after header audit.
                    if split in ('train','valid') and int(frames)>=20*16000:
                        frame_candidates.setdefault(int(frames), set()).add(os.path.join(base,rel))
            sets[key]=dict(speakers=speakers, items=items, candidate_uids=uids, rows=n, foreign_rows=foreign,
                           path=str(f), sha256=sha(f))
    needed = set(reference_uids.values()); representatives={}; candidates=[]; prompts={}; reference_prompts={}
    write(out/'live_status.json',dict(stage='reading_source_cache',model_inferences=0))
    with gzip.open(cache,'rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r.get('fail'):continue
            if r['stem'] in needed:representatives[r['spk']]=r
            if r.get('item') in protocol['items']:
                candidates.append(r);prompts.setdefault(r['item'],set()).add(r.get('prompt') or '')
            elif r.get('item'):
                reference_prompts.setdefault(r['item'],set()).add(r.get('prompt') or '')
    if len({r['stem'] for r in candidates})!=len(candidates):raise ValueError('duplicate source UID')
    with (out/'source_records.jsonl.gz').open('wb') as raw:
        with gzip.open(raw,'wt',encoding='utf8') as f:
            for r in candidates:emit(f,r)
    write(out/'live_status.json',dict(stage='original_annotations_and_audio',records=len(candidates),processed=0))
    results=[]; chunks_counter=collections.defaultdict(collections.Counter); conditional=collections.defaultdict(collections.Counter)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool, \
            (out/'recordings.jsonl').open('w') as fr, (out/'windows.jsonl').open('w') as fw:
        for i,(r, chunks, _) in enumerate(pool.map(audit_record, ((r,config) for r in candidates),chunksize=16)):
            results.append(r);emit(fr,r)
            for c in chunks:
                emit(fw,c)
                for role in ('all',c['historical_role']):
                    key=(c['item'],role,c['mode'])
                    chunks_counter[key].update({'chunks':1,'samples':c['end_sample']-c['start_sample'],
                        'complete_annotation_chunks':int(c['complete_annotation_window']),
                        'unknown_tag_chunks':int(bool(c['tag']['interval_counts'].get('unknown'))),
                        'unknown_lab_chunks':int(bool(c['lab']['interval_counts'].get('unknown'))),
                        'tag_boundary_cut_chunks':int(bool(c['tag']['boundary_crossings'])),
                        'lab_boundary_cut_chunks':int(bool(c['lab']['boundary_crossings']))})
                    conditional[key].update(c.get('conditional_annotation_counts',{}))
                    conditional[key].update({'normal:'+k:v for k,v in c.get('conditional_normal_phone_counts',{}).items()})
                    conditional[key].update({'error:'+k:v for k,v in c.get('conditional_error_phone_counts',{}).items()})
            if (i+1)%250==0:
                write(out/'live_status.json',dict(stage='original_annotations_and_audio',processed=i+1,records=len(candidates)))
                print('audited',i+1,flush=True)
    write(out/'live_status.json',dict(stage='speaker_alias_metadata',speakers=len(representatives)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        aliases=list(pool.map(alias_record, ((r,bases) for r in representatives.values()),chunksize=32))
    with (out/'reference_speaker_aliases.jsonl').open('w') as f:
        for r in aliases:emit(f,r)
    ref_alias={r['speaker']:r.get('user_id_sha256') for r in aliases}
    memberships=[]
    for key,s in sets.items():
        users={ref_alias.get(x) for x in s['speakers']}-{None}
        for item in protocol['items']:
            for role in ('all','test','valid','train'):
                rr=[r for r in results if r['item']==item and (role=='all' or r['historical_role']==role)]
                memberships.append(dict(source=key,item=item,role=role,records=len(rr),
                    uid_overlap=sum(r['uid'] in s['candidate_uids'] for r in rr),
                    speaker_overlap=sum(r['speaker'] in s['speakers'] for r in rr),
                    user_id_overlap=sum(r.get('user_id_sha256') in users for r in rr),
                    reference_speakers=len(s['speakers']),reference_alias_missing=sum(not ref_alias.get(x) for x in s['speakers'])))
    write(out/'memberships.json',memberships)
    write(out/'manifest_bindings.json',{k:{x:(sorted(v) if isinstance(v,set) else v) for x,v in s.items()
        if x not in ('speakers','candidate_uids')} for k,s in sets.items()})
    # Whole-audio duplicate check, exhaustive within candidate records and equal-length indexed long assets.
    pcm=collections.defaultdict(list)
    for r in results:
        if r.get('pcm_sha256'):pcm[r['pcm_sha256']].append(r['uid'])
    dup=[uids for uids in pcm.values() if len(uids)>1];candidate_frames={r.get('frames') for r in results}
    check_paths=set().union(*(v for k,v in frame_candidates.items() if k in candidate_frames)) if frame_candidates else set()
    short_frames={n for n in candidate_frames if n is not None and n<20*16000}
    if short_frames:
        for key,s in sets.items():
            if not (key.endswith('_train') or key.endswith('_valid')):continue
            with Path(s['path']).open() as f:
                base=f.readline().strip()
                for line in f:
                    rel,n=line.rstrip().split('\t')
                    if int(n) in short_frames:check_paths.add(os.path.join(base,rel))
    matches=[];failed=[]
    for path in sorted(check_paths):
        try:
            with wave.open(path,'rb') as w:
                sig=(w.getframerate(),w.getnchannels(),w.getsampwidth());data=w.readframes(w.getnframes())
            digest=hashlib.sha256(data).hexdigest()
            if sig==(16000,1,2) and digest in pcm:matches.append(dict(path=path,candidates=pcm[digest]))
        except Exception as e:failed.append(dict(path=path,error=type(e).__name__))
    write(out/'pcm_duplicates.json',dict(candidate_duplicates=dup,indexed_equal_length_paths_checked=len(check_paths),
        matches=matches,failed=failed,short_candidate_records_unchecked=0,
        scope='exact whole 16k PCM only; no near-duplicates, different crops, or unindexed assets'))
    text_audit(out,prompts,reference_prompts,native,manifest,sets)
    summaries=[]
    for item in protocol['items']:
        for role in ('all','test','valid','train'):
            rr=[r for r in results if r['item']==item and (role=='all' or r['historical_role']==role)]
            durations=[r['seconds'] for r in rr if 'seconds' in r]
            summaries.append(dict(item=item,role=role,recordings=len(rr),speakers=len({r['speaker'] for r in rr}),
                source_failures=sum('failure' in r for r in rr),
                raw_cache_mismatches=sum(not r.get('raw_matches_cache') for r in rr),
                tag_unknown_records=sum(bool(r.get('tag_check',{}).get('kinds',{}).get('unknown')) for r in rr),
                lab_unknown_records=sum(bool(r.get('lab_check',{}).get('kinds',{}).get('unknown')) for r in rr),
                tag_time_issue_records=sum(bool(r.get('tag_check',{}).get('issues')) for r in rr),
                lab_time_issue_records=sum(bool(r.get('lab_check',{}).get('issues')) for r in rr),
                duration_quantiles=np.quantile(durations,[0,.25,.5,.75,1]).tolist() if durations else None,
                hours=sum(durations)/3600,over_20_seconds=sum(t>20 for t in durations)))
    write(out/'report.json',dict(status='source_audit_complete_no_evaluation_or_training',summaries=summaries,
        windows=[dict(item=k[0],role=k[1],mode=k[2],**dict(v),conditional_counts=dict(conditional[k])) for k,v in sorted(chunks_counter.items())],
        vocabulary=vocabulary,source_recordings=len(results),raw_files_checked=2*len(results),
        alias_records_checked=len(aliases),alias_failures=sum('failure' in r for r in aliases),
        independent_script_families=3,new_recordings=0,new_model_inferences=0,optimizer_updates=0,
        evaluation_frozen=False,reason='readiness verdict and scoring protocol require explicit audit closure',
        limitations=['original annotation agreement is provenance, not independent listening',
            'UserID equality cannot identify aliases with different IDs',
            'conditional complete-window counts are label-conditioned and cannot replace the whole cohort',
            'all windows retained; no audio copied, no labels repaired, no model output inspected',
            'three scripts and public SSL exposure uncertainty preclude a broad independent final-test claim']))
    write(out/'finished.json',dict(status='source_audit_completed',model_inferences=0,optimizer_updates=0))
    write(out/'live_status.json',dict(stage='source_audit_completed',processed=len(results)))


def text_audit(out,prompts,reference_prompts,native,manifest,sets):
    queries=[]
    for item,variants in sorted(prompts.items()):
        for v,prompt in enumerate(sorted(variants)):
            for kind,parts in [('script',[prompt]),('sentence',re.split(r'[.!?]+',prompt))]:
                for k,t in enumerate(parts):
                    norm=normal_text(t)
                    if not norm:continue
                    queries.append(dict(item=item,variant=v,kind=kind,part=k,text=t.strip(),normal=norm,
                        foreign_exact=[],foreign_contains=[],foreign_contained=[],native_exact=0,native_contains=0,
                        replay_exact=0,replay_contains=0,native_examples=[]))
    for q in queries:
        for item,variants in reference_prompts.items():
            for v in variants:
                text=normal_text(v)
                if not text:continue
                if text==q['normal']:q['foreign_exact'].append(item)
                if q['normal'] in text:q['foreign_contains'].append(item)
                if text in q['normal']:q['foreign_contained'].append(item)
        for field in ('foreign_exact','foreign_contains','foreign_contained'):q[field]=sorted(set(q[field]))
        q['manifest_item_overlaps']={key:{field:sorted(set(q[field])&s['items'])
            for field in ('foreign_exact','foreign_contains','foreign_contained')}
            for key,s in sets.items() if key.endswith('_train')}
    replay=set()
    with (manifest/'train.tsv').open() as f:
        next(f)
        for line in f:
            uid=Path(line.split('\t')[0]).stem
            if len(uid.split('-'))!=7:replay.add(uid)
    # Long shared substrings reveal partial overlap without asserting semantic identity.
    kmer_items=collections.defaultdict(set)
    for q in queries:
        if q['kind']=='script':
            for i in range(max(0,len(q['normal'])-11)):kmer_items[q['normal'][i:i+12]].add(q['item'])
    partial=collections.Counter();examples=collections.defaultdict(list);n=0;kmer_set=set(kmer_items)
    for line in (native/'train.meta.jsonl').open():
        r=json.loads(line);n+=1;text=r['norm_text'];uid=Path(r['rel']).stem;rep=uid in replay
        for q in queries:
            exact=text==q['normal'];contains=q['normal'] in text
            q['native_exact']+=int(exact);q['native_contains']+=int(contains)
            q['replay_exact']+=int(rep and exact);q['replay_contains']+=int(rep and contains)
            if contains and len(q['native_examples'])<3:q['native_examples'].append(uid)
        shared={text[i:i+12] for i in range(max(0,len(text)-11))}&kmer_set
        items={item for k in shared for item in kmer_items[k]}
        for item in items:
            partial[item]+=1
            if len(examples[item])<10:examples[item].append(dict(uid=uid,shared_12_chars=sorted(k for k in shared if item in kmer_items[k]),replay=rep))
    write(out/'text_overlap.json',dict(queries=queries,native_manifest_rows=n,
        native_shared_12_character_records=dict(partial),native_partial_examples=dict(examples),
        bindings={str(native/'train.meta.jsonl'):sha(native/'train.meta.jsonl')},
        scope='normalized orthographic membership, not proof of actual optimizer or public SSL exposure; foreign query includes all cache item variants'))


if __name__=='__main__':
    main()

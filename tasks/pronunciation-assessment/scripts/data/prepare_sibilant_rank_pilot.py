"""Prepare an isolated actual-vs-counterfactual CTC pilot from official splits.

No model inference, synthetic audio, or analyzed evaluation feature caches.
Only unambiguous single ㅅ↔ㅆ substitutions and wholly correct utterances.
The pre-existing encoder has seen training prompts; additional pilot splits
are speaker-, item-, normalized-prompt-, and waveform-disjoint.
"""
import sys as _sys, pathlib as _pl
_root = _pl.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root (pyproject.toml) not found")
    _root = _root.parent
_sys.path[:0] = [str(_root), *(str(p) for p in sorted(_root.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, gzip, hashlib, heapq, json, wave
from pathlib import Path

VOWELS=set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
SS={'ㅅ','ㅆ'}
def priority(text):return int(hashlib.sha256(('ss-rank-sample-v1:'+text).encode()).hexdigest(),16)
def item_bucket(item):return int(hashlib.sha256(('ssrank-v1:'+item).encode()).hexdigest()[:8],16)%5
def destination(item):
    b=item_bucket(item)
    return 'train' if b in (0,1) else 'cal' if b in (2,3) else 'test'
def add_pool(pool,key,row,limit=256):
    h=priority(row['uid']);bucket=pool[key];entry=(-h,row['uid'],row)
    if len(bucket)<limit:heapq.heappush(bucket,entry)
    elif h < -bucket[0][0]:heapq.heapreplace(bucket,entry)
def pooled(pool):return [e[2] for bucket in pool.values() for e in bucket]
def balanced(rows,n,used=None):
    used=set() if used is None else used
    groups=collections.defaultdict(list)
    for r in rows:
        if r['uid'] not in used:groups[r['item']].append(r)
    for k in groups:groups[k].sort(key=lambda r:priority(r['uid']))
    keys=sorted(groups,key=priority);out=[];pos=collections.Counter()
    while len(out)<n:
        changed=False
        for k in keys:
            while pos[k]<len(groups[k]) and groups[k][pos[k]]['uid'] in used:pos[k]+=1
            if pos[k]<len(groups[k]):
                row=groups[k][pos[k]];pos[k]+=1;out.append(row);used.add(row['uid']);changed=True
                if len(out)==n:break
        if not changed:break
    return out
def classification(actual,canonical,vocab):
    if not actual or len(actual)!=len(canonical):return None,'sequence_length'
    if any(t not in vocab for t in actual+canonical):return None,'oov'
    qs=[q for q in range(len(canonical)-1) if canonical[q] in SS and canonical[q+1] in VOWELS]
    diffs=[q for q,(a,c) in enumerate(zip(actual,canonical)) if a!=c]
    if not diffs:return ('normal' if qs else 'replay',qs),'ok'
    if len(diffs)!=1:return None,'multiple_differences'
    q=diffs[0]
    if q not in qs or actual[q] not in SS:return None,'not_single_ss_onset'
    return ('error',qs),'ok'
def record(meta,actual,canonical,frames,wav,source_split,manifest_index,vocab):
    assert Path(wav).stem == meta['stem'], ('metadata_row_mismatch', wav, meta['stem'])
    result,why=classification(actual,canonical,vocab)
    if result is None:return None,why
    if not 8000<=frames<=192000:return None,'duration'
    if frames//320 < len(actual)+sum(a==b for a,b in zip(actual,actual[1:])):return None,'ctc_length'
    if any(meta.get('dropped',{}).values()):return None,'dropped_annotation'
    role,qs=result
    r={'uid':meta['stem'],'wav':str(wav),'frames':frames,'speaker':meta['spk'],
       'prompt':meta['prompt'],'norm_prompt':meta['norm_prompt'],'item':meta['item'],
       'source_split':source_split,'actual':actual,'canonical':canonical,
       'sites':[{'q':q,'a':q,'canon':canonical[q],'actual':actual[q]} for q in qs],
       'role':role,'manifest_index':manifest_index,'language':meta.get('language'),
       'label_provenance':'human_TAG_actual__LAB_phonemic_canonical',
       'dataset':meta.get('ds'),'dataset_part':meta.get('part')}
    if role=='error':
        q=next(q for q in qs if canonical[q]!=actual[q]);r['direction']=canonical[q]+'→'+actual[q]
        # Times are for manual inspection only; do not crop training audio.
        tag=[x for x in meta.get('tag_phones',[]) if x[2] not in ('','.','-','spn','<unk>')]
        if len(tag)==len(actual) and all(len(x[2])==1 for x in tag):r['target_tag_seconds']=tag[q][:2]
    return r,'ok'
def stats(rows):
    return {'utterances':len(rows),'speakers':len({r['speaker'] for r in rows}),
            'items':len({r['item'] for r in rows}),'prompts':len({r['norm_prompt'] for r in rows}),
            'roles':dict(collections.Counter(r['role'] for r in rows)),
            'directions':dict(collections.Counter(r.get('direction') for r in rows if r['role']=='error')),
            'sites':dict(collections.Counter(s['canon']+'→'+s['actual'] for r in rows for s in r['sites'])),
            'seconds':round(sum(r['frames'] for r in rows)/16000,2),
            'vocab_size':len({t for r in rows for t in r['actual']})}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default=str(MDD_OUTPUTS/'manifest_full_tag_replay10'))
    ap.add_argument('--cache',default=str(MDD_OUTPUTS/'cache/foreign_tag_lab_parsed.jsonl.gz'))
    ap.add_argument('--vocab',default=str(DATA_ROOT/'checkpoints/w2vbert_synth_tense_v2/step_3000/jamo_vocab.json'))
    ap.add_argument('--out',default=str(MDD_OUTPUTS/'analysis/ss_rank_2026-10-23/data'))
    a=ap.parse_args();m=Path(a.manifest);out=Path(a.out)
    existing=[str(out/name) for name in ('train.jsonl','cal.jsonl','test.jsonl','summary.json') if (out/name).exists()]
    if existing:
        ap.error('Refusing to overwrite existing pilot artifacts: '+', '.join(existing))
    out.mkdir(parents=True,exist_ok=True)
    vocab=json.load(open(a.vocab))['vocab'];available={};pools={};excluded={}
    # Official valid/test supply actual/canonical/meta directly, not inference caches.
    for src,dst in [('valid','cal'),('test','test')]:
        pools[dst]=collections.defaultdict(list);cnt=collections.Counter();skip=collections.Counter()
        with open(m/(src+'.tsv')) as ft,open(m/(src+'.ltr')) as fa,open(m/(src+'.canon')) as fc,open(m/(src+'.meta.jsonl')) as fm:
            root=next(ft).strip()
            for idx,(t,aa,cc,mm) in enumerate(zip(ft,fa,fc,fm)):
                meta=json.loads(mm)
                if destination(meta['item'])!=dst:continue
                rel,fr=t.rstrip().split('\t');wav=Path(root)/rel
                r,why=record(meta,aa.split(),cc.split(),int(fr),wav,src,idx,vocab)
                if r is None:skip[why]+=1;continue
                cnt[r.get('direction',r['role'])]+=1
                add_pool(pools[dst],(r['role'],r.get('direction',''),r['item']),r)
        available[dst]=dict(cnt);excluded[dst]=dict(skip)
        print(dst,'available',json.dumps(available[dst],ensure_ascii=False),flush=True)
    # Index train membership and duration; repeated upsampling rows become one uid.
    train_index={}
    with open(m/'train.tsv') as f:
        root=next(f).strip()
        for idx,line in enumerate(f):
            rel,fr=line.rstrip().split('\t')
            if '/mdd_gold_wav16k/' not in '/'+rel:continue
            stem=Path(rel).stem
            if not 8000<=int(fr)<=192000:continue
            item=stem.split('-')[5] if len(stem.split('-'))>5 else ''
            if not item or destination(item)!='train':continue
            train_index.setdefault(stem,(int(fr),idx,str(Path(root)/rel)))
    print('official_train_unique_eligible_members',len(train_index),flush=True)
    pools['train']=collections.defaultdict(list);cnt=collections.Counter();skip=collections.Counter();seen=set()
    with gzip.open(a.cache,'rt',encoding='utf-8') as f:
        for i,line in enumerate(f):
            # Parse all records once; select only actual official train membership.
            meta=json.loads(line);st=meta['stem']
            if st not in train_index or st in seen:continue
            seen.add(st);fr,idx,wav=train_index[st]
            r,why=record(meta,meta['ltr'],meta['canon'],fr,wav,'train',idx,vocab)
            if r is None:skip[why]+=1;continue
            cnt[r.get('direction',r['role'])]+=1
            add_pool(pools['train'],(r['role'],r.get('direction',''),r['item']),r)
            if len(seen)%100000==0:print('train scanned',len(seen),'usable',json.dumps(dict(cnt),ensure_ascii=False),flush=True)
    available['train']=dict(cnt);excluded['train']=dict(skip)
    print('train available',json.dumps(available['train'],ensure_ascii=False),flush=True)
    result={}
    for dst in ('train','cal','test'):
        candidates=pooled(pools[dst]);used=set();rows=[]
        nerr=128 if dst=='train' else 64
        for direction in ('ㅅ→ㅆ','ㅆ→ㅅ'):
            rows+=balanced([r for r in candidates if r.get('direction')==direction],nerr,used)
        if dst=='train':
            quota=collections.Counter(r['item'] for r in rows)
            for item,n in sorted(quota.items()):
                normals=balanced([r for r in candidates if r['role']=='normal' and r['item']==item],n,used)
                rows+=normals
            # Broad correct replay prioritizes non-sibilant prompts, then any other normal.
            replay=balanced([r for r in candidates if r['role']=='replay'],256,used)
            if len(replay)<256:replay+=balanced([r for r in candidates if r['role']=='normal'],256-len(replay),used)
            for r in replay:r['role']='replay'
            rows+=replay
        else:
            rows+=balanced([r for r in candidates if r['role']=='normal'],512-len(rows),used)
        result[dst]=sorted(rows,key=lambda r:priority(r['uid']))
    # Enforce exact training-ltr agreement for every chosen training record.
    chosen={r['uid']:r for r in result['train']};verified=set()
    with open(m/'train.tsv') as ft,open(m/'train.ltr') as fl:
        next(ft)
        for line,label in zip(ft,fl):
            st=Path(line.split('\t')[0]).stem
            if st in chosen:
                assert label.split()==chosen[st]['actual'],('train_label_mismatch',st)
                verified.add(st)
    assert verified==set(chosen),('not_in_official_train',set(chosen)-verified)
    checks={}
    for dst,rows in result.items():
        assert len({r['uid'] for r in rows})==len(rows)
        for r in rows:
            with wave.open(r['wav'],'rb') as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==(16000,1,2,r['frames']),r['uid']
            assert all(t in vocab for t in r['actual']+r['canonical'])
            for s in r['sites']:
                assert r['actual'][s['a']]==s['actual'] and r['canonical'][s['q']]==s['canon']
            if r['role']=='error':
                sites=[s for s in r['sites'] if s['canon']!=s['actual']];assert len(sites)==1
                cf=list(r['actual']);cf[sites[0]['a']]=sites[0]['canon'];assert cf==r['canonical']
    for x,y in [('train','cal'),('train','test'),('cal','test')]:
        checks[x+'∩'+y]={}
        for key in ('uid','wav','speaker','item','norm_prompt'):
            shared={r[key] for r in result[x]}&{r[key] for r in result[y]}
            checks[x+'∩'+y][key]=len(shared);assert not shared,(x,y,key,shared)
    summary={'config':vars(a),'constraints':[
        'Training is intersected with official train.tsv and exact train.ltr; no analyzed evaluation cache is used.',
        'Calibration comes from official valid; evaluation from official test.',
        'sha256(ssrank-v1:item)[:8]%5: train buckets0,1; cal2,3; test4; all overlap checks mandatory.',
        'Only 0.5–12.0s mono PCM16/16kHz; no truncation; exact checkpoint vocabulary.',
        'Error utterance has same actual/canonical length and exactly one substitution, both ㅅ/ㅆ before vowel.',
        'Normal/replay utterance is globally actual==canonical; ambiguous edits, dropped labels and OOV excluded.',
        'Rows are unique waveform utterances; hash ordering and item round robin avoid frequency-only sampling.',
        'Existing pretrained base may have seen prompts; these guarantees apply to added pilot training only.',
        'Human TAG annotations are accepted as labels; this does not certify a new perceptual listening audit.'
    ],'available_after_filters':available,'excluded':excluded,'selected':{k:stats(v) for k,v in result.items()},'overlap_checks':checks,
      'training_ltr_verified':len(verified),'vocab':vocab}
    for dst,rows in result.items():
        with open(out/(dst+'.jsonl'),'x') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    json.dump(summary,open(out/'summary.json','x'),ensure_ascii=False,indent=2)
    print('COMPLETE',json.dumps(summary['selected'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()

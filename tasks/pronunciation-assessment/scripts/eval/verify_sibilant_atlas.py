"""Independently check SS coverage, saved outputs and selected acoustic values."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import gzip
import hashlib
import json
import wave
from collections import Counter
import numpy as np
from scipy.signal import periodogram
from mdd.phone_atlas import sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--atlas', type=Path, required=True)
    args = ap.parse_args(); folder = args.atlas / 'sibilant_context_v1'
    report = json.loads((folder/'report.json').read_text())
    cases = json.loads((folder/'cases.json').read_text())
    excluded = json.loads((folder/'excluded.json').read_text())
    with gzip.open(args.atlas/'sites.jsonl.gz', 'rt', encoding='utf-8') as f:
        targets = {(r['uid'],r['mapped_index']):r for r in map(json.loads,f)
                   if r['accepted'] and r['actual'] in ('ㅅ','ㅆ')}
    keys = {(r['uid'],r['saved_mapped_index']) for r in cases}
    missing = {(r['uid'],r['mapped_index']) for r in excluded}
    assert not keys & missing and keys | missing == set(targets)
    assert len(keys) == len(cases) and len(missing) == len(excluded)
    for c in cases:
        s = targets[c['uid'],c['saved_mapped_index']]
        assert all(c[k] == s[k] for k in ('expected','actual','predicted','warning'))
        assert abs(c['features']['tag_duration_ms'] - 1000*(c['tag_end']-c['tag_start'])) < 1e-8
        if c['high_run']:
            assert c['high_run'][0] >= c['tag_start']-1e-8 and c['high_run'][1] <= c['tag_end']+1e-8
        assert all(v is None or np.isfinite(v) for v in c['features'].values())
    lookup = {c['case_id']:c for c in cases}; paired_keys = set()
    for pair in report['pairs']:
        a,b = lookup[pair['s']],lookup[pair['ss']]
        assert (a['actual'],b['actual']) == ('ㅅ','ㅆ')
        assert all(a[k] == b[k] for k in ('speaker','previous','following','position','word_position'))
        key = (a['speaker'],*pair['context']); assert key not in paired_keys; paired_keys.add(key)
    for r in report['contexts']:
        rows = [c for c in cases if [c[k] for k in ('previous','following','position','word_position')] == r['context']]
        assert [sum(c['actual']==p for c in rows) for p in ('ㅅ','ㅆ')] == [r['s'],r['ss']]
    # A distinct SciPy spectral calculation and direct lag dot-products, on
    # hash-selected audio rather than examples selected for the figures.
    sample = sorted(cases,key=lambda c: hashlib.sha256(('ss-verify:'+c['case_id']).encode()).hexdigest())[:40]
    checked = Counter()
    for c in sample:
        with wave.open(c['audio_path'],'rb') as f:
            assert (f.getframerate(),f.getsampwidth(),f.getnchannels()) == (16000,2,1)
            sr=16000; a=round(c['clip_start']*sr); b=round(c['clip_end']*sr)
            f.setpos(a); x=np.frombuffer(f.readframes(b-a),'<i2').astype(float)/32768
        offset=a/sr; start=c['tag_start']-offset; end=c['tag_end']-offset
        mid=(start+.25*(end-start),start+.75*(end-start)); values=[]
        for i in range(0,len(x)-160+1,32):
            if i/sr < mid[0]-1e-8 or (i+160)/sr > mid[1]+1e-8: continue
            y=x[i:i+160]-x[i:i+160].mean()
            if np.mean(y*y)<1e-6: continue
            f,power=periodogram(y,fs=sr,window=np.hanning(160),detrend=False,nfft=1024)
            take=(f>=1000)&(f<7900)
            values.append(float(np.sum(f[take]*power[take])/power[take].sum()/1000))
        expected=float(np.median(values)) if values else None
        assert (expected is None)==(c['features']['centroid_khz'] is None)
        if expected is not None:
            assert abs(expected-c['features']['centroid_khz'])<1e-9; checked['scipy_centroid']+=1
        if c['vowel_span']:
            va,vb=[v-offset for v in c['vowel_span']]; strengths=[]
            for i in range(0,len(x)-480+1,80):
                if i/sr < va-1e-8 or (i+480)/sr > min(vb,va+.04)+1e-8: continue
                y=x[i:i+480]-x[i:i+480].mean()
                if np.mean(y*y)<1e-6: continue
                strengths.append(max(0.,min(1.,max(np.dot(y[:-lag],y[lag:])/np.sqrt(np.dot(y[:-lag],y[:-lag])*np.dot(y[lag:],y[lag:])) for lag in range(32,201)))))
            p=float(np.median(strengths)) if strengths else None
            assert (p is None)==(c['features']['vowel_periodicity'] is None)
            if p is not None:
                assert abs(p-c['features']['vowel_periodicity'])<1e-8; checked['direct_periodicity']+=1
            if c['features']['vowel_rise_db'] is not None:
                assert vb-va>=.09-1e-8
                y=x[round(va*sr):round((va+.03)*sr)]
                z=x[round((va+.06)*sr):round((va+.09)*sr)]
                ratio=10*np.log10(np.var(z)/np.var(y))
                assert abs(ratio-c['features']['vowel_rise_db'])<1e-9; checked['direct_vowel_rise']+=1
    nbpath=args.atlas.parent/'한국어_자모별_음성_전사율.ipynb'
    nb=json.loads(nbpath.read_text()); codes=[c for c in nb['cells'] if c['cell_type']=='code']
    errors=[o for c in codes for o in c['outputs'] if o['output_type']=='error']
    warnings=[o.get('text','') for c in codes for o in c['outputs'] if o['output_type']=='stream' and 'Warning' in str(o.get('text',''))]
    assert not errors and not warnings
    assert all(c.get('execution_count') is not None for c in codes)
    csvhash=sha(args.atlas/'자모별_전사율.csv')
    assert csvhash=='b994be62fbf51ac2865d20a02681991a3174ef1cd5ca092002fa5bae825e0f5a'
    result=dict(status='verified',population_partition=[len(targets),len(cases),len(excluded)],
        saved_prediction_matches=len(cases),contexts=len(report['contexts']),speaker_context_pairs=len(paired_keys),
        independent_acoustics_sample=len(sample),independent_measurements=dict(checked),
        original_all_jamo_csv_unchanged=True,executed_cells=len(codes),notebook_errors=len(errors),notebook_warnings=len(warnings),
        verification_source_sha256=sha(__file__),
        outputs_sha256={str(p):sha(p) for p in (nbpath,nbpath.with_suffix('.html'),folder/'report.json',folder/'cases.json',folder/'features.csv')})
    (folder/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

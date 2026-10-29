"""P4-D raw annotation provenance and acoustic QC; never independent listening."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/data')]
import argparse
import collections
import hashlib
import json
import numpy as np
from shared.paths import DATA_ROOT
from build_foreign_clean_v2 import parse_one
from diagnose_h0_p4 import read_jsonl,write_json
from train_w2vbert_ctc import read_wav
from train_h0_backbone_control import file_sha
from fit_h0_representation_probes import table


def acoustic_qc(audio,span):
    if span is None:return dict(status='no_confirmed_TAG_window')
    left=max(0,int(round(span[0]*16000)));right=min(len(audio),int(round(span[1]*16000)))
    x=audio[left:right]
    if len(x)<2:return dict(status='no_samples')
    centered=x-x.mean();power=np.abs(np.fft.rfft(centered*np.hanning(len(x))))**2
    frequencies=np.fft.rfftfreq(len(x),1/16000.);total=power.sum()
    return dict(status='measured',samples=len(x),duration_ms=len(x)/16.,rms=float(np.sqrt(np.mean(x*x))),
        peak=float(np.abs(x).max()),clipped_fraction=float((np.abs(x)>=32767/32768).mean()),
        dc=float(x.mean()),spectral_centroid_hz=float((power*frequencies).sum()/total) if total else None,
        energy_below_1k=float(power[frequencies<1000].sum()/total) if total else None,
        span_end_exceeds_audio_ms=max(0.,1000*span[1]-len(audio)/16.),
        pcm_segment_sha256=hashlib.sha256(x.tobytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);a=ap.parse_args()
    root=Path(a.root);p=json.loads((root/'protocol.json').read_text());out=root/'audio_labels';out.mkdir(exist_ok=False)
    records=read_jsonl(Path(p['p1'])/'utterances.jsonl');sites=read_jsonl(Path(p['p1'])/'sites.jsonl')
    meta={m['stem']:m for split in ('valid','valid_unseen') for m in read_jsonl(Path(p['manifest'])/(split+'.meta.jsonl'))}
    bases={d.name[:3]:d for d in (DATA_ROOT/'aihub').glob('04[0-3].*')}
    selected=[];sourcehash={};matched_contexts=collections.Counter();raw_utterances=set();prompt_names={}
    for m in meta.values():
        if m['item'] in ('RW052','RW054'):prompt_names[m['item']]=m['prompt']
    for r in records:
        if r['role']!='selection':continue
        chosen=[]
        for j,c in enumerate(r['canonical']):
            s=sites[r['offset']+j];t=s['actual'];left=r['canonical'][j-1] if j else '<BOS>';right=r['canonical'][j+1] if j+1<len(r['canonical']) else '<EOS>'
            cohorts=[]
            if r['item']=='RW054' and r['whole_tag_normal'] and c=='ㅜ':cohorts.append('normal_suri_475')
            if t=='ㅗ' and left=='ㅅ' and right=='ㄹ':cohorts.append('actual_o_same_canonical_neighbors')
            if r['split']=='valid_unseen' and c=='ㅜ' and t=='ㅗ':cohorts.append('unseen_expected_u_actual_o')
            if r['split']=='valid_unseen' and c=='ㅗ' and t=='ㅗ':cohorts.append('unseen_normal_o')
            if cohorts:chosen.append((j,s,left,right,cohorts))
        if not chosen:continue
        m=meta[r['uid']];base=bases[m['ds']]/'01-1.정식개방데이터'/('Training' if m['part']=='TL' else 'Validation')/'02.라벨링데이터'
        tag=base/(m['part']+'_pronunciation_tag')/(r['uid']+'.TextGrid');lab=base/(m['part']+'_pronunciation_lab')/(r['uid']+'.json')
        parsed=parse_one((r['uid'],m['ds'],m['part'],str(tag),str(lab)))
        if parsed.get('canon')!=r['canonical'] or parsed.get('ltr')!=r['actual'] or parsed.get('tag_phones')!=m['tag_phones']:
            raise ValueError('raw annotation differs from evaluation')
        sourcehash[str(tag)]=file_sha(tag);sourcehash[str(lab)]=file_sha(lab);raw_utterances.add(r['uid'])
        x=read_wav(r['path'],20.)
        for j,s,left,right,cohorts in chosen:
            row=dict(id=s['id'],uid=r['uid'],q=j,split=r['split'],item=r['item'],speaker=r['speaker'],
                language=m['language'],task=r['task'],prompt=r['prompt'],cohorts=cohorts,expected=s['expected'],actual=s['actual'],
                canonical_left=left,canonical_right=right,tag_window=s['tag_window'],timing_status=s['timing_status'],
                original_TAG=str(tag),original_LAB=str(lab),audio_path=r['path'],
                raw_annotation_reproduced=True,audio_listened=False,independent_phonetic_judgment=None,
                **acoustic_qc(x,s['tag_window']))
            selected.append(row)
            for cohort in cohorts:matched_contexts[cohort,r['split'],r['item']]+=1
    summary=[]
    for cohort in sorted({c for r in selected for c in r['cohorts']}):
        rs=[r for r in selected if cohort in r['cohorts']];valid=[r for r in rs if r['status']=='measured']
        summary.append(dict(cohort=cohort,sites=len(rs),utterances=len({r['uid'] for r in rs}),speakers=len({r['speaker'] for r in rs}),
            items=dict(collections.Counter(r['item'] for r in rs)),available=len(valid),
            quantiles={key:np.quantile([r[key] for r in valid if r[key] is not None],[.1,.5,.9]).tolist()
                       if any(r[key] is not None for r in valid) else None for key in
                       ('duration_ms','rms','clipped_fraction','spectral_centroid_hz','energy_below_1k')}))
    write_json(out/'review_manifest_NOT_LISTENED.json',dict(sites=selected,raw_file_hashes=sourcehash))
    table(out/'acoustic_qc.tsv',selected);table(out/'cohorts.tsv',summary)
    write_json(out/'report.json',dict(status='completed',raw_utterances=len(raw_utterances),raw_files=len(sourcehash),
        prompt_names=prompt_names,cohorts=summary,source_sha256=file_sha(__file__),
        limitations=['source agreement does not establish perceptual label correctness',
            'canonical neighbors are coarse matching, not identical actual coarticulation or speaker/language matching',
            'RMS and spectral statistics are quality/descriptive checks, not validated vowel or formant labels',
            'single Suri all-u cohort has no binary AUC',
            'no listening, relabeling, exclusion, or independent evaluation collection']))


if __name__=='__main__':main()

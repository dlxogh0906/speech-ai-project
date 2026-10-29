"""TRAIN pilot of observing unlinked C slots from runtime following-V tokens.

Original API bodies are reproduced before extraction. Vowel token time is only
a reference. TAG support is attached after the window/features are fixed.
The cache explicitly uses relative_vowel to reject accidental C-centered reuse.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse
import collections
import hashlib
import json
import time
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from,render
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_observed_anchor import observed_tokens
from mdd.sibilant_vowel_reference import SCHEMA,reference_for_site,original_window,annotated_intended_vowel_interval
from mdd.sibilant_source_view import source_relative
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_frame_clock import stacked_window_centers


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    annotations={z['stem']:z for z in service.read_jsonl(p['train_annotations'])}
    app,captured=app_from(p),{};app.S['cb']=TimePhraseBuilder(app.S['cb']);dec=app.S['dec'];forward=app.S['fwd']
    fe=SeamlessM4TFeatureExtractor.from_pretrained(p['checkpoint'])
    def capture(decoder,wavs):
        lp,length=forward(decoder,wavs);captured.update(wav=wavs[0].copy(),lp=lp[0,:length[0]].numpy().copy());return lp,length
    app.S['fwd']=capture
    arrays,features,audits=[],[],[];counts=collections.Counter();tick=time.monotonic()
    with (out/'responses.jsonl').open('x') as stream:
        for n,record in enumerate(service.read_jsonl(p['sources']),1):
            row,prior=record['row'],record['prior'];x=service.load_audio(row)[:320000]
            result,_,raw=app.analyze(x,row['prompt'],'');wav,lp=captured['wav'],captured['lp']
            sites,_=annotate_margin(lp,raw['canon'],raw['sites'],dec.d.index,dec.blank,p['pair_thresholds'])
            if raw['canon']!=prior['raw']['canon'] or render(app,row,raw,sites)!=prior['body']:raise AssertionError('original complete API response differs')
            indices=retained_samples(x);np.testing.assert_array_equal(x[indices],wav)
            tokens,spans=observed_tokens(lp,dec.d,dec.blank)
            if tokens!=result['pred_jamo']:raise AssertionError('decoder sequence differs')
            opt=service.all_optimal_mapping(raw['canon'],tokens);dt=len(wav)/(16000.*len(lp))
            # Only automatic observed information constructs the window.
            windows=[]
            for request in record['runtime_unlinked_queries']:
                q=request['q_runtime'];anchor,state=reference_for_site(tokens,spans,raw['canon'],raw['sites'],q,opt)
                value={'uid':row['uid'],'q_runtime':q,'state':state,'feature_index':None,'baseline_visible_warning':request['baseline_visible_warning']}
                if anchor is not None:
                    left,right,shift=original_window(indices,len(x),anchor['frame'],dt);clip=x[left:right]
                    extracted=fe([clip],sampling_rate=16000,return_tensors='np',padding=True);length=int(extracted['attention_mask'][0].sum())
                    if not 3<=length<=p['max_frames']:value['state']='source_length_unsupported'
                    else:
                        value.update(feature_index=len(arrays),left=left,right=right,frame_sec=dt,anchor=anchor,
                            runtime_C=raw['canon'][q],runtime_expected_V=raw['canon'][q+1],observed_V=anchor['token'],input_frames=length,
                            clip_sha256=hashlib.sha256(clip.tobytes()).hexdigest(),coordinate_schema=SCHEMA)
                        arrays.append((extracted['input_features'][0,:length].astype(np.float32),source_relative(left,right,length,shift,anchor['frame'],dt)))
                        features.append(dict(value))
                windows.append(value);counts[value['state']]+=1
            # Existing TAG is used only now, for an annotation-support diagnostic.
            canonmap=service.all_optimal_mapping(row['canonical'],raw['canon']);actualmap=service.all_optimal_mapping(row['canonical'],row['actual'])
            intervals=single_phone_intervals(annotations[row['uid']]['tag_phones'],row['actual'])
            for value in windows:
                audit=dict(value,TAG_state='automatic_unavailable')
                if value['feature_index'] is not None:
                    ls=[z for z in row['sites'] if canonmap[z['q']]=={value['q_runtime']}]
                    audit['TAG_state']='label_mapping_unknown'
                    if len(ls)==1 and not ls[0]['alignment_ambiguous']:
                        z=ls[0];audit.update(expected=z['canon'],actual=z.get('actual'),label_kind=z['label_kind'],q=z['q'])
                        clip_indices=np.arange(value['left'],value['right']);centers=stacked_window_centers(value['left'],value['right'],value['input_frames'])/16000.
                        if z.get('a') is None:audit['TAG_state']='TAG_C_deleted'
                        elif intervals is not None and intervals[z['a']] is not None:
                            start,end,token=intervals[z['a']]
                            c=int(((clip_indices>=np.ceil(start*16000))&(clip_indices<np.ceil(end*16000))).sum())
                            audit.update(TAG_state='TAG_C_in_window' if c else 'TAG_C_outside_window',C_clip_samples=c,
                                known_C_frame_centers=int(((centers>start+.01)&(centers<end-.01)).sum()))
                        options=actualmap[z['q']+1] if z['q']+1<len(actualmap) else set()
                        vowel=annotated_intended_vowel_interval(intervals,options)
                        if vowel is not None:
                            vstart,vend,vtoken=vowel
                            audit.update(intended_V_token=vtoken,V_clip_samples=int(((clip_indices>=np.ceil(vstart*16000))&(clip_indices<np.ceil(vend*16000))).sum()),
                                known_V_frame_centers=int(((centers>vstart+.01)&(centers<vend-.01)).sum()))
                audits.append(audit)
            counts['complete_API_bodies_and_VAD_reproduced']+=1;service.write_line(stream,record)
            if n%25==0:print(n,dict(counts),'seconds',round(time.monotonic()-tick,1),flush=True)
    data={'x':np.zeros((len(arrays),p['max_frames'],160),np.float32),'relative_vowel':np.zeros((len(arrays),p['max_frames']),np.float32),'lengths':np.array([len(a[0]) for a in arrays],np.int64)}
    for i,(x,r) in enumerate(arrays):data['x'][i,:len(x)]=x;data['relative_vowel'][i,:len(x)]=r
    np.savez(out/'windows.npz',**data)
    for name,rows in [('features',features),('support_audit',audits)]:
        with (out/(name+'.jsonl')).open('x') as stream:
            for row in rows:service.write_line(stream,row)
    report={'scope':p['scope'],'schema':SCHEMA,'counts':dict(counts),'features':len(features),'all_requested_sites':len(audits),
        'TAG_states':dict(collections.Counter(a['TAG_state'] for a in audits)),
        'existing_warning_TAG_states':dict(collections.Counter(a['TAG_state'] for a in audits if a['baseline_visible_warning'])),
        'both_C_and_intended_V_samples':sum(a.get('C_clip_samples',0)>0 and a.get('V_clip_samples',0)>0 for a in audits),
        'both_C_and_intended_V_known_frame_centers':sum(a.get('known_C_frame_centers',0)>0 and a.get('known_V_frame_centers',0)>0 for a in audits),
        'TAG_deleted_C_with_intended_V_samples':sum(a['TAG_state']=='TAG_C_deleted' and a.get('V_clip_samples',0)>0 for a in audits),
        'seconds':time.monotonic()-tick,'runtime_inputs_use_TAG':False,'old_C_centered_head_not_used':True,'service_improvement_proven':False}
    write=lambda f,v:f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    write(out/'report.json',report);write(out/'finished.json',{'sha256':service.sha(out/'report.json')});print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

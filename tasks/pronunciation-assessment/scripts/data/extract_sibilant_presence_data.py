"""Original-PCM V-reference data for linked controls and unlinked C queries.

Observe all declared runtime queries before attaching TAG targets. TRAIN masks
unsupported class supervision; CAL keeps all outcomes and never uses TAG time.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import hashlib
import json
import time
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from,render
from eval_speech_rescue import observe
from audit_sibilant_anchor_coverage import retained_samples
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_observed_anchor import observed_tokens
from mdd.sibilant_vowel_reference import original_window,annotated_intended_vowel_interval
from mdd.sibilant_source_view import source_relative
from mdd.sibilant_tag_support import single_phone_intervals
from mdd.sibilant_frame_clock import stacked_window_centers
from mdd.sibilant_vowel_presence import SCHEMA,next_reference,actual_class,training_target
from mdd.sibilant_transition import PAIR,VOWELS


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    train=p['partition']=='train';annotations={z['stem']:z for z in service.read_jsonl(p['train_annotations'])} if train else {}
    app,captured=app_from(p),{};app.S['cb']=TimePhraseBuilder(app.S['cb']);dec=app.S['dec'];forward=app.S['fwd']
    fe=SeamlessM4TFeatureExtractor.from_pretrained(p['checkpoint'])
    def capture(decoder,wavs):
        lp,length=forward(decoder,wavs);captured.update(wav=wavs[0].copy(),lp=lp[0,:length[0]].numpy().copy());return lp,length
    app.S['fwd']=capture
    arrays,features,labels,targets,phases,queries=[],[],[],[],[],[];counts=collections.Counter();tick=time.monotonic()
    with (out/'data.responses.jsonl').open('x') as stream:
        for n,record in enumerate(service.read_jsonl(p['sources']),1):
            row,prior=record['row'],record['prior'];by_q={};raw=prior['raw']
            if raw is None:counts['input_rejected_utterances']+=1
            else:
                requested=record.get('presence_queries')
                if requested is None:requested=[q for q,t in enumerate(raw['canon'][:-1]) if t in PAIR and raw['canon'][q+1] in VOWELS]
                if requested:
                    x=service.load_audio(row)[:320000];result,_,current=app.analyze(x,row['prompt'],'')
                    wav,lp=captured['wav'],captured['lp'];sites,_=annotate_margin(lp,current['canon'],current['sites'],dec.d.index,dec.blank,p['pair_thresholds'])
                    if current['canon']!=raw['canon'] or render(app,row,current,sites)!=prior['body']:raise AssertionError('cached complete API response did not reproduce: '+row['uid'])
                    indices=retained_samples(x);np.testing.assert_array_equal(x[indices],wav)
                    tokens,spans=observed_tokens(lp,dec.d,dec.blank)
                    if tokens!=result['pred_jamo']:raise AssertionError('decoded sequence differs')
                    optimal=service.all_optimal_mapping(raw['canon'],tokens);dt=len(wav)/(16000.*len(lp));phones=service.phonemes(prior['body'])
                    for q in requested:
                        anchor,state=next_reference(tokens,spans,raw['canon'],raw['sites'],q,optimal)
                        query={'uid':row['uid'],'q_runtime':q,'state':state,'feature_index':None,
                               'C_linked':raw['sites'][q].get('j') is not None,'baseline_warning':bool(phones[q].get('errorType'))}
                        if anchor is not None:
                            left,right,shift=original_window(indices,len(x),anchor['frame'],dt);clip=x[left:right]
                            value=fe([clip],sampling_rate=16000,return_tensors='np',padding=True);length=int(value['attention_mask'][0].sum())
                            if not 3<=length<=p['max_frames']:query['state']='unsupported_length'
                            else:
                                i=len(arrays);by_q[q]=i;query['feature_index']=i
                                arrays.append((value['input_features'][0,:length].astype(np.float32),source_relative(left,right,length,shift,anchor['frame'],dt)))
                                features.append(dict(query,canonical=raw['canon'][q],baseline_pred=raw['sites'][q].get('pred'),
                                    baseline_actual=phones[q].get('actual_jamo'),expected_V=raw['canon'][q+1],observed_V=anchor['token'],
                                    left=left,right=right,anchor=anchor,frame_sec=dt,input_frames=length,coordinate_schema=SCHEMA,
                                    clip_sha256=hashlib.sha256(clip.tobytes()).hexdigest()))
                                targets.append(-100);phases.append(np.full(p['max_frames'],-100,np.int64))
                        queries.append(query);counts[query['state']]+=1
                    counts['fresh_API_VAD_and_decoder_reproductions']+=1
            # All query audio/coordinates are fixed before labels are consulted.
            mapping=service.all_optimal_mapping(row['canonical'],raw['canon']) if raw else None
            prior_views=observe(prior,row)
            by_rq=collections.defaultdict(list)
            for z in row['sites']:
                options=mapping[z['q']] if mapping is not None else set();rq=next(iter(options)) if len(options)==1 and None not in options else None
                label=dict(z,expected=z['canon'],uid=row['uid'],speaker=row['speaker'],item=row['item'],language=row.get('language'),
                           q_runtime=rq,feature_index=by_q.get(rq),existing=prior_views[z['q']])
                labels.append(label)
                if rq is not None:by_rq[rq].append(label)
            actualmap=service.all_optimal_mapping(row['canonical'],row['actual']) if train else None
            intervals=single_phone_intervals(annotations[row['uid']]['tag_phones'],row['actual']) if train else None
            for q,i in by_q.items():
                ls=by_rq[q]
                if len(ls)!=1:counts['target/label_mapping_unknown']+=1;continue
                z=ls[0];target=actual_class(z);state='CAL_identity_only'
                if train:
                    f=features[i];opts=actualmap[z['q']+1] if z['q']+1<len(actualmap) else set();v=annotated_intended_vowel_interval(intervals,opts)
                    c=intervals[z['a']] if intervals is not None and z.get('a') is not None else None
                    adjacent=z.get('a') is not None and opts=={z['a']+1}
                    centers=stacked_window_centers(f['left'],f['right'],f['input_frames'])/16000.
                    target,phase,state=training_target(z,c,v,f['left'],f['right'],centers,adjacent)
                    phases[i][:len(phase)]=phase
                targets[i]=target;counts['target/'+state]+=1
            service.write_line(stream,record);counts['source_utterances']+=1
            if n%100==0:print(p['partition'],n,dict(counts),'features',len(arrays),'seconds',round(time.monotonic()-tick,1),flush=True)
    data={'x':np.zeros((len(arrays),p['max_frames'],160),np.float32),'relative_vowel':np.zeros((len(arrays),p['max_frames']),np.float32),'lengths':np.array([len(x) for x,_ in arrays],np.int64)}
    for i,(x,relative) in enumerate(arrays):data['x'][i,:len(x)]=x;data['relative_vowel'][i,:len(x)]=relative
    np.savez(out/'data.npz',**data);np.savez(out/'targets.npz',y=np.array(targets,np.int64),phase=np.asarray(phases,np.int64))
    for name,rows in [('features',features),('labels',labels),('queries',queries)]:
        with (out/('data.'+name+'.jsonl')).open('x') as stream:
            for row in rows:service.write_line(stream,row)
    report={'scope':p['scope'],'partition':p['partition'],'schema':SCHEMA,'counts':dict(counts),'features':len(features),'all_label_sites':len(labels),
        'class_targets':{str(k):int(sum(y==k for y in targets)) for k in [-100,0,1,2,3,4]},'seconds':time.monotonic()-tick,
        'runtime_TAG_inputs':False,'sha256':{f.name:service.sha(f) for f in out.glob('data.*')},'targets_sha256':service.sha(out/'targets.npz')}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

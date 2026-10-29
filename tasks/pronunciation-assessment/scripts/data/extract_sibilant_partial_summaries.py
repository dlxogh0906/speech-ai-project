"""Original automatic runtime observations before attaching partial TAG labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from eval_sibilant_typed_margin import app_from,render
from audit_sibilant_anchor_coverage import retained_samples
from extract_sibilant_normal_summaries import native_grid,summarize_clip
from mdd.canonical_time import TimePhraseBuilder
from mdd.sibilant_margin import annotate_margin
from mdd.sibilant_observed_anchor import observed_tokens
from mdd.sibilant_vowel_reference import original_window
from mdd.sibilant_source_view import source_relative
from mdd.sibilant_vowel_presence import SCHEMA,next_reference,training_target
from mdd.sibilant_transition import PAIR,VOWELS
from mdd.sibilant_frozen_probe import load_probe
from mdd.sibilant_hubert_adaptation import load_native,tensor_digest
from mdd.sibilant_precision_tail import PrecisionTailHead,SUMMARY_SCHEMA
from mdd.sibilant_partial_runtime import runtime_labels


def write(path,value):
    Path(path).write_text(json.dumps(service.jsonable(value),ensure_ascii=False,indent=2)+'\n')


def observe_one(ctx,runtime_input,offset):
    """This function receives only UID/WAV/frames/prompt, no source annotations."""
    x = service.load_audio(runtime_input)[:320000]
    code,_,gate = ctx.app.check_audio(x,runtime_input['prompt'])
    record = {'rejected':code,'raw':None,'body':None}
    features,summaries,moments,availability,queries = [],[],[],[],[]
    if code:
        return record,gate,features,summaries,moments,availability,queries
    result,_,raw = ctx.app.analyze(x,runtime_input['prompt'],'')
    wav,lp = ctx.captured['wav'],ctx.captured['lp']
    sites,_ = annotate_margin(lp,raw['canon'],raw['sites'],ctx.dec.d.index,ctx.dec.blank,ctx.p['pair_thresholds'])
    record.update(raw=raw,body=render(ctx.app,runtime_input,raw,sites))
    indices = retained_samples(x)
    np.testing.assert_array_equal(x[indices],wav)
    tokens,spans = observed_tokens(lp,ctx.dec.d,ctx.dec.blank)
    if tokens != result['pred_jamo']:
        raise ValueError('observed decoder sequence changed')
    optimal = service.all_optimal_mapping(raw['canon'],tokens)
    dt = len(wav)/(16000.*len(lp));phones = service.phonemes(record['body'])
    for q,token in enumerate(raw['canon'][:-1]):
        if token not in PAIR or raw['canon'][q+1] not in VOWELS:
            continue
        anchor,state = next_reference(tokens,spans,raw['canon'],raw['sites'],q,optimal)
        query = {'uid':runtime_input['uid'],'q_runtime':q,'state':state,'feature_index':None,
            'C_linked':raw['sites'][q].get('j') is not None,'baseline_warning':bool(phones[q].get('errorType'))}
        if anchor is not None:
            left,right,shift = original_window(indices,len(x),anchor['frame'],dt)
            clip = x[left:right]
            values = ctx.fe([clip],sampling_rate=16000,return_tensors='np',padding=True)
            length = int(values['attention_mask'][0].sum())
            if not 3 <= length <= ctx.p['max_frames']:
                query['state'] = 'unsupported_length'
            else:
                query['feature_index'] = offset+len(features)
                feature = dict(query,canonical=token,baseline_pred=raw['sites'][q].get('pred'),
                    baseline_actual=phones[q].get('actual_jamo'),expected_V=raw['canon'][q+1],
                    observed_V=anchor['token'],left=left,right=right,anchor=anchor,frame_sec=dt,
                    input_frames=length,coordinate_schema=SCHEMA,
                    clip_sha256=hashlib.sha256(clip.tobytes()).hexdigest())
                relative = source_relative(left,right,length,shift,anchor['frame'],dt)
                grid,relative = native_grid(feature,relative)
                summary,moment,available = summarize_clip(ctx.encoder,ctx.processor,ctx.model,x,grid,relative)
                features.append(grid);summaries.append(summary);moments.append(moment);availability.append(available)
        queries.append(query)
    return record,gate,features,summaries,moments,availability,queries


def check_controls(p,features,summary,moments,available,records,model,device):
    c = p['control']
    original = service.read_jsonl(c['features'])
    old = np.load(c['summary']);old_scores = np.load(c['scores'])
    by = {(f['uid'],f['q_runtime']):i for i,f in enumerate(features)}
    ix = []
    for i in c['indices']:
        f = original[i];j = by[(f['uid'],f['q_runtime'])];current = features[j]
        for key in ('left','right','clip_sha256','input_frames','frame_first_source_sample','observed_reference_seconds'):
            if current[key] != f[key]:
                raise ValueError('original-control runtime geometry changed: '+key)
        ix.append(j)
    expected = np.asarray(c['indices'],np.int64)
    np.testing.assert_allclose(summary[ix],old['summary'][expected],atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(moments[ix],old['moments'][expected],atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(available[ix],old['available'][expected])
    with torch.no_grad():
        ps = model.from_summary(torch.as_tensor(summary[ix],device=device),
            torch.as_tensor(moments[ix],device=device)).softmax(-1).cpu().numpy()
    np.testing.assert_allclose(ps,old_scores['probabilities'][expected],atol=2e-6,rtol=2e-6)
    source = {z['row']['uid']:z['prior'] for z in service.read_jsonl(c['responses'])}
    for uid,record in records.items():
        if record['body'] != source[uid]['body'] or record['rejected'] != source[uid]['rejected']:
            raise ValueError('original-control complete API body/gate changed: '+uid)
    return {'controls':len(ix),'original_runtime_geometry_exact':True,'complete_API_bodies_exact':len(records),
        'availability_exact':True,'summary_max_difference':float(abs(summary[ix]-old['summary'][expected]).max()),
        'moment_max_difference':float(abs(moments[ix]-old['moments'][expected]).max()),
        'probability_max_difference':float(abs(ps-old_scores['probabilities'][expected]).max()),
        'scope':'fixed prior streaming tolerance; not bitwise feature parity or performance'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a = ap.parse_args();p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for name,digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen partial runtime source changed: '+name)
    rows = service.read_jsonl(p['rows'])
    if len(rows) != p['expected_rows']:
        raise ValueError('fixed source population changed')
    torch.set_num_threads(2);device = torch.device(p.get('device','cuda'))
    encoder,processor,_ = load_native(p['model_manifest'],device)
    artifact = torch.load(p['control_artifact'],map_location='cpu')
    model = PrecisionTailHead(load_probe(artifact,p['binding'],device),'ce_full').to(device).eval().requires_grad_(False)
    before = tensor_digest(model.state_dict());captured = {};app = app_from(p)
    app.S['cb'] = TimePhraseBuilder(app.S['cb']);forward = app.S['fwd']
    def capture(decoder,wavs):
        lp,lengths = forward(decoder,wavs)
        captured.update(wav=wavs[0].copy(),lp=lp[0,:lengths[0]].numpy().copy())
        return lp,lengths
    app.S['fwd'] = capture
    ctx = SimpleNamespace(app=app,captured=captured,dec=app.S['dec'],p=p,encoder=encoder,processor=processor,
        model=model,fe=SeamlessM4TFeatureExtractor.from_pretrained(p['checkpoint']))
    out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',p)
    features,labels,summaries,moments,availability,targets,records = [],[],[],[],[],[],{}
    counts = collections.Counter();tick = time.monotonic()
    with torch.no_grad(),(out/'runtime_provenance.jsonl').open('x') as stream:
        for n,row in enumerate(rows,1):
            runtime_input = {k:row[k] for k in ('uid','wav','frames','prompt')}
            start = len(features)
            record,gate,fs,ss,mm,av,queries = observe_one(ctx,runtime_input,start)
            features.extend(fs);summaries.extend(ss);moments.extend(mm);availability.extend(av);targets.extend([-100]*len(fs))
            counts['utterances'] += 1;counts['rejected_utterances' if record['rejected'] else 'accepted_utterances'] += 1
            counts.update('query/'+q['state'] for q in queries)
            if p.get('control'):
                records[row['uid']] = record
            else:
                # All runtime observations for this row now exist. First use
                # of expected-partial/TAG/site information occurs below.
                raw = record['raw'];canonical = raw['canon'] if raw else None
                phones = service.phonemes(record['body']) if raw else []
                by_q = {f['q_runtime']:f['feature_index'] for f in fs}
                current = runtime_labels(row,canonical,phones,by_q,record['rejected'])
                indexed = {z['feature_index']:z for z in current if z['feature_index'] is not None}
                for f in fs:
                    i = f['feature_index'];z = indexed.get(i)
                    if z is None:
                        counts['target/runtime_query_without_certain_source_label'] += 1
                        current.append({'uid':row['uid'],'speaker':row['speaker'],'item':row['item'],
                            'language':row.get('language'),'expected':f['canonical'],'actual':None,
                            'following':f['expected_V'],'position':'runtime_context_only','label_kind':'ambiguous',
                            'alignment_ambiguous':True,'q':None,'q_runtime':f['q_runtime'],'feature_index':i,
                            'source_time_witness':False,'runtime_unmapped_source_query':True,
                            'existing':{'state':'runtime_source_unknown','phone':phones[f['q_runtime']]}})
                        continue
                    centers = (f['frame_first_source_sample']+np.arange(f['input_frames'])*320)/16000.
                    y,_,state = training_target(z,z['C_TAG_interval'],z['V_TAG_interval'],f['left'],f['right'],
                        centers,z['source_time_witness'] and z['state'] == 'literal_CV_same_all_optimal_correspondences')
                    targets[i] = y;z['TRAIN_target_state'] = state;z['TRAIN_target_y'] = int(y)
                    counts['target/'+state] += 1
                labels.extend(current)
            raw = record['raw']
            service.write_line(stream,{'uid':row['uid'],'rejected':record['rejected'],'audio_gate':gate,
                'runtime_canonical':raw['canon'] if raw else None,'queries':queries,
                'raw_sha256':hashlib.sha256(json.dumps(service.jsonable(raw),sort_keys=True).encode()).hexdigest(),
                'API_body_sha256':hashlib.sha256(json.dumps(service.jsonable(record['body']),sort_keys=True).encode()).hexdigest()})
            if n%25 == 0 or n == len(rows):
                progress = {'rows':n,'features':len(features),'counts':dict(counts),'seconds':time.monotonic()-tick}
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary = np.asarray(summaries,np.float32).reshape(-1,3*model.base.components.shape[1])
    moment = np.asarray(moments,np.float32).reshape(-1,9);available = np.asarray(availability,bool)
    y = np.asarray(targets,np.int64)
    if tensor_digest(model.state_dict()) != before:
        raise ValueError('frozen native observations/readout changed')
    control_report = check_controls(p,features,summary,moment,available,records,model,device) if p.get('control') else None
    ps = []
    with torch.no_grad():
        for start in range(0,len(summary),64):
            ps.extend(model.from_summary(torch.as_tensor(summary[start:start+64],device=device),
                torch.as_tensor(moment[start:start+64],device=device)).softmax(-1).cpu().numpy())
    np.savez(out/'summary.npz',summary=summary,moments=moment,available=available,y=y,
        probabilities=np.asarray(ps,np.float32).reshape(-1,5))
    for name,values in [('features',features),('labels',labels)]:
        with (out/('data.'+name+'.jsonl')).open('x') as stream:
            for value in values:service.write_line(stream,value)
    kinds = collections.Counter(z['label_kind'] for z in labels)
    supported = collections.Counter(z['label_kind'] for z in labels if z['feature_index'] is not None and y[z['feature_index']] >= 0)
    if not p.get('control') and kinds['pair_error'] != p['expected_source_pair_sites']:
        raise ValueError('source728error denominator changed')
    report = {'scope':p['scope'],'partition':'train','schema':SUMMARY_SCHEMA,'rows':len(rows),
        'features':len(features),'all_labels':len(labels),'source_label_kinds':dict(kinds),
        'supported_label_kinds':dict(supported),'known_input_classes':int((y>=0).sum()),
        'unknown_input_classes':int((y<0).sum()),'no_input_labels':sum(z['feature_index'] is None for z in labels),
        'counts':dict(counts),'control':control_report,'TAG_in_runtime_input':False,
        'frozen_parent_unchanged':True,'seconds':time.monotonic()-tick,'deployment':False,'goal_complete':False,
        'sha256':{name:service.sha(out/name) for name in ('summary.npz','data.features.jsonl','data.labels.jsonl','runtime_provenance.jsonl')}}
    write(out/'report.json',report);write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':main()

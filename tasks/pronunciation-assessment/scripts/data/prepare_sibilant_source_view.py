"""Change only the local head's audio view; retain baseline ASR/API and labels.

Recover the exact cached runtime anchor, map its center back to original input,
then take a continuous source window of the same theoretical duration. Rebuild
all old features first. Existing TAG is a separate diagnostic/TRAIN auxiliary
target, never an inference boundary or a source-selection criterion here.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import collections
import hashlib
import json
import time
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from audit_sibilant_anchor_coverage import retained_samples
from mdd.sibilant_source_view import cached_anchor, source_bounds, source_relative
from mdd.sibilant_tag_support import single_phone_intervals, coarse_roles
from mdd.sibilant_frame_clock import stacked_window_centers
from mdd.sibilant_transition import VOWELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p['sha256'].items():
        if service.sha(path) != digest:
            raise ValueError('frozen source changed: '+path)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_text(Path(a.protocol).read_text())
    old = dict(np.load(p['prefix']+'.npz'))
    features = service.read_jsonl(p['prefix']+'.features.jsonl')
    labels = service.read_jsonl(p['prefix']+'.labels.jsonl')
    records = service.read_jsonl(p['sources'])
    wanted = {r['row']['uid'] for r in records}
    selected = [i for i,f in enumerate(features) if f['uid'] in wanted]
    remap = {j:i for i,j in enumerate(selected)}
    data = {k:v[selected].copy() for k,v in old.items()}
    by_uid, by_feature = collections.defaultdict(list), collections.defaultdict(list)
    for i in selected:
        by_uid[features[i]['uid']].append(i)
    for label in labels:
        if label.get('feature_index') is not None:
            by_feature[label['feature_index']].append(label)
    annotations = ({v['stem']:v for v in service.read_jsonl(p['annotations'])} if p.get('annotations') else {})
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p['checkpoint'])
    rows, audits, counts, transitions, tick = [], [], collections.Counter(), collections.Counter(), time.monotonic()
    saved = {}
    for n, record in enumerate(records,1):
        row = record['row']
        if not by_uid[row['uid']]:
            continue
        x = service.load_audio(row)[:320000]
        indices = retained_samples(x)
        wav = x[indices]
        digest = hashlib.sha256(wav.tobytes()).hexdigest()
        for i in by_uid[row['uid']]:
            f, length = features[i], int(old['lengths'][i])
            if digest != f['vad_pcm_sha256']:
                raise AssertionError('cached VAD PCM differs')
            clip = wav[f['left']:f['right']]
            value = fe([clip],sampling_rate=16000,return_tensors='np',padding=True)
            if hashlib.sha256(clip.tobytes()).hexdigest()!=f['clip_sha256'] or int(value['attention_mask'][0].sum())!=length:
                raise AssertionError('cached original clip differs')
            np.testing.assert_array_equal(value['input_features'][0,:length].astype(np.float32),old['x'][i,:length])
            frame,dt = cached_anchor(f['left'],f['right'],length,old['relative'][i],len(wav))
            left,right,shift = source_bounds(indices,frame,dt,len(x))
            source = x[left:right]
            extracted = fe([source],sampling_rate=16000,return_tensors='np',padding=True)
            slen = int(extracted['attention_mask'][0].sum())
            mode = 'source_continuous'
            if not 3 <= slen <= p['max_frames']:
                mode = 'fallback_unsupported_source_length'
                source, slen = clip, length
                newx, newrel = old['x'][i,:length], old['relative'][i,:length]
                source_indices = indices[f['left']:f['right']]
            else:
                newx = extracted['input_features'][0,:slen].astype(np.float32)
                newrel = source_relative(left,right,slen,shift,frame,dt)
                source_indices = np.arange(left,right)
            identical = length==slen and np.array_equal(old['x'][i,:length],newx) and np.array_equal(old['relative'][i,:length],newrel)
            counts['old_inputs_reproduced'] += 1
            counts['changed_inputs'] += int(not identical)
            counts[mode] += 1
            j = remap[i]
            data['x'][j] = 0;data['relative'][j] = 0
            data['x'][j,:slen] = newx;data['relative'][j,:slen] = newrel;data['lengths'][j] = slen
            nf = dict(f,old_feature_index=i,input_frames=slen,source_view_mode=mode,anchor_frame=frame,frame_sec=dt,
                      source_left=left,source_right=right,source_shift=shift,source_clip_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                      unchanged_input=identical,baseline_vad_left=f['left'],baseline_vad_right=f['right'],
                      baseline_clip_sha256=f['clip_sha256'],clip_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                      coordinate_domain='original' if mode=='source_continuous' else 'retained_vad_fallback')
            if mode=='source_continuous':
                nf.update(left=left,right=right)
            saved[i] = nf
            audit = {'uid':row['uid'],'old_feature_index':i,'feature_index':j,'q_runtime':f['q_runtime'],
                     'changed_input':not identical,'mode':mode,'state':'no_TAG_diagnostic'}
            # Neither these intervals nor the following phase targets modify
            # the already computed waveform, anchor, length or relative input.
            ls = by_feature[i]
            intervals = single_phone_intervals(annotations[row['uid']]['tag_phones'],row['actual']) if row['uid'] in annotations else None
            if len(ls)==1 and ls[0].get('a') is not None and not ls[0]['alignment_ambiguous'] and intervals is not None and intervals[ls[0]['a']] is not None:
                z = ls[0];start,end,token = intervals[z['a']];lo,hi = int(np.ceil(start*16000)),int(np.ceil(end*16000))
                old_indices = indices[f['left']:f['right']]
                oldcount = int(((old_indices>=lo)&(old_indices<hi)).sum());newcount = int(((source_indices>=lo)&(source_indices<hi)).sum())
                retained = int(((indices>=lo)&(indices<hi)).sum())
                before = 'present' if oldcount else 'outside' if retained else 'removed_by_VAD'
                after = 'present' if newcount else 'absent'
                transitions[(before,after)]+=1
                audit.update(state='TAG_support_known',before=before,after=after,old_C_samples=oldcount,new_C_samples=newcount,
                             expected=z['expected'],actual=token,following=z.get('following'))
            if 'phase' in data and not identical and mode=='source_continuous':
                target = np.full(p['max_frames'],-100,np.int64)
                if len(ls)==1 and ls[0].get('a') is not None and not ls[0]['alignment_ambiguous']:
                    centers = stacked_window_centers(left,right,slen)/16000.
                    target[:slen] = coarse_roles(centers,intervals,ls[0]['a'],VOWELS,p['vowel_prefix_seconds'],p['edge_ignore_seconds'])
                data['phase'][j] = target
            audits.append(audit)
        counts['VAD_waveforms_reproduced']+=1
        if n%500==0:
            print(n,dict(counts),'seconds',round(time.monotonic()-tick,1),flush=True)
    fs = [saved[i] for i in selected]
    for label in labels:
        if label['uid'] in wanted:
            z=dict(label)
            if z.get('feature_index') is not None:z['feature_index']=remap[z['feature_index']]
            rows.append(z)
    np.savez(out/'data.npz',**data)
    for name, values in [('features',fs),('labels',rows),('responses',records),('support_audit',audits)]:
        with (out/('data.'+name+'.jsonl')).open('x') as stream:
            for value in values:service.write_line(stream,value)
    repairs = transitions[('removed_by_VAD','present')]+transitions[('outside','present')]
    losses = transitions[('present','absent')]
    old_missing = sum(v for (before,after),v in transitions.items() if before!='present')
    old_present = sum(v for (before,after),v in transitions.items() if before=='present')
    gate = bool(old_missing and repairs/old_missing>=.1 and old_present and losses/old_present<=.01 and repairs>losses)
    report={'scope':p['scope'],'counts':dict(counts),'all_features':len(selected),'all_labels':len(rows),
            'transitions':{'/'.join(k):v for k,v in transitions.items()},'repairs':repairs,'losses':losses,
            'TRAIN_diagnostic_gate_passed':gate,'seconds':time.monotonic()-tick,'source_labels_unchanged':True,
            'inference_uses_TAG':False,'operating_ASR_or_API_modified':False,
            'sha256':{f.name:service.sha(f) for f in out.glob('data.*')}}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

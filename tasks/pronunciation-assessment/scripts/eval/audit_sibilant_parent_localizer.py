"""Change only the alignment encoder in a paired TRAIN localization diagnostic.

Use the same cached runtime canonical sequence, audio/VAD and paired-path
algorithm. Parent predictions do not change baseline words, scores or flags.
TAG supports are diagnostic outputs, never alignment inputs.
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
import torch
import eval_sibilant_directional as service
from audit_sibilant_anchor_coverage import retained_samples
from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
from mdd.sibilant_transition import paired_anchor
from mdd.sibilant_tag_support import single_phone_intervals


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    records=service.read_jsonl(p['sources']);features=service.read_jsonl(p['train_prefix']+'.features.jsonl')
    labels=service.read_jsonl(p['train_prefix']+'.labels.jsonl');annotations={z['stem']:z for z in service.read_jsonl(p['annotations'])}
    wanted={z['row']['uid'] for z in records};by_uid=collections.defaultdict(list);by_feature=collections.defaultdict(list)
    for i,f in enumerate(features):
        if f['uid'] in wanted:by_uid[f['uid']].append((i,f))
    for z in labels:
        if z.get('feature_index') is not None:by_feature[z['feature_index']].append(z)
    torch.set_num_threads(2);dec=HFDecoder(p['parent_checkpoint']);counts=collections.Counter();transitions=collections.Counter();rows=[];tick=time.monotonic()
    with (out/'sites.jsonl').open('x') as stream:
        for n,record in enumerate(records,1):
            row,raw=record['row'],record['prior']['raw'];x=service.load_audio(row)[:320000]
            indices=retained_samples(x);wav=x[indices];digest=hashlib.sha256(wav.tobytes()).hexdigest()
            post,lengths=forward_lprobs(dec,[wav]);lp=post[0,:lengths[0]].numpy();dt=len(wav)/(16000.*len(lp))
            ids=[dec.d.index(t) for t in raw['canon']]
            if dec.d.unk() in ids:raise ValueError('parent vocabulary does not cover runtime canonical')
            for i,f in by_uid[row['uid']]:
                if digest!=f['vad_pcm_sha256'] or hashlib.sha256(wav[f['left']:f['right']].tobytes()).hexdigest()!=f['clip_sha256']:
                    raise AssertionError('original VAD/clip input differs')
                anchor=paired_anchor(lp,ids,f['q_runtime'],[dec.d.index('ㅅ'),dec.d.index('ㅆ')],dec.blank)
                value={'uid':row['uid'],'feature_index':i,'q_runtime':f['q_runtime'],'anchor':anchor,'frame_sec':dt,'state':'TAG_support_unknown'}
                counts['sites_all']+=1
                if anchor is None:
                    left,right=0,0;counts['parent_alignment_unavailable']+=1
                else:
                    left=max(0,int((anchor['frame']-15)*dt*16000));right=min(len(wav),int((anchor['frame']+16)*dt*16000))
                    value.update(left=left,right=right)
                    counts['window_changed']+=int((left,right)!=(f['left'],f['right']))
                # Attach TAG only after the automatic parent window is fixed.
                ls=by_feature[i];intervals=single_phone_intervals(annotations[row['uid']]['tag_phones'],row['actual'])
                if len(ls)==1 and ls[0].get('a') is not None and not ls[0]['alignment_ambiguous'] and intervals is not None and intervals[ls[0]['a']] is not None:
                    z=ls[0];start,end,token=intervals[z['a']];lo,hi=int(np.ceil(start*16000)),int(np.ceil(end*16000))
                    retained=int(((indices>=lo)&(indices<hi)).sum());before=indices[f['left']:f['right']];after=indices[left:right]
                    oldcount=int(((before>=lo)&(before<hi)).sum());newcount=int(((after>=lo)&(after<hi)).sum())
                    oldstate='present' if oldcount else 'outside' if retained else 'removed_by_VAD'
                    newstate='present' if newcount else 'absent' if anchor is not None else 'unavailable'
                    transitions[(oldstate,newstate)]+=1
                    value.update(state='TAG_support_known',before=oldstate,after=newstate,old_C_samples=oldcount,new_C_samples=newcount,
                                 actual=token,expected=z['expected'],label_kind=z['label_kind'],following=z.get('following'))
                rows.append(value);service.write_line(stream,value)
            counts['unchanged_VAD_inputs']+=1
            if n%100==0:print(n,dict(counts),'seconds',round(time.monotonic()-tick,1),flush=True)
    outside=sum(v for (a,b),v in transitions.items() if a=='outside');present=sum(v for (a,b),v in transitions.items() if a=='present')
    repair=transitions[('outside','present')];loss=sum(v for (a,b),v in transitions.items() if a=='present' and b!='present')
    gate=bool(outside and repair/outside>=.1 and present and loss/present<=.01 and repair>loss)
    report={'scope':p['scope'],'counts':dict(counts),'transitions':{'/'.join(k):v for k,v in transitions.items()},
            'outside_repairs':repair,'old_outside':outside,'present_losses':loss,'old_present':present,
            'TRAIN_diagnostic_gate_passed':gate,'seconds':time.monotonic()-tick,'source_selection_enriched':True,
            'baseline_API_modified':False,'TAG_alignment_input':False,'service_improvement_proven':False}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

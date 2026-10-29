"""Fixed2x2 CAL screen: actual-vowel consistency x coarse-role availability.

Following-vowel context is read from cached runtime canonical, never TAG.
A mismatch means this add-on abstains; it does not mean normal pronunciation,
vowel deletion or an F0 failure. Identical vowels elsewhere remain a limitation.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from train_sibilant_local_encoder import load_encoder,hidden,write
from train_sibilant_cv_support import calibrate
from train_sibilant_joint_vowel import aucs
from mdd.sibilant_cv_support import reliable_support
from mdd.sibilant_joint_vowel import strict_or_joint_support
from mdd.sibilant_vowel_aux import VowelAuxReadout,TOKENS,IDS


def runtime_vowels(features,records):
    raw={r['row']['uid']:r['prior']['raw'] for r in records}
    values=[]
    for f in features:
        canon=raw[f['uid']]['canon'];q=f['q_runtime']
        if canon[q]!=f['canonical'] or q+1>=len(canon) or canon[q+1] not in IDS:raise ValueError('runtime feature context differs')
        values.append(IDS[canon[q+1]])
    return np.array(values,np.int64)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text());torch.set_num_threads(2)
    data=dict(np.load(p['cal_prefix']+'.npz'));features=service.read_jsonl(p['cal_prefix']+'.features.jsonl');records=service.read_jsonl(p['cal_prefix']+'.responses.jsonl')
    if set(data)!={'x','lengths','relative'}:raise ValueError('only acoustic inference cache allowed')
    expected=runtime_vowels(features,records)
    artifact=torch.load(p['artifact'],map_location='cpu')
    if artifact.get('readout')!='vowel_aux' or artifact['vowel_tokens']!=list(TOKENS):raise ValueError('actual-vowel auxiliary artifact required')
    encoder=load_encoder(p,False);encoder.encoder.layers[-2:].load_state_dict(artifact['last2']);head=VowelAuxReadout(encoder.config.hidden_size).to('cuda');head.load_state_dict(artifact['head'])
    probabilities,vowels,strict,joint=[],[],[],[];tick=time.monotonic()
    with torch.no_grad():
        for i in range(0,len(data['x']),64):
            x=torch.as_tensor(data['x'][i:i+64],device='cuda');length=torch.as_tensor(data['lengths'][i:i+64],device='cuda')
            mask=torch.arange(x.shape[1],device='cuda')[None,:]<length[:,None]
            logits,phase,v=head.all_outputs(hidden(encoder,x,mask),mask,torch.as_tensor(data['relative'][i:i+64],device='cuda'))
            probabilities.extend(logits.softmax(-1).cpu().numpy());vowels.extend(v.softmax(-1).cpu().numpy())
            strict.extend(reliable_support(phase,mask).cpu().tolist());joint.extend(strict_or_joint_support(phase,mask).cpu().tolist())
    ps,vp,sa,ja=np.asarray(probabilities),np.asarray(vowels),np.asarray(strict),np.asarray(joint)
    reference=np.load(p['prior_scores'])
    np.testing.assert_array_equal(ps,reference['probabilities']);np.testing.assert_array_equal(vp,reference['vowel_probabilities']);np.testing.assert_array_equal(sa,reference['available'])
    match=vp.argmax(-1)==expected
    np.savez(out/'cal_scores.npz',probabilities=ps,vowel_probabilities=vp,available_strict=sa,available_joint=ja,runtime_vowels=expected,vowel_match=match)
    labels=service.read_jsonl(p['cal_prefix']+'.labels.jsonl');prior={r['row']['uid']:observe(r['prior'],r['row']) for r in records};labels=[dict(z,existing=prior[z['uid']][z['q']]) for z in labels]
    report={'scope':p['scope'],'inference_batch':64,'test_scored':False,'all_CAL_sites':len(labels),'all_CAL_utterances':len(records),'all_CAL_inputs':len(features),'seconds':time.monotonic()-tick,'cells':{}}
    for policy,available in [('strict',sa),('joint',ja)]:
        for binding,selected in [('ungated',available),('vowel_match',available&match)]:
            th,audit=calibrate(labels,features,ps,selected)
            for k in th:
                if k.endswith('/affricate'):th[k]=None
            report['cells'][binding+'/'+policy]={'thresholds':th,'calibration':audit,'available':int(selected.sum()),
                'additional_exact':sum(v['cal_additional_exact'] for k,v in audit.items() if k.endswith('/pair')),
                'conditional_AUC':aucs(labels,ps,selected)}
    choices=[(v['additional_exact'],int(k.endswith('/strict')),int(k.startswith('ungated/')),k) for k,v in report['cells'].items() if v['additional_exact']>=1]
    report['selected_for_single_request']=max(choices)[-1] if choices else None
    write(out/'report.json',report);write(out/'finished.json',{'sha256':service.sha(out/'report.json')});print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

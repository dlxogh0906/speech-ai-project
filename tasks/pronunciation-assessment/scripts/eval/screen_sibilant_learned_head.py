"""Complete-CAL batch screen of a frozen ordinary learned-support head.

Reuse the actual prior-union observations and original calibration rule. This
is a computational screen only; a positive requires batch-one API preflight.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import json
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_context_rank import model
from eval_speech_rescue import observe
from train_sibilant_cv_support import score,calibrate
from train_sibilant_joint_vowel import aucs


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    data=dict(np.load(p['cal_prefix']+'.npz'));features=service.read_jsonl(p['cal_prefix']+'.features.jsonl')
    labels=service.read_jsonl(p['cal_prefix']+'.labels.jsonl');records=service.read_jsonl(p['cal_prefix']+'.responses.jsonl')
    if set(data)!={'x','lengths','relative'}:raise ValueError('TAG inference input')
    torch.set_num_threads(2);artifact=torch.load(p['artifact'],map_location='cpu')
    if artifact.get('readout','existing')!='existing':raise ValueError('ordinary learned-support head required')
    encoder,head=model(p,artifact);ps,av,seconds=score(encoder,head,data)
    np.savez(out/'cal_scores.npz',probabilities=ps,available=av)
    prior={v['row']['uid']:observe(v['prior'],v['row']) for v in records}
    labels=[dict(v,existing=prior[v['uid']][v['q']]) for v in labels]
    report={'scope':p['scope'],'inference_batch':64,'all_CAL_utterances':len(records),'all_CAL_sites':len(labels),'all_features':len(features),'test_scored':False,'arms':{}}
    old=np.load(p['control_scores'])
    for arm,prob,available in [('control',old['probabilities'],old['available_strict']),('candidate',ps,av)]:
        th,audit=calibrate(labels,features,prob,available)
        for k in th:
            if k.endswith('/affricate'):th[k]=None
        report['arms'][arm]={'thresholds':th,'calibration':audit,'additional_exact':sum(v['cal_additional_exact'] for k,v in audit.items() if k.endswith('/pair')),
                             'unconditional_AUC':aucs(labels,prob,np.ones(len(prob),dtype=bool)),'support_available':int(available.sum())}
    report['candidate_seconds']=seconds;report['single_request_preflight_required']=report['arms']['candidate']['additional_exact']>0
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

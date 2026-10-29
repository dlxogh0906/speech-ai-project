"""TRAIN causal diagnostic of restored source samples with one frozen head.

This is not model training or held-out service evaluation. Existing labels are
read for metrics only after scoring both acoustic views. Failed localization
gate is preserved; this diagnostic tests a separate acoustic-recovery claim.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),
              str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import json
import numpy as np
import torch
import eval_sibilant_directional as service
from eval_sibilant_context_rank import model
from train_sibilant_cv_support import score


def summarize(indices, labels, oldps, oldav, newps, newav):
    known=[i for i in indices if int(labels[i])>=0]
    before=sum(oldps[i].argmax()==labels[i] for i in known)
    after=sum(newps[i].argmax()==labels[i] for i in known)
    gains=sum(oldps[i].argmax()!=labels[i] and newps[i].argmax()==labels[i] for i in known)
    losses=sum(oldps[i].argmax()==labels[i] and newps[i].argmax()!=labels[i] for i in known)
    return {'all':len(indices),'known_actual_class':len(known),'old_top1_correct':int(before),'new_top1_correct':int(after),
            'gains':int(gains),'losses':int(losses),'old_role_available':int(oldav[indices].sum()),
            'new_role_available':int(newav[indices].sum()),
            'old_mean_true_class_probability':float(np.mean([oldps[i,labels[i]] for i in known])) if known else None,
            'new_mean_true_class_probability':float(np.mean([newps[i,labels[i]] for i in known])) if known else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();p=json.loads(Path(a.protocol).read_text());out=Path(a.out)
    for f,d in p['sha256'].items():
        if service.sha(f)!=d:raise ValueError('frozen source changed: '+f)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(Path(a.protocol).read_text())
    root=Path(p['source_cache']);new=dict(np.load(root/'data.npz'));old=dict(np.load(p['prefix']+'.npz'))
    features=service.read_jsonl(root/'data.features.jsonl');ix=[f['old_feature_index'] for f in features]
    oldinput={k:old[k][ix] for k in ('x','lengths','relative')};newinput={k:new[k] for k in oldinput}
    torch.set_num_threads(2);artifact=torch.load(p['artifact'],map_location='cpu')
    if artifact.get('readout','existing')!='existing':raise ValueError('ordinary learned support only')
    encoder,head=model(p,artifact)
    op,oa,ot=score(encoder,head,oldinput);np_,na,nt=score(encoder,head,newinput)
    np.savez(out/'scores.npz',old_probabilities=op,old_available=oa,new_probabilities=np_,new_available=na)
    unchanged=np.array([f['unchanged_input'] for f in features])
    np.testing.assert_array_equal(op[unchanged],np_[unchanged]);np.testing.assert_array_equal(oa[unchanged],na[unchanged])
    # Metrics only: attach the original source-label and TAG support metadata.
    labels=service.read_jsonl(root/'data.labels.jsonl');audits=service.read_jsonl(root/'data.support_audit.jsonl')
    byindex=collections.defaultdict(list)
    for z in labels:
        if z.get('feature_index') is not None:byindex[z['feature_index']].append(z)
    restored=[z['feature_index'] for z in audits if z.get('before')=='removed_by_VAD' and z.get('after')=='present']
    pair=[i for i in restored if len(byindex[i])==1 and byindex[i][0]['label_kind']=='pair_error']
    groups={'all':list(range(len(op))),'changed':np.flatnonzero(~unchanged).tolist(),'restored_C':restored,'restored_C_pair_errors':pair}
    stats={k:summarize(v,new['y'],op,oa,np_,na) for k,v in groups.items()}
    passed=(stats['restored_C']['new_top1_correct']-stats['restored_C']['old_top1_correct']>=5
            and stats['restored_C_pair_errors']['new_top1_correct']-stats['restored_C_pair_errors']['old_top1_correct']>=3
            and stats['changed']['gains']>=stats['changed']['losses'])
    result={'scope':p['scope'],'fixed_head_sha256':service.sha(p['artifact']),'inference_batch':64,
            'old_seconds':ot,'new_seconds':nt,'unchanged_inputs_and_predictions_exact':int(unchanged.sum()),
            'groups':stats,'acoustic_recovery_gate_passed':bool(passed),'service_improvement_proven':False,
            'original_localization_gate_remains_failed':True,'TAG_used_at_inference':False}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'finished.json').write_text(json.dumps({'sha256':service.sha(out/'report.json')})+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

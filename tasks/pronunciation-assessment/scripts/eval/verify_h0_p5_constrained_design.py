"""Recount saved TRAIN weights independently of numerical fitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,collections,csv,hashlib,json
import numpy as np
from train_h0_backbone_control import stream,file_sha
from train_w2vbert_ctc import read_split
from mdd.h0_phone_audit import surface_position
from audit_h0_p5_readiness import write


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--p3-protocol',type=Path,required=True);a=ap.parse_args();out=a.out
    p=json.loads(a.p3_protocol.read_text());frozen=json.loads((out/'protocol.json').read_text());report=json.loads((out/'report.json').read_text());manifest=Path(p['manifest'])
    assert file_sha(a.p3_protocol)==frozen['parent_protocol_sha256']
    assert file_sha(frozen['source'])==frozen['source_sha256']
    for file,h in frozen['TRAIN_bindings'].items():assert file_sha(file)==h
    rows,labels=read_split(str(manifest),'train');sources=(manifest/'train.source').read_text().splitlines();co=collections.Counter();draw=collections.Counter();batches=stream(rows,p['batch_samples'],p['seed']);digest=hashlib.sha256();nr=ns=0
    for micro in range(p['stop_step']*p['accum']):
        ids=next(batches);digest.update((' '.join(map(str,ids))+'\n').encode());nr+=len(ids);ns+=sum(rows[i][1] for i in ids)
        if micro>=p['start_step']*p['accum']:
            draw.update(ids)
            for i in ids:co[i]+=1/len(ids)
    assert dict(row_draws=nr,audio_samples=ns,batch_order_sha256=digest.hexdigest())==p['expected_final_exposure']
    weights={arm:np.load(out/(arm+'_weights.npy'),mmap_mode='r') for arm in ('C0','C1','C2')}
    assert np.all(weights['C0']==1)
    aggregate=collections.defaultdict(collections.Counter);context=collections.defaultdict(lambda:np.zeros(2));baseprompt=collections.Counter()
    native_rows=0
    for i,num in draw.items():
        if sources[i]=='native':
            native_rows+=num
            assert all(w[i]==1 for w in weights.values());continue
        parts=Path(rows[i][0]).stem.split('-');item,speaker,l1=parts[5],parts[0],parts[3];baseprompt[sources[i],item]+=co[i]
        cs=collections.Counter((token,surface_position(labels[i],q),labels[i][q-1] if q else '<BOS>',labels[i][q+1] if q+1<len(labels[i]) else '<EOS>') for q,token in enumerate(labels[i]))
        for arm,w in weights.items():
            value=float(w[i]);assert .25-1e-10<=value<=4+1e-10
            for dim,key in [('source',sources[i]),('prompt',sources[i]+'/'+item),('speaker',speaker),('l1',l1)]:aggregate[arm,dim][key]+=co[i]*value
            for c,qty in cs.items():
                mass=co[i]*value*qty/len(labels[i]);aggregate[arm,'phone'][c[0]]+=mass
                context[arm,c]+=np.array([mass,num*value*qty])
    groups=list(csv.DictReader((out/'group_contributions.tsv').open(),delimiter='\t'))
    for r in groups:np.testing.assert_allclose(aggregate[r['arm'],r['dimension']][r['key']],float(r['weighted']),rtol=1e-10,atol=1e-10)
    for r in csv.DictReader((out/'contexts.tsv').open(),delimiter='\t'):
        values=context[r['arm'],(r['phone'],r['position'],r['left'],r['right'])]
        np.testing.assert_allclose(values,[float(r['weighted_proxy']),float(r['weighted_token_draws'])],rtol=1e-10,atol=1e-9)
    for src,mass in aggregate['C0','source'].items():
        for arm in ('C1','C2'):np.testing.assert_allclose(aggregate[arm,'source'][src],mass,rtol=1e-10)
        keys=[k for k in aggregate['C0','prompt'] if k.startswith(src+'/')];cap=1.2*mass/len(keys)
        assert max(aggregate['C2','prompt'][k] for k in keys)<=cap+1e-8
    write(out/'verification.json',dict(status='passed',all_three_saved_weight_arrays_recounted=True,
        independent_group_rows=len(groups),independent_context_rows=len(context),
        exact_continuation_rows=sum(draw.values()),exact_continuation_samples=sum(draw[i]*rows[i][1] for i in draw),native_draws_weight_one=native_rows,
        source_totals_preserved=True,C2_universal_prompt_caps_passed=True,weight_bounds_passed=True,
        weights_sha256={arm:file_sha(out/(arm+'_weights.npy')) for arm in weights},
        report_sha256=file_sha(out/'report.json'),model_optimizer_updates=0,
        distinction='verification of arithmetic and constraints does not mean manipulation gates or model performance passed'))
    print(json.dumps(json.loads((out/'verification.json').read_text())))

if __name__=='__main__':main()

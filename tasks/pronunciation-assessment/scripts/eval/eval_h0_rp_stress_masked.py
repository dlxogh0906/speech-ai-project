"""Add original unknown-time and chunk-boundary masks before RP result inspection.

These annotation times are used only to limit scoring, never to select audio,
produce the model's prediction, fit a threshold, or repair a label.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import eval_h0_rp_stress as base
from mdd.h0_partial_reference import UNKNOWN
from mdd.h0_partial_scoring import partial_ctc_viterbi
from audit_h0_p5_readiness import write,sha
ORIGINAL=base.score_one


def masked_score_one(job):
    r,lpdir,vocab=job;result=ORIGINAL(job)
    unknown=[s['owner'] for s in r['sites'] if s['expected']==UNKNOWN]
    unknown += [owner for token,owner in zip(r['actual'],r['actual_owners']) if token==UNKNOWN]
    data=np.load(Path(lpdir)/(r['uid']+'.npz'));centers=data['centers'];lp=data['lp'];data.close()
    if unknown:
        from mdd.gop_feats_v2 import viterbi_align
        ids=[None if x==UNKNOWN else vocab[x] for x in r['reference']]
        if None in ids:
            aligned=partial_ctc_viterbi(lp,ids,list(vocab.values()));segments=None if aligned is None else aligned['segments']
        else:segments=viterbi_align(lp,ids,blank=0)
        unknown_frames=np.zeros(len(centers),bool)
        for owner in unknown:
            unknown_frames|=(centers>=round(owner['start']*16000)-640)&(centers<=round(owner['end']*16000)+640)
    seams=np.arange(192000,int(centers[-1])+1,192000)
    for q,(s,original) in enumerate(zip(result['sites'],r['sites'])):
        owner=original['owner'];lo=round(owner['start']*16000);hi=round(owner['end']*16000)
        if len(seams):
            distance=int(np.maximum(np.maximum(lo-seams,seams-hi),0).min())
            s['nearest_seam_samples']=min(s['nearest_seam_samples'],distance)
        overlap=bool(unknown and segments is not None and any(unknown_frames[segments[q]]))
        s['source_unknown_time_mask']=overlap
        if overlap:s['unmasked_score']=s['score'];s['score']=None
    for gap in result['gaps']:
        g=gap['g']
        if 0<g<len(result['sites']):
            left,right=result['sites'][g-1:g+1]
            gap['nearest_seam_samples']=min(left['nearest_seam_samples'],right['nearest_seam_samples'])
            if left['score'] is None or right['score'] is None:gap['predicted']=None
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['freeze','score'],required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.mode=='freeze':
        assert not (a.parent/'report.json').exists(),'freeze before viewing RP scores'
        a.out.mkdir(exist_ok=False,parents=True);p=json.loads((a.parent/'protocol.json').read_text())
        p['temporal_mask_extension']=dict(created=time.time(),parent=str(a.parent),
            unknown='GOP support intersecting original LAB or TAG unknown intervals expanded by 40ms is unobserved',
            seams='minimum distance of GOP frames OR original LAB interval to audio-only core seams',
            scope='annotation-time scoring mask only; no input selection, label repair or threshold change',
            model_scores_inspected=False,original_protocol_sha256=sha(a.parent/'protocol.json'))
        p['source_bindings'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
        write(a.out/'protocol.json',p)
        for name in ('annotations.jsonl.gz','annotation_support.json','log_probs','inference.jsonl','extraction_finished.json','runtime_check.json'):(a.out/name).symlink_to(a.parent/name)
        write(a.out/'frozen.json',dict(protocol_sha256=sha(a.out/'protocol.json'),annotations_sha256=sha(a.out/'annotations.jsonl.gz'),model_scored=False))
    else:
        p=json.loads((a.out/'protocol.json').read_text());f=json.loads((a.out/'frozen.json').read_text())
        assert sha(a.out/'protocol.json')==f['protocol_sha256']
        for path,digest in p['source_bindings'].items():assert sha(path)==digest,path
        assert (a.out/'extraction_finished.json').exists()
        base.score_one=masked_score_one;base.score(a,p)

if __name__=='__main__':main()

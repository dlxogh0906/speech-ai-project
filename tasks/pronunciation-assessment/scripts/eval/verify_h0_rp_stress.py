"""Independent scalar recount of the frozen RP secondary stress evaluation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse,collections,gzip,hashlib,itertools,json
import numpy as np
from audit_h0_p5_readiness import write,sha
from mdd.h0_partial_reference import UNKNOWN,DELETED


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    p=json.loads((out/'protocol.json').read_text());f=json.loads((out/'frozen.json').read_text());report=json.loads((out/'report.json').read_text())
    assert sha(out/'protocol.json')==f['protocol_sha256'] and sha(out/'annotations.jsonl.gz')==f['annotations_sha256']
    for path,digest in p['source_bindings'].items():assert sha(path)==digest,path
    for path,digest in p['input_bindings'].items():assert sha(path)==digest,path
    p3=json.loads(Path(p['p3_protocol']).read_text())
    inherited={path:digest for path,digest in p3['bindings'].items() if Path(path).name in
        ('h0_intermediate.py','train_h0_intermediate.py','train_h0_backbone_control.py','train_w2vbert_ctc.py')}
    assert len(inherited)==4
    for path,digest in inherited.items():assert sha(path)==digest,path
    with gzip.open(str(out/'annotations.jsonl.gz'),'rt') as inp:annotations={r['uid']:r for r in map(json.loads,inp)}
    with gzip.open(str(Path(p['parent_partial'])/'annotation_records.jsonl.gz'),'rt') as inp:
        for old in map(json.loads,inp):
            new=annotations[old['uid']]
            assert all(new[k]==v for k,v in old.items())
    ledger={r['uid']:r for r in map(json.loads,(out/'inference.jsonl').open())}
    assert set(ledger)==set(annotations) and len(ledger)==1118
    original_inputs=[json.loads(x) for x in (Path(p['parent_partial'])/'inputs.jsonl').open()]
    counts=collections.defaultdict(collections.Counter);gcounts=collections.defaultdict(collections.Counter);npz_hash=hashlib.sha256();uids=[];frames=samples=windows=0
    ex=json.loads((out/'extraction_finished.json').read_text());inv={v:k for k,v in ex['vocab'].items()}
    with gzip.open(str(out/'evaluated.jsonl.gz'),'rt') as inp:
        for r in map(json.loads,inp):
            uid=r['uid'];uids.append(uid);ann=annotations[uid];led=ledger[uid];file=out/'log_probs'/(uid+'.npz');data=np.load(file)
            lp=data['lp'];centers=data['centers'];data.close();npz_hash.update((uid+':'+sha(file)+'\n').encode())
            assert np.isfinite(lp).all() and len(lp)==led['frames']==len(centers)
            assert all(centers[1:]-centers[:-1]==320) and all((centers-280)%320==0)
            assert centers.min()>=0 and centers.max()<led['audio_samples']
            assert [inv[k] for k,_ in itertools.groupby(lp.argmax(1)) if k in inv]==r['predicted']
            win=led['windows'];assert win[0]['core_start']==0 and win[-1]['core_end']==led['audio_samples']
            assert all(x['core_end']==y['core_start'] for x,y in zip(win,win[1:]))
            assert sum(x['owned_frames'] for x in win)==len(lp)
            frames+=len(lp);samples+=led['audio_samples'];windows+=len(win)
            assert len(r['sites'])==len(ann['sites'])
            for s,old,mask in zip(r['sites'],ann['sites'],ann['annotation_mask']):
                assert s['expected']==old['expected'] and s['actual']==old['identified_actual'] and s['annotation_exclusion']==mask
                for band in (640,0,6400):
                    for policy,t in [('greedy',None),('seen_CAL_fixed',p['threshold'])]:
                        for dim,key in [('all','all'),('item',r['item']),('phone',s['expected'])]:
                            c=counts[band,policy,dim,key];c['known_reference_sites']+=s['expected']!=UNKNOWN;c['reference_entries']+=1
                            if mask:c['excluded_'+mask]+=1;continue
                            err=s['actual']!=s['expected'];kind='error' if err else 'normal';c[kind]+=1;c['eligible_sites']+=1
                            if err:c['deletion' if s['actual']==DELETED else 'substitution']+=1
                            pred=s['prediction_options'];observed=s['score'] is not None and s['nearest_seam_samples']>band and len(pred)==1 and (pred[0]!=DELETED or s['prediction_deletion_anchored'])
                            if not observed:
                                c['unobserved_'+kind]+=1;c['fp_upper']+=not err;c['exact_upper']+=err;continue
                            c['observed_'+kind]+=1
                            warning=pred[0]!=s['expected'] and (t is None or s['score']>=t)
                            fp=warning and not err;exact=warning and err and pred[0]==s['actual']
                            c['fp']+=fp;c['exact']+=exact;c['fp_upper']+=fp;c['exact_upper']+=exact;c['wrong_actual']+=warning and err and not exact
                            if s['actual']==DELETED:c['observed_deletion']+=1;c['deletion_exact']+=exact
            assert len(r['gaps'])==len(ann['truth_gaps'])
            for gap,truth in zip(r['gaps'],ann['truth_gaps']):
                assert gap['actual']==truth
                for band in (640,0,6400):
                    for item in ('all',r['item']):
                        c=gcounts[band,item];c['gap_entries']+=1
                        if truth is None:c['annotation_unknown_gap']+=1;continue
                        err=bool(truth);kind='error' if err else 'normal';c[kind]+=1
                        if gap['predicted'] is None or gap['nearest_seam_samples']<=band:
                            c['unobserved_'+kind]+=1;c['fp_upper']+=not err;c['exact_upper']+=err;continue
                        c['observed_'+kind]+=1;fp=not err and bool(gap['predicted']);exact=err and truth==gap['predicted']
                        c['fp']+=fp;c['exact']+=exact;c['fp_upper']+=fp;c['exact_upper']+=exact
    assert len(uids)==len(set(uids))==1118 and set(uids)==set(annotations)
    compared=0
    for r in report['site_results']:
        key=r['band_samples'],r['policy'],r['dimension'],r['key'];actual=counts[key]
        assert all(actual[k]==r['counts'].get(k,0) for k in set(actual)|set(r['counts'])),key;compared+=1
    for r in report['gap_results']:
        actual=gcounts[r['band_samples'],r['item']]
        assert all(actual[k]==r['counts'].get(k,0) for k in set(actual)|set(r['counts']))
    assert samples==852325678 and windows==len(original_inputs)==4988 and frames==ex['total_frames']
    outer_file=out/'original_denominator_bounds.json';outer_verified=0
    if outer_file.exists():
        denominators=collections.defaultdict(collections.Counter)
        for r in annotations.values():
            for s in r['sites']:
                for key in [('all','all'),('item',r['item']),('phone',s['expected'])]:
                    if s['status']=='normal':denominators[key]['normal']+=1
                    elif s['status'] in ('substitution','deletion'):denominators[key]['error']+=1
        for row in json.loads(outer_file.read_text())['rows']:
            den=denominators[row['dimension'],row['key']];c=counts[row['band_samples'],row['policy'],row['dimension'],row['key']]
            n,e=den['normal'],den['error'];assert row['original_normal']==n and row['original_error']==e
            expected=dict(normal_coverage=c['observed_normal']/n if n else None,error_coverage=c['observed_error']/e if e else None,
                fpr_lower=c['fp']/n if n else None,fpr_upper=(c['fp']+n-c['observed_normal'])/n if n else None,
                exact_lower=c['exact']/e if e else None,exact_upper=(c['exact']+e-c['observed_error'])/e if e else None)
            assert all(row[k]==v for k,v in expected.items());outer_verified+=1
    write(out/'verification.json',dict(status='passed',recordings=1118,windows=windows,audio_samples=samples,frames=frames,
        independent_site_aggregations=compared,independent_gap_aggregations=len(report['gap_results']),
        outer_original_denominator_aggregations=outer_verified,
        all_original_annotation_fields_unchanged=True,all_input_samples_preserved=True,all_saved_greedy_sequences_recomputed=True,
        log_prob_files_sha256=npz_hash.hexdigest(),protocol_sha256=sha(out/'protocol.json'),report_sha256=sha(out/'report.json'),
        inherited_model_source_hashes_match_original_P3=inherited,
        evaluated_sha256=sha(out/'evaluated.jsonl.gz'),model_optimizer_updates=0))
    print(json.dumps(json.loads((out/'verification.json').read_text()),ensure_ascii=False))

if __name__=='__main__':main()

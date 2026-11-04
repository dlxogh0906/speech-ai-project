"""Independently verify output-frame ranks against lossless cached probabilities."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--folder',type=Path,required=True)
    folder=ap.parse_args().folder
    rows=json.loads((folder/'cases.json').read_text())
    utterances=json.loads((folder/'utterances.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    v=json.loads((folder/'vocab.json').read_text());tokens=v['tokens'];blank=v['blank']
    ids=np.array([i for i,p in enumerate(tokens) if len(p)==1 and '\u3131'<=p<='\u3163'])
    by_uid=defaultdict(list)
    for r in rows: by_uid[r['uid']].append(r)
    checked=0
    for u in utterances:
        path=folder/'log_probs'/(u['uid']+'.npz')
        assert sha(path)==u['cache_sha256']
        lp=np.load(path)['log_probs'];pred=lp.argmax(1)
        assert lp.shape==(u['frames'],len(tokens))
        for r in by_uid[u['uid']]:
            if r['status']!='ranked':
                assert r['status']=='deleted_no_emitted_token' and r['predicted'] is None
                continue
            start,end,peak=[r['run'][k] for k in ('start','end','peak')]
            pi=tokens.index(r['predicted']);ai=tokens.index(r['actual'])
            assert np.all(pred[start:end]==pi)
            assert start==0 or pred[start-1]!=pi
            assert end==len(pred) or pred[end]!=pi
            assert peak==start+int(lp[start:end,pi].argmax())
            assert round(float(np.exp(lp[peak,pi])),3)==r['prediction_probability']
            order=ids[np.argsort(-lp[peak,ids],kind='stable')]
            p=np.exp(lp[peak].astype(float))
            assert np.isclose(p.sum(),1,atol=2e-6)
            assert r['actual_rank_jamo']==1+int(np.count_nonzero(lp[peak,ids]>lp[peak,ai]))
            assert r['actual_rank_all_tokens']==1+int(np.count_nonzero(lp[peak]>lp[peak,ai]))
            assert r['actual_tie_count_jamo']==int(np.count_nonzero(lp[peak,ids]==lp[peak,ai]))
            assert r['actual_probability']==float(p[ai]) and r['blank_probability']==float(p[blank])
            assert [z['phone'] for z in r['top_phones']]==[tokens[i] for i in order[:5]]
            assert np.allclose([z['probability'] for z in r['top_phones']],p[order[:5]],rtol=0,atol=0)
            min_rank=1+np.count_nonzero(lp[start:end,ids]>lp[start:end,ai,None],axis=1).min()
            assert r['actual_rank_min_in_run']==min_rank
            checked+=1
    for s in report['summary']:
        a=[r for r in rows if r['actual']==s['phone']];b=[r for r in a if r['status']=='ranked']
        assert len(a)==s['misses']==s['error_n']-s['error_correct']
        assert len(b)==s['ranked']
        assert sum(r['actual_rank_jamo']==2 for r in b)==s['rank2']
        assert sum(r['actual_rank_jamo']<=3 for r in b)==s['rank_le3']
        assert float(np.median([r['actual_probability'] for r in b]))==s['median_actual_probability']
    assert len(rows)==report['target_sites']
    assert all(u['matched'] for u in utterances)
    assert len(utterances)==report['reproduced_recordings']==report['recordings']
    assert dict(Counter(r['status'] for r in rows))==report['status_counts']
    receipt=dict(status='passed',recordings=len(utterances),ranked_sites_checked=checked,
        full_vocabulary=len(tokens),jamo=len(ids),status_counts=report['status_counts'],
        output_sha256={name:sha(folder/name) for name in ['cases.json','utterances.json','report.json','summary.csv','사례별_확률순위.csv']},
        verification_source_sha256=sha(__file__))
    (folder/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__': main()
